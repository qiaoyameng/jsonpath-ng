#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Incremental JSONPath query sessions over a JSON Patch (RFC 6902) stream.

A :class:`QuerySession` pins a JSONPath expression to a document and caches
the matching nodes.  Each batch of JSON Patch operations is validated and
applied in order (a failing batch is rolled back and publishes no partial
results), after which the session reports how the match set changed:
``added``, ``removed``, ``changed`` (value changes) and ``moved`` (pure
position changes caused by list insertion/removal or ``move`` operations).

The declarative subset of JSONPath -- fields, indices, slices, descendants
and extended filters -- is maintained incrementally: only the cached rows
that intersect a patched path are re-evaluated.  Filter condition fields
take part in the read-set analysis, so patching a field that only appears
inside a ``[?(...)]`` predicate still triggers recomputation of the
affected candidates.  Expressions containing nodes that cannot be analyzed
(e.g. ``sorted``/``len``/arithmetic extensions, unions, ``where``) fall
back to a full ``find()`` for affected batches; the fallback is recorded
in ``session.stats['fallback_reasons']`` and counted via
``session.stats['full_requeries']``.
"""

from __future__ import annotations

import copy

from jsonpath_ng.exceptions import JSONPathError, JsonPatchError
from jsonpath_ng.jsonpath import (
    JSONPath,
    Child,
    Descendants,
    Fields,
    Index,
    Intersect,
    Parent,
    Root,
    Slice,
    This,
    Union,
    Where,
    WhereNot,
)
from jsonpath_ng.ext.arithmetic import Operation
from jsonpath_ng.ext.filter import Filter as ExtFilter

__all__ = [
    'JsonPatchError',
    'Match',
    'ValueChange',
    'PositionChange',
    'PatchDiff',
    'QuerySession',
    'datum_path_tuple',
    'path_to_pointer',
]

_MISSING = object()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _escape_segment(segment):
    return str(segment).replace('~', '~0').replace('/', '~1')


def path_to_pointer(path):
    """Render a tuple of path segments as an RFC 6901 JSON pointer."""
    if isinstance(path, str):
        return path
    return ''.join('/' + _escape_segment(segment) for segment in path)


def _parse_pointer(pointer):
    if pointer == '':
        return ()
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise JsonPatchError(f'invalid JSON pointer {pointer!r}')
    return tuple(
        segment.replace('~1', '/').replace('~0', '~')
        for segment in pointer.split('/')[1:]
    )


def datum_path_tuple(datum):
    """Convert the ``full_path`` of a :class:`DatumInContext` to a tuple."""
    segments = []

    def walk(node):
        if isinstance(node, Root):
            return
        if isinstance(node, Child):
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, Fields):
            segments.extend(node.fields)
            return
        if isinstance(node, Index):
            segments.extend(node.indices)
            return
        raise JSONPathError(f'cannot convert path node {node!r} to a tuple')

    walk(datum.full_path)
    return tuple(segments)


def _is_prefix(prefix, path):
    """True if tuple ``prefix`` is a (non-strict) prefix of tuple ``path``."""
    return (isinstance(prefix, tuple) and isinstance(path, tuple)
            and len(prefix) <= len(path) and path[:len(prefix)] == prefix)


def _proper_prefix(prefix, path):
    return _is_prefix(prefix, path) and prefix != path


def _related(path, root):
    return _is_prefix(root, path) or _is_prefix(path, root)


def _json_equal(left, right):
    """JSON-aware equality: ``true`` is distinct from ``1``, unlike ``==``."""
    if left is right:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return (len(left) == len(right)
                and all(key in right and _json_equal(value, right[key])
                        for key, value in left.items()))
    if isinstance(left, list) and isinstance(right, list):
        return (len(left) == len(right)
                and all(_json_equal(a, b) for a, b in zip(left, right)))
    if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
        return False
    return left == right


def _get(document, path, default=_MISSING):
    node = document
    for segment in path:
        if isinstance(node, list):
            if (not isinstance(segment, int) or isinstance(segment, bool)
                    or segment < 0 or segment >= len(node)):
                return default
            node = node[segment]
        elif isinstance(node, dict):
            if not isinstance(segment, str) or segment not in node:
                return default
            node = node[segment]
        else:
            return default
    return node


def _iter_descendants(path, value):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_descendants(path + (key,), child)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_descendants(path + (index,), child)


# ---------------------------------------------------------------------------
# JSON Patch (RFC 6902) application with journaling for atomic batches
# ---------------------------------------------------------------------------

def _to_index(segment, upper):
    if not isinstance(segment, str) or not segment.isdigit():
        raise JsonPatchError(f'invalid array index {segment!r}')
    index = int(segment)
    if index > upper:
        raise JsonPatchError(f'array index {segment!r} out of range')
    return index


def _navigate(document, segments):
    node = document
    for segment in segments:
        if isinstance(node, list):
            node = node[_to_index(segment, len(node) - 1)]
        elif isinstance(node, dict):
            if segment not in node:
                raise JsonPatchError(f'no such object member {segment!r}')
            node = node[segment]
        else:
            raise JsonPatchError(f'cannot descend into scalar value {node!r}')
    return node


def _navigate_typed(document, segments):
    """Navigate to the parent of ``segments``.

    Returns ``(parent, typed_parent_path)`` where list indices in the path
    are converted to integers so they line up with query match paths.
    """
    node = document
    typed = []
    for segment in segments[:-1]:
        if isinstance(node, list):
            index = _to_index(segment, len(node) - 1)
            node = node[index]
            typed.append(index)
        elif isinstance(node, dict):
            if segment not in node:
                raise JsonPatchError(f'no such object member {segment!r}')
            node = node[segment]
            typed.append(segment)
        else:
            raise JsonPatchError(f'cannot descend into scalar value {node!r}')
    return node, tuple(typed)


def _resolve_typed_prefix(document, segments):
    """Type-resolve as much of ``segments`` as currently exists."""
    node = document
    typed = []
    for segment in segments:
        if isinstance(node, list):
            if not isinstance(segment, str) or not segment.isdigit():
                break
            index = int(segment)
            if index >= len(node):
                break
            node = node[index]
            typed.append(index)
        elif isinstance(node, dict):
            if segment not in node:
                break
            node = node[segment]
            typed.append(segment)
        else:
            break
    return tuple(typed)


def _rollback(journal):
    # ``root`` entries need no rollback: the session only swaps its document
    # reference after the whole batch succeeds, so the original root object
    # is never discarded on failure.
    for entry in reversed(journal):
        kind = entry[0]
        if kind == 'list_insert':
            entry[1].pop(entry[2])
        elif kind == 'list_remove':
            entry[1].insert(entry[2], entry[3])
        elif kind == 'list_set':
            entry[1][entry[2]] = entry[3]
        elif kind == 'dict_set':
            _, container, key, existed, old_value = entry
            if existed:
                container[key] = old_value
            else:
                container.pop(key, None)


def _validate_operation(operation):
    if not isinstance(operation, dict):
        raise JsonPatchError(f'patch operation must be an object, got {operation!r}')
    name = operation.get('op')
    if name not in ('add', 'remove', 'replace', 'move', 'copy', 'test'):
        raise JsonPatchError(f'unsupported patch operation {name!r}')
    path = operation.get('path')
    if not isinstance(path, str):
        raise JsonPatchError(f'patch operation {name!r} requires a string "path"')
    normalized = {'op': name, 'path': _parse_pointer(path)}
    if name in ('add', 'replace', 'test'):
        if 'value' not in operation:
            raise JsonPatchError(f'patch operation {name!r} requires "value"')
        normalized['value'] = operation['value']
    if name in ('move', 'copy'):
        source = operation.get('from')
        if not isinstance(source, str):
            raise JsonPatchError(f'patch operation {name!r} requires a string "from"')
        normalized['from'] = _parse_pointer(source)
    return normalized


def _insert_shift(container, index):
    def remap(path):
        if (isinstance(path, tuple) and len(path) > len(container)
                and path[:len(container)] == container):
            current = path[len(container)]
            if isinstance(current, int) and current >= index:
                return container + (current + 1,) + path[len(container) + 1:]
        return path
    return remap


def _remove_shift(container, index):
    def remap(path):
        if (isinstance(path, tuple) and len(path) > len(container)
                and path[:len(container)] == container):
            current = path[len(container)]
            if isinstance(current, int):
                if current == index:
                    return None
                if current > index:
                    return container + (current - 1,) + path[len(container) + 1:]
        return path
    return remap


def _drop_subtree(target):
    def remap(path):
        return None if _is_prefix(target, path) else path
    return remap


def _move_remap(source, target, source_shift, target_shift):
    remove = _remove_shift(*source_shift) if source_shift else None
    insert = _insert_shift(*target_shift) if target_shift else None

    def remap(path):
        if _is_prefix(source, path):
            return target + path[len(source):]
        if remove is not None:
            path = remove(path)
            if path is None:
                return None
        if insert is not None:
            path = insert(path)
        return path
    return remap


def _apply_add(document, segments, value, journal, roots, remaps):
    """RFC 6902 ``add``; returns ``(shift_info, typed_path)``."""
    parent, typed_parent = _navigate_typed(document, segments)
    last = segments[-1]
    if isinstance(parent, list):
        index = len(parent) if last == '-' else _to_index(last, len(parent))
        parent.insert(index, value)
        journal.append(('list_insert', parent, index))
        roots.append(typed_parent)
        remaps.append(_insert_shift(typed_parent, index))
        return (typed_parent, index), typed_parent + (index,)
    if isinstance(parent, dict):
        existed = last in parent
        journal.append(('dict_set', parent, last, existed, parent.get(last)))
        parent[last] = value
        roots.append(typed_parent + (last,))
        return None, typed_parent + (last,)
    raise JsonPatchError(f'cannot add inside scalar value {parent!r}')


def _apply_remove(document, segments, journal, roots, remaps):
    """RFC 6902 ``remove``; returns ``(shift_info, typed_path)``."""
    parent, typed_parent = _navigate_typed(document, segments)
    last = segments[-1]
    if isinstance(parent, list):
        index = _to_index(last, len(parent) - 1)
        old_value = parent.pop(index)
        journal.append(('list_remove', parent, index, old_value))
        roots.append(typed_parent)
        remaps.append(_remove_shift(typed_parent, index))
        return (typed_parent, index), typed_parent + (index,)
    if isinstance(parent, dict):
        if last not in parent:
            raise JsonPatchError(f'no such object member {last!r}')
        old_value = parent.pop(last)
        journal.append(('dict_set', parent, last, True, old_value))
        roots.append(typed_parent + (last,))
        remaps.append(_drop_subtree(typed_parent + (last,)))
        return None, typed_parent + (last,)
    raise JsonPatchError(f'cannot remove inside scalar value {parent!r}')


def _apply_operation(document, operation, journal, roots, remaps):
    """Apply one validated operation, recording journal/remap information.

    Returns the (possibly replaced) document root.
    """
    name = operation['op']
    segments = operation['path']

    if name == 'test':
        value = _navigate(document, segments)
        if not _json_equal(value, operation['value']):
            raise JsonPatchError(
                f'test operation failed at {path_to_pointer(segments)!r}')
        return document

    if name in ('add', 'copy'):
        if name == 'copy':
            value = copy.deepcopy(_navigate(document, operation['from']))
        else:
            value = operation['value']
        if not segments:
            journal.append(('root', document))
            roots.append(())
            return value
        _apply_add(document, segments, value, journal, roots, remaps)
        return document

    if name == 'replace':
        if not segments:
            journal.append(('root', document))
            roots.append(())
            return operation['value']
        parent, typed_parent = _navigate_typed(document, segments)
        last = segments[-1]
        if isinstance(parent, list):
            index = _to_index(last, len(parent) - 1)
            journal.append(('list_set', parent, index, parent[index]))
            parent[index] = operation['value']
            roots.append(typed_parent + (index,))
        elif isinstance(parent, dict):
            if last not in parent:
                raise JsonPatchError(f'no such object member {last!r}')
            journal.append(('dict_set', parent, last, True, parent[last]))
            parent[last] = operation['value']
            roots.append(typed_parent + (last,))
        else:
            raise JsonPatchError(f'cannot replace inside scalar value {parent!r}')
        return document

    if name == 'remove':
        if not segments:
            raise JsonPatchError('cannot remove the document root')
        _apply_remove(document, segments, journal, roots, remaps)
        return document

    # name == 'move'
    source = operation['from']
    if not source:
        raise JsonPatchError('cannot move the document root')
    if _proper_prefix(source, segments):
        raise JsonPatchError('cannot move a value into one of its children')
    value = _navigate(document, source)
    # The inner remove/add record their own remaps; replace them with the
    # single composite move remap below.
    inner_remaps = len(remaps)
    source_shift, typed_source = _apply_remove(document, source, journal, roots, remaps)
    if not segments:
        journal.append(('root', document))
        roots.append(())
        del remaps[inner_remaps:]
        remaps.append(_move_remap(typed_source, (), source_shift, None))
        return value
    target_shift, typed_target = _apply_add(document, segments, value, journal, roots, remaps)
    del remaps[inner_remaps:]
    remaps.append(_move_remap(typed_source, typed_target, source_shift, target_shift))
    return document


# ---------------------------------------------------------------------------
# Query analysis: flattening expressions into declarative pipeline steps
# ---------------------------------------------------------------------------

class _Unanalyzable(Exception):
    def __init__(self, reason, escapes=False):
        super().__init__(reason)
        self.escapes = escapes


class _FieldsStep:
    def __init__(self, names):
        self.names = tuple(names)

    def eval(self, path, value, out):
        if isinstance(value, dict):
            if '*' in self.names:
                for key, child in value.items():
                    out[path + (key,)] = child
            else:
                for name in self.names:
                    if name in value:
                        out[path + (name,)] = value[name]

    def eval_child(self, path, value, segment, out):
        if (isinstance(segment, str) and isinstance(value, dict)
                and segment in value
                and ('*' in self.names or segment in self.names)):
            out[path + (segment,)] = value[segment]


class _IndexStep:
    def __init__(self, indices):
        self.indices = tuple(indices)

    def eval(self, path, value, out):
        if isinstance(value, list) and value:
            length = len(value)
            for index in self.indices:
                if -length <= index < length:
                    out[path + (index % length,)] = value[index % length]

    def eval_child(self, path, value, segment, out):
        if (isinstance(segment, int) and not isinstance(segment, bool)
                and isinstance(value, list) and 0 <= segment < len(value)
                and any(index % len(value) == segment for index in self.indices
                        if -len(value) <= index < len(value))):
            out[path + (segment,)] = value[segment]


class _SliceStep:
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step

    def _indices(self, length):
        return range(*slice(self.start, self.end, self.step).indices(length))

    def eval(self, path, value, out):
        if value is None:
            return
        sequence = value if isinstance(value, list) else [value]
        for index in self._indices(len(sequence)):
            out[path + (index,)] = sequence[index]

    def eval_child(self, path, value, segment, out):
        if isinstance(segment, int) and not isinstance(segment, bool):
            sequence = value if isinstance(value, list) else [value]
            if segment in self._indices(len(sequence)):
                out[path + (segment,)] = sequence[segment]


class _DescendantsStep:
    def eval(self, path, value, out):
        for descendant_path, descendant in _iter_descendants(path, value):
            out[descendant_path] = descendant


class _FilterStep:
    def __init__(self, expressions, dep_chains):
        self.expressions = expressions
        self.dep_chains = dep_chains

    def _passes(self, candidate):
        return all(expression.find(candidate) for expression in self.expressions)

    def eval(self, path, value, out):
        if isinstance(value, dict):
            for key, child in value.items():
                if self._passes(child):
                    out[path + (key,)] = child
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if self._passes(child):
                    out[path + (index,)] = child

    def eval_child(self, path, value, segment, out):
        if isinstance(value, dict) and isinstance(segment, str) and segment in value:
            if self._passes(value[segment]):
                out[path + (segment,)] = value[segment]
        elif (isinstance(value, list) and isinstance(segment, int)
                and not isinstance(segment, bool) and 0 <= segment < len(value)):
            if self._passes(value[segment]):
                out[path + (segment,)] = value[segment]


def _dep_chains(node):
    """Relative read paths of a filter target, rooted at the candidate."""
    if isinstance(node, This):
        return [()]
    if isinstance(node, Fields):
        return [tuple(node.fields)]
    if isinstance(node, Index):
        return [tuple(index if isinstance(index, int) and index >= 0 else '*'
                      for index in node.indices)]
    if isinstance(node, Child):
        return [left + right
                for left in _dep_chains(node.left)
                for right in _dep_chains(node.right)]
    if isinstance(node, Operation):
        chains = []
        for side in (node.left, node.right):
            if isinstance(side, JSONPath):
                chains.extend(_dep_chains(side))
        return chains
    if isinstance(node, Root):
        raise _Unanalyzable(f'absolute reference {node!r} in filter', escapes=True)
    raise _Unanalyzable(f'unsupported filter target {node!r}')


def _flatten_steps(expression):
    """Flatten an expression into declarative steps.

    Returns ``(steps, error)``; on failure ``steps`` holds the analyzable
    prefix and ``error`` the :class:`_Unanalyzable` that stopped analysis.
    """
    steps = []
    error = None

    def walk(node):
        if isinstance(node, Root):
            return
        if isinstance(node, Child):
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, Fields):
            steps.append(_FieldsStep(node.fields))
            return
        if isinstance(node, Index):
            steps.append(_IndexStep(node.indices))
            return
        if isinstance(node, Slice):
            steps.append(_SliceStep(node.start, node.end, node.step))
            return
        if isinstance(node, Descendants):
            walk(node.left)
            steps.append(_DescendantsStep())
            walk(node.right)
            return
        if isinstance(node, ExtFilter):
            dep_chains = []
            for expression in node.expressions:
                target = getattr(expression, 'target', None)
                if target is None or isinstance(expression.value, JSONPath):
                    raise _Unanalyzable(f'unsupported filter expression {expression!r}')
                dep_chains.extend(_dep_chains(target))
            steps.append(_FilterStep(node.expressions, dep_chains))
            return
        escapes = isinstance(node, (Parent, Where, WhereNot, Operation, Union, Intersect))
        raise _Unanalyzable(f'unsupported node {node!r}', escapes=escapes)

    try:
        walk(expression)
    except _Unanalyzable as exc:
        error = exc
    return steps, error


def _read_patterns(steps):
    """Static read-set patterns: tuples of segments, ``*`` and ``**``."""
    current = [()]
    extra = []
    for step in steps:
        next_current = []
        for pattern in current:
            if pattern and pattern[-1] == '**':
                next_current.append(pattern)
                continue
            if isinstance(step, _FieldsStep):
                next_current.extend(pattern + (name,) for name in step.names)
            elif isinstance(step, _IndexStep):
                next_current.extend(
                    pattern + (index if isinstance(index, int) and index >= 0 else '*',)
                    for index in step.indices)
            elif isinstance(step, _SliceStep):
                next_current.append(pattern + ('*',))
            elif isinstance(step, _FilterStep):
                next_current.append(pattern + ('*',))
                # Predicate dependencies are read but are not positions the
                # pipeline passes through, so they do not extend `current`.
                extra.extend(pattern + ('*',) + chain for chain in step.dep_chains)
            elif isinstance(step, _DescendantsStep):
                next_current.append(pattern + ('**',))
        current = next_current
    return current + extra


def _pattern_intersects(pattern, path):
    """True if some concrete path matching ``pattern`` shares a prefix
    relation with ``path`` (either direction)."""
    for pattern_segment, path_segment in zip(pattern, path):
        if pattern_segment == '**':
            return True
        if pattern_segment == '*':
            continue
        if pattern_segment != path_segment:
            return False
    return True


# ---------------------------------------------------------------------------
# Match diffing
# ---------------------------------------------------------------------------

class Match:
    """A single query match: a tuple ``path`` and the matched ``value``."""
    __slots__ = ('path', 'value')

    def __init__(self, path, value):
        self.path = path
        self.value = value

    @property
    def pointer(self):
        return path_to_pointer(self.path)

    def __repr__(self):
        return f'Match({self.pointer!r}, {self.value!r})'

    def __eq__(self, other):
        return (isinstance(other, Match) and self.path == other.path
                and _json_equal(self.value, other.value))


class ValueChange:
    """A match whose value changed; ``old_path`` differs from ``path`` when
    the node also moved."""
    __slots__ = ('path', 'old_value', 'new_value', 'old_path')

    def __init__(self, path, old_value, new_value, old_path=None):
        self.path = path
        self.old_value = old_value
        self.new_value = new_value
        self.old_path = old_path if old_path is not None else path

    @property
    def pointer(self):
        return path_to_pointer(self.path)

    @property
    def old_pointer(self):
        return path_to_pointer(self.old_path)

    def __repr__(self):
        return f'ValueChange({self.pointer!r}, {self.old_value!r} -> {self.new_value!r})'


class PositionChange:
    """A match that kept its value but changed position (list insertion,
    removal or ``move``)."""
    __slots__ = ('old_path', 'new_path', 'value')

    def __init__(self, old_path, new_path, value):
        self.old_path = old_path
        self.new_path = new_path
        self.value = value

    @property
    def path(self):
        return self.new_path

    @property
    def old_pointer(self):
        return path_to_pointer(self.old_path)

    @property
    def new_pointer(self):
        return path_to_pointer(self.new_path)

    @property
    def pointer(self):
        return self.new_pointer

    def __repr__(self):
        return f'PositionChange({self.old_pointer!r} -> {self.new_pointer!r})'


class PatchDiff:
    """The match-set delta produced by :meth:`QuerySession.apply_patch`."""
    __slots__ = ('added', 'removed', 'changed', 'moved')

    def __init__(self, added=(), removed=(), changed=(), moved=()):
        self.added = list(added)
        self.removed = list(removed)
        self.changed = list(changed)
        self.moved = list(moved)

    @property
    def empty(self):
        return not (self.added or self.removed or self.changed or self.moved)

    def __bool__(self):
        return not self.empty

    def __repr__(self):
        return (f'PatchDiff(added={self.added!r}, removed={self.removed!r}, '
                f'changed={self.changed!r}, moved={self.moved!r})')


def _compose_remaps(remaps):
    if not remaps:
        return lambda path: path

    def remap(path):
        for remap_op in remaps:
            path = remap_op(path)
            if path is None:
                return None
        return path
    return remap


def _compute_diff(old, new, remap, snapshots):
    added, removed, changed, moved = [], [], [], []
    remapped = {}
    for path, value in old.items():
        new_path = remap(path)
        if new_path is not None:
            remapped[new_path] = (path, snapshots.get(path, value))
    for path, value in new.items():
        if path in remapped:
            old_path, old_value = remapped[path]
            if _json_equal(old_value, value):
                if old_path != path:
                    moved.append(PositionChange(old_path, path, value))
            else:
                changed.append(ValueChange(path, old_value, value, old_path))
        else:
            added.append(Match(path, value))
    for path, value in old.items():
        new_path = remap(path)
        if new_path is None or new_path not in new:
            removed.append(Match(path, snapshots.get(path, value)))
    return PatchDiff(added, removed, changed, moved)


# ---------------------------------------------------------------------------
# The query session
# ---------------------------------------------------------------------------

class QuerySession:
    """A long-lived JSONPath query over a document patched via JSON Patch.

    ``expression`` may be a JSONPath string (parsed with the extended
    parser, so filters are supported) or an already parsed
    :class:`JSONPath`.  ``document`` is owned by the session and mutated in
    place as patches are applied.
    """

    def __init__(self, expression, document):
        if isinstance(expression, str):
            from jsonpath_ng.ext import parse as _ext_parse
            expression = _ext_parse(expression)
        self._expression = expression
        self._document = document
        self.stats = {
            'batches_applied': 0,
            'batches_failed': 0,
            'operations_applied': 0,
            'batches_skipped': 0,
            'batches_incremental': 0,
            'full_requeries': 0,
            'fallback_reasons': [],
        }
        steps, error = _flatten_steps(expression)
        self._patterns = _read_patterns(steps)
        if error is not None:
            self._steps = None
            if error.escapes:
                self._patterns = [('**',)]
            else:
                self._patterns = [
                    pattern if pattern and pattern[-1] == '**' else pattern + ('**',)
                    for pattern in self._patterns
                ]
            self.stats['fallback_reasons'].append(str(error))
        else:
            self._steps = steps
        if self._steps is None:
            self._rows = None
            self._matches = self._find_all()
        else:
            self._rows = [{(): document}]
            for step in self._steps:
                self._rows.append(self._eval_step(step, self._rows[-1]))
            self._matches = self._rows[-1]

    @property
    def document(self):
        return self._document

    @property
    def expression(self):
        return self._expression

    @property
    def fallback(self):
        """True if the expression could not be fully analyzed, so affected
        batches fall back to a full query (see ``stats['fallback_reasons']``)."""
        return self._steps is None

    @property
    def matches(self):
        return [Match(path, value) for path, value in self._matches.items()]

    # -- evaluation ------------------------------------------------------

    @staticmethod
    def _eval_step(step, rows):
        out = {}
        for path, value in rows.items():
            step.eval(path, value, out)
        return out

    def _find_all(self):
        matches = {}
        for datum in self._expression.find(self._document):
            try:
                path = datum_path_tuple(datum)
            except JSONPathError:
                # Opaque path (e.g. produced by ``sorted``/``len``): keep a
                # stable string key so diffs remain consistent.
                path = str(datum.full_path)
            matches[path] = datum.value
        return matches

    def _update_rows(self, root):
        """Incrementally re-derive cached rows affected by a write at ``root``."""
        document = self._document
        old_input_paths = [set(rows) for rows in self._rows]
        self._rows[0] = {(): document}
        for index, step in enumerate(self._steps):
            inputs = self._rows[index]
            old_out = self._rows[index + 1]
            existing = old_input_paths[index]
            if isinstance(step, _DescendantsStep):
                new_out = {path: value for path, value in old_out.items()
                           if not _is_prefix(root, path)
                           and any(path[:depth] in inputs
                                   for depth in range(len(path) + 1))}
                node = _get(document, root, _MISSING)
                if node is not _MISSING:
                    for path, value in _iter_descendants(root, node):
                        if any(path[:depth] in inputs
                               for depth in range(len(path) + 1)):
                            new_out[path] = value
            elif isinstance(step, _FilterStep):
                # A write inside a candidate may flip its predicate even
                # though the candidate path itself is unchanged.
                stale = {root[:len(path) + 1] for path in inputs
                         if path in existing and _proper_prefix(path, root)}
                new_out = {path: value for path, value in old_out.items()
                           if not _is_prefix(root, path) and path not in stale
                           and path[:-1] in inputs}
                for path, value in inputs.items():
                    if path not in existing or _is_prefix(root, path):
                        step.eval(path, value, new_out)
                    elif _proper_prefix(path, root):
                        step.eval_child(path, value, root[len(path)], new_out)
            else:
                new_out = {path: value for path, value in old_out.items()
                           if not _is_prefix(root, path)
                           and path[:-1] in inputs}
                parent = root[:-1]
                for path, value in inputs.items():
                    if path not in existing or _is_prefix(root, path):
                        step.eval(path, value, new_out)
                    elif root and path == parent:
                        step.eval_child(path, value, root[-1], new_out)
            self._rows[index + 1] = new_out

    # -- patching ---------------------------------------------------------

    @staticmethod
    def _conservative_scopes(document, operation):
        """Pre-application write scopes (superset of the real invalidation
        roots) used to snapshot old match values."""
        name = operation['op']
        scopes = []
        if name in ('add', 'remove', 'replace'):
            scopes.append(_resolve_typed_prefix(document, operation['path'][:-1]))
        elif name in ('move', 'copy'):
            scopes.append(_resolve_typed_prefix(document, operation['from'][:-1]))
            scopes.append(_resolve_typed_prefix(document, operation['path'][:-1]))
        return scopes

    def apply_patch(self, operations):
        """Validate and apply a batch of JSON Patch operations, in order.

        On success returns a :class:`PatchDiff` describing how the match
        set changed.  If any operation is invalid, the batch is rolled
        back, no partial results are published, and ``JsonPatchError`` is
        raised.
        """
        if isinstance(operations, dict):
            operations = [operations]
        try:
            normalized = [_validate_operation(operation) for operation in operations]
        except JsonPatchError:
            self.stats['batches_failed'] += 1
            raise

        # Snapshot the old values of matches that may be affected, before
        # any mutation happens (containers are patched in place).
        scopes = [
            scope
            for operation in normalized
            for scope in self._conservative_scopes(self._document, operation)
        ]
        snapshots = {}
        for path, value in self._matches.items():
            if any(_related(path, scope) for scope in scopes):
                snapshots[path] = copy.deepcopy(value)

        journal = []
        roots = []
        remaps = []
        document = self._document
        try:
            for operation in normalized:
                document = _apply_operation(document, operation, journal, roots, remaps)
        except JsonPatchError:
            _rollback(journal)
            self.stats['batches_failed'] += 1
            raise
        self._document = document
        self.stats['batches_applied'] += 1
        self.stats['operations_applied'] += len(normalized)

        relevant = any(
            _pattern_intersects(pattern, root)
            for pattern in self._patterns
            for root in roots
        )
        if not relevant:
            self.stats['batches_skipped'] += 1
            return PatchDiff()

        old_matches = self._matches
        remap = _compose_remaps(remaps)
        if self._steps is None:
            self.stats['full_requeries'] += 1
            new_matches = self._find_all()
        else:
            for root in roots:
                self._update_rows(root)
            new_matches = self._rows[-1]
            self.stats['batches_incremental'] += 1
        self._matches = new_matches
        return _compute_diff(old_matches, new_matches, remap, snapshots)
