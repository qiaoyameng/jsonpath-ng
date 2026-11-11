"""Incremental JSONPath subscriptions.

A :class:`SubscriptionDocument` wraps a JSON document together with a
monotonically increasing version.  A bounded number of
:class:`SubscriptionSession` objects can be opened against a document; each
session registers compiled JSONPath expressions and is notified of every
document change through a versioned ``add`` / ``remove`` / ``replace`` /
``move`` patch that describes how the expression's hit set changed.

First-version scope
-------------------

The incremental engine maintains the matched hit set locally for expressions
built from the following constructs:

* object fields (``$.store.book``),
* array indices and wildcards (``$.items[0]``, ``$.items[*]``),
* array slices (``$.items[0:3]``),
* filters with explicit scalar comparisons
  (``$.items[?price > 10]``), including filters that reference the root
  identifier (``$.items[?price < $.limit]``).

Expressions containing constructs that cannot be maintained locally
(recursive descent, unions, ``where`` clauses, ...) are still supported: the
subscription is marked for an explicit full recomputation on every change.
The recomputed hit set is diffed in exactly the same way, so subscribers see
a uniform patch format.

Atomicity
---------

Every change is submitted as a :class:`Patch` carrying the base version it was
produced from.  Applying a patch whose base version does not match the
document version, or a patch containing an illegal path, leaves the document
*and every registered subscription* in their previous state.

Access counting
---------------

The document exposes a traversal proxy used by full recomputations.  Only the
containers actually visited while evaluating an expression are counted, which
makes it possible to prove that unaffected branches of the tree are never
touched by a localized update.
"""

from __future__ import annotations

import copy
import typing as t

from .exceptions import JSONPathError
from .jsonpath import Child
from .jsonpath import DatumInContext
from .jsonpath import Fields
from .jsonpath import Index
from .jsonpath import JSONPath
from .jsonpath import Root
from .jsonpath import Slice
from .jsonpath import This
from .jsonpath import Wildcard
from .ext.filter import Filter

#
# Paths are represented as tuples of concrete keys: strings for object members
# and non-negative integers for array positions.  The empty tuple is the root.
#
Path: t.TypeAlias = tuple[t.Any, ...]


class SubscriptionError(JSONPathError):
    """Base class for subscription related errors."""


class PatchVersionError(SubscriptionError):
    """Raised when a patch's base version does not match the document version."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"Patch base version {expected} does not match document version {actual}"
        )
        self.expected = expected
        self.actual = actual


class IllegalPatchPathError(SubscriptionError):
    """Raised when a patch operation addresses a path that does not exist."""


@t.final
class PatchOperation:
    """A single change within a :class:`Patch`.

    The ``op`` attribute is one of ``add``, ``remove``, ``replace`` or
    ``move``.  For ``move`` operations, ``from_path`` records the position the
    value moved from; otherwise it is ``None``.
    """

    __slots__ = ("op", "path", "value", "from_path")

    def __init__(
        self,
        op: str,
        path: Path,
        value: t.Any = None,
        from_path: Path | None = None,
    ) -> None:
        self.op = op
        self.path = path
        self.value = value
        self.from_path = from_path

    def __eq__(self, other: t.Any) -> bool:
        if not isinstance(other, PatchOperation):
            return NotImplemented
        return (
            self.op == other.op
            and self.path == other.path
            and _json_equal(self.value, other.value)
            and self.from_path == other.from_path
        )

    def __repr__(self) -> str:
        if self.from_path is not None:
            return (
                f"PatchOperation({self.op!r}, path={self.path!r}, "
                f"from_path={self.from_path!r}, value={self.value!r})"
            )
        return f"PatchOperation({self.op!r}, path={self.path!r}, value={self.value!r})"


@t.final
class Patch:
    """A versioned set of changes to apply to a :class:`SubscriptionDocument`."""

    __slots__ = ("base_version", "operations")

    def __init__(
        self, base_version: int, operations: t.Iterable[PatchOperation]
    ) -> None:
        self.base_version = base_version
        self.operations = list(operations)

    def __repr__(self) -> str:
        return (
            f"Patch(base_version={self.base_version}, "
            f"operations={self.operations!r})"
        )


def _json_equal(left: t.Any, right: t.Any) -> bool:
    """Compare two JSON values, treating booleans as distinct from integers."""

    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(value, right[key]) for key, value in left.items()
        )
    return left == right


class _CountingDict(dict):
    """A dict that reports member accesses to the owning document."""

    __slots__ = ("_document", "_path")

    def __init__(
        self, document: SubscriptionDocument, path: Path, values: dict
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_document", document)
        object.__setattr__(self, "_path", path)
        for key, value in values.items():
            dict.__setitem__(self, key, _wrap(document, path + (key,), value))

    def __getitem__(self, key: t.Any) -> t.Any:
        self._document._record_access(self._path)
        return dict.__getitem__(self, key)

    def get(self, key: t.Any, default: t.Any = None) -> t.Any:
        self._document._record_access(self._path)
        return dict.get(self, key, default)

    def items(self) -> t.Any:
        self._document._record_access(self._path)
        return dict.items(self)

    def keys(self) -> t.Any:
        self._document._record_access(self._path)
        return dict.keys(self)

    def values(self) -> t.Any:
        self._document._record_access(self._path)
        return dict.values(self)

    def __contains__(self, key: t.Any) -> bool:
        self._document._record_access(self._path)
        return dict.__contains__(self, key)

    def __iter__(self) -> t.Any:
        self._document._record_access(self._path)
        return dict.__iter__(self)


class _CountingList(list):
    """A list that reports element and length accesses to the owning document."""

    __slots__ = ("_document", "_path")

    def __init__(
        self, document: SubscriptionDocument, path: Path, values: list
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_document", document)
        object.__setattr__(self, "_path", path)
        for index, value in enumerate(values):
            list.append(self, _wrap(document, path + (index,), value))

    def _touch(self) -> None:
        self._document._record_access(self._path)

    def __getitem__(self, index: t.Any) -> t.Any:
        self._touch()
        return list.__getitem__(self, index)

    def __iter__(self) -> t.Any:
        self._touch()
        return list.__iter__(self)

    def __len__(self) -> int:
        self._touch()
        return list.__len__(self)

    def __contains__(self, value: t.Any) -> bool:
        self._touch()
        return list.__contains__(self, value)


def _wrap(document: SubscriptionDocument, path: Path, value: t.Any) -> t.Any:
    """Recursively wrap raw JSON containers with counting proxies."""

    if isinstance(value, bool) or not isinstance(value, (dict, list)):
        return value
    if isinstance(value, _CountingDict) or isinstance(value, _CountingList):
        return value
    if isinstance(value, dict):
        return _CountingDict(document, path, value)
    return _CountingList(document, path, value)


def _parent_and_key(
    root: t.Any, path: Path
) -> tuple[t.Any, t.Any]:
    """Return the parent container and final key for *path*.

    Raises :class:`IllegalPatchPathError` when any intermediate node is
    missing or of the wrong container type.
    """

    if not path:
        msg = "The document root cannot be addressed by a patch operation"
        raise IllegalPatchPathError(msg)
    current = root
    for key in path[:-1]:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise IllegalPatchPathError(
                f"Intermediate path {path} does not exist"
            ) from exc
    return current, path[-1]


def _apply_operation(root: t.Any, operation: PatchOperation) -> None:
    """Apply a single operation to the raw (unproxied) document tree."""

    op = operation.op
    if op == "move":
        if operation.from_path is None:
            msg = "A move operation requires a from_path"
            raise IllegalPatchPathError(msg)
        destination = operation.path
        from_parent, from_key = _parent_and_key(root, operation.from_path)
        to_parent, to_key = _parent_and_key(root, destination)
        # RFC 6902: the destination is evaluated after removing the source,
        # so an array index at or after the removed element shifts down by one.
        if (
            isinstance(from_parent, list)
            and from_parent is to_parent
            and isinstance(from_key, int)
            and isinstance(to_key, int)
            and to_key > from_key
        ):
            destination = destination[:-1] + (to_key - 1,)
        value = _remove_path(root, operation.from_path)
        _add_value(root, destination, value)
        return

    parent, key = _parent_and_key(root, operation.path)
    if op == "add":
        _add_to_parent(parent, key, copy.deepcopy(operation.value))
    elif op == "replace":
        if isinstance(parent, list):
            if not -len(parent) <= key < len(parent):
                raise IllegalPatchPathError(
                    f"Array index {key} does not exist at {operation.path}"
                )
        elif key not in parent:
            raise IllegalPatchPathError(
                f"Object member {operation.path!r} does not exist"
            )
        parent[key] = copy.deepcopy(operation.value)
    elif op == "remove":
        if isinstance(parent, list):
            if not -len(parent) <= key < len(parent):
                raise IllegalPatchPathError(
                    f"Array index {key} does not exist at {operation.path}"
                )
            del parent[key]
        else:
            try:
                del parent[key]
            except KeyError as exc:
                raise IllegalPatchPathError(
                    f"Object member {operation.path!r} does not exist"
                ) from exc
    else:
        msg = f"Unknown patch operation {op!r}"
        raise IllegalPatchPathError(msg)


def _add_value(root: t.Any, path: Path, value: t.Any) -> None:
    parent, key = _parent_and_key(root, path)
    _add_to_parent(parent, key, value)


def _add_to_parent(parent: t.Any, key: t.Any, value: t.Any) -> None:
    if isinstance(parent, list):
        if key == "-":
            parent.append(value)
            return
        if not isinstance(key, int) or not -len(parent) <= key <= len(parent):
            raise IllegalPatchPathError(
                f"Array insertion index {key!r} is out of bounds"
            )
        parent.insert(key, value)
    else:
        if not isinstance(parent, dict) or not isinstance(key, str):
            raise IllegalPatchPathError(
                f"Member name {key!r} is not valid for this container"
            )
        parent[key] = value


def _remove_path(root: t.Any, path: Path) -> t.Any:
    parent, key = _parent_and_key(root, path)
    if isinstance(parent, list):
        if not -len(parent) <= key < len(parent):
            raise IllegalPatchPathError(
                f"Array index {key} does not exist at {path}"
            )
        value = parent[key]
        del parent[key]
        return value
    try:
        value = parent[key]
    except KeyError as exc:
        raise IllegalPatchPathError(
            f"Object member {path!r} does not exist"
        ) from exc
    del parent[key]
    return value


class SubscriptionDocument:
    """A versioned JSON document that feeds subscription sessions.

    The document holds the raw JSON tree and increments :attr:`version` each
    time a :class:`Patch` is successfully applied.  Subscriptions only ever
    traverse the document through the counting proxy returned by
    :meth:`proxy`, so the number of visited containers can be measured after
    each update.
    """

    def __init__(self, data: t.Any) -> None:
        self._data = data
        self._version = 0
        self._sessions: list[SubscriptionSession] = []
        self._access_count = 0
        self._closed = False

    @property
    def version(self) -> int:
        return self._version

    @property
    def data(self) -> t.Any:
        """The raw JSON tree (not wrapped in a counting proxy)."""

        return self._data

    @property
    def access_count(self) -> int:
        """Number of container visits recorded since the last reset."""

        return self._access_count

    def reset_access_count(self) -> None:
        self._access_count = 0

    def _record_access(self, path: Path) -> None:
        self._access_count += 1

    def proxy(self) -> t.Any:
        """Return a counting proxy over a deep copy of the current tree."""

        return _wrap(self, (), copy.deepcopy(self._data))

    def open_session(self) -> SubscriptionSession:
        if self._closed:
            msg = "Cannot open a session on a closed document"
            raise SubscriptionError(msg)
        session = SubscriptionSession(self)
        self._sessions.append(session)
        return session

    def _detach_session(self, session: SubscriptionSession) -> None:
        try:
            self._sessions.remove(session)
        except ValueError:
            pass

    def close(self) -> None:
        """Close every session and drop references to the document tree."""

        for session in list(self._sessions):
            session.close()
        self._closed = True
        self._data = None
        self._sessions.clear()

    def apply_patch(self, patch: Patch) -> list[SubscriptionPatch]:
        """Apply *patch* and update every subscription atomically.

        Returns the :class:`SubscriptionPatch` produced for each
        subscription, in registration order.  On failure the document and all
        subscriptions are left untouched.
        """

        if self._closed:
            msg = "Cannot apply a patch to a closed document"
            raise SubscriptionError(msg)
        if patch.base_version != self._version:
            raise PatchVersionError(patch.base_version, self._version)

        candidate = copy.deepcopy(self._data)
        for operation in patch.operations:
            _apply_operation(candidate, operation)

        changes = [
            subscription._refresh(
                self,
                candidate,
                patch.operations,
                self._version + 1,
            )
            for session in self._sessions
            for subscription in session._subscriptions
        ]

        self._data = candidate
        self._version += 1
        return changes


@t.final
class SubscriptionPatch:
    """The change of one subscription's hit set between two versions."""

    __slots__ = ("expression", "from_version", "to_version", "operations")

    def __init__(
        self,
        expression: JSONPath,
        from_version: int,
        to_version: int,
        operations: list[PatchOperation],
    ) -> None:
        self.expression = expression
        self.from_version = from_version
        self.to_version = to_version
        self.operations = operations

    @property
    def added(self) -> list[PatchOperation]:
        return [op for op in self.operations if op.op == "add"]

    @property
    def removed(self) -> list[PatchOperation]:
        return [op for op in self.operations if op.op == "remove"]

    @property
    def replaced(self) -> list[PatchOperation]:
        return [op for op in self.operations if op.op == "replace"]

    @property
    def moved(self) -> list[PatchOperation]:
        return [op for op in self.operations if op.op == "move"]

    def __repr__(self) -> str:
        return (
            f"SubscriptionPatch({self.expression!r}, "
            f"{self.from_version}->{self.to_version}, {self.operations!r})"
        )


class SubscriptionSession:
    """A bounded-lifetime grouping of subscriptions on one document."""

    def __init__(self, document: SubscriptionDocument) -> None:
        self._document: SubscriptionDocument | None = document
        self._subscriptions: list[Subscription] = []
        self._closed = False

    @property
    def document(self) -> SubscriptionDocument | None:
        return self._document

    @property
    def closed(self) -> bool:
        return self._closed

    def register(self, expression: JSONPath | str) -> Subscription:
        """Register a JSONPath expression and return its subscription."""

        if self._closed or self._document is None:
            msg = "Cannot register on a closed session"
            raise SubscriptionError(msg)
        if isinstance(expression, str):
            from .ext import parse

            expression = parse(expression)
        subscription = Subscription(self._document, expression)
        self._subscriptions.append(subscription)
        return subscription

    def close(self) -> None:
        """Detach from the document and release every subscription."""

        if self._closed:
            return
        self._closed = True
        if self._document is not None:
            self._document._detach_session(self)
        self._document = None
        for subscription in self._subscriptions:
            subscription._release()
        self._subscriptions.clear()


#
# Dependency records
# ==================
#
# A field/index dependency records a concrete requirement at a fixed path.
# A wildcard/slice/filter dependency records that a whole subtree rooted at a
# path is consumed, so any structural change inside it invalidates the match.
#
@t.final
class _Dependency:
    __slots__ = ("path", "kind", "root_paths", "node")

    def __init__(
        self,
        path: Path,
        kind: str,
        node: JSONPath,
        root_paths: t.FrozenSet[Path] = frozenset(),
    ) -> None:
        # kind is one of: "field", "index", "wildcard", "slice", "filter".
        self.path = path
        self.kind = kind
        self.node = node
        # Absolute paths referenced through the root identifier inside a filter.
        self.root_paths = root_paths


class _Dependencies(t.NamedTuple):
    records: list[_Dependency]
    maintainable: bool


def _analyze(expression: JSONPath) -> _Dependencies:
    """Walk an expression AST and record its structural dependencies."""

    records: list[_Dependency] = []
    state = {"maintainable": True}

    def visit(node: JSONPath, base: Path) -> None:
        if isinstance(node, Child):
            # The left side locates the parent context; the right side is
            # evaluated against it and therefore extends the concrete path.
            visit(node.left, base)
            visit(node.right, _extend_base(node.left, base))
        elif isinstance(node, Root) or isinstance(node, This):
            return
        elif isinstance(node, Fields):
            if node._deprecated_wildcard_use:
                records.append(_Dependency(base, "wildcard", node))
            else:
                for field in node.fields:
                    records.append(_Dependency(base + (field,), "field", node))
        elif isinstance(node, Index):
            for index in node.indices:
                records.append(_Dependency(base + (index,), "index", node))
        elif isinstance(node, Wildcard):
            records.append(_Dependency(base, "wildcard", node))
        elif isinstance(node, Slice):
            records.append(_Dependency(base, "slice", node))
        elif isinstance(node, Filter):
            root_paths = _filter_root_paths(node)
            records.append(
                _Dependency(base, "filter", node, frozenset(root_paths))
            )
        else:
            # Descendants, Where, unions, arithmetic and other constructs may
            # visit arbitrary branches; maintain them by full recomputation.
            state["maintainable"] = False

    visit(expression, ())
    return _Dependencies(records, state["maintainable"])


def _extend_base(node: JSONPath, base: Path) -> Path:
    """Return the concrete path reached after navigating *node* from *base*."""

    if isinstance(node, Child):
        return _extend_base(node.right, _extend_base(node.left, base))
    if isinstance(node, Root) or isinstance(node, This):
        return base
    if isinstance(node, Fields):
        return base + node.fields[:1]
    if isinstance(node, Index):
        return base + node.indices[:1]
    # Wildcards, slices and filters do not add a static segment.
    return base


def _filter_root_paths(filter_node: Filter) -> list[Path]:
    """Collect absolute paths referenced through ``$`` in a filter.

    Every occurrence of the root identifier starts a new absolute path; the
    concrete field and index steps that follow it are accumulated.
    """

    paths: list[Path] = []
    seen: set[Path] = set()
    # Walk every Child subtree that is anchored at a Root so the complete
    # static suffix (for example `cfg.limit` in `$.cfg.limit`) is captured.
    for child in _collect_nodes(filter_node, Child):
        if not _collect_nodes(child, Root):
            continue
        for path in _static_paths_below(child):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths and _collect_nodes(filter_node, Root):
        paths.append(())
    return paths


def _collect_nodes(root: JSONPath, wanted: type) -> list[JSONPath]:
    """Return every node of type *wanted* inside the *root* subtree."""

    found: list[JSONPath] = []
    pending: list[JSONPath] = [root]
    while pending:
        current = pending.pop()
        if isinstance(current, wanted):
            found.append(current)
        for attr in ("left", "right", "target", "value"):
            child = getattr(current, attr, None)
            if isinstance(child, JSONPath):
                pending.append(child)
        for attr in ("operands", "expressions"):
            children = getattr(current, attr, None)
            if isinstance(children, list):
                pending.extend(c for c in children if isinstance(c, JSONPath))
    return found


def _static_paths_below(node: JSONPath) -> list[Path]:
    """Enumerate concrete field/index paths starting at a Root node."""

    if isinstance(node, Root):
        return [()]
    if isinstance(node, Child):
        prefixes = _static_paths_below(node.left)
        right = node.right
        if isinstance(right, Fields):
            steps = [(field,) for field in right.fields]
        elif isinstance(right, Index):
            steps = [(index,) for index in right.indices]
        else:
            # Wildcards, filters and slices make the suffix non-static;
            # the prefix is still a relevant dependency anchor.
            steps = [()]
        return [prefix + step for prefix in prefixes for step in steps]
    return [()]


def _match_path(datum: t.Any) -> Path:
    """Return the concrete tuple path of a ``find()`` match."""

    keys: list[t.Any] = []
    current = datum
    while isinstance(current, DatumInContext) and current.context is not None:
        path = current.path
        if isinstance(path, Fields):
            keys.append(path.fields[0])
        elif isinstance(path, Index):
            keys.append(path.indices[0])
        else:
            break
        current = current.context
    keys.reverse()
    return tuple(keys)


class Subscription:
    """The maintained hit set of one registered JSONPath expression."""

    def __init__(
        self, document: SubscriptionDocument, expression: JSONPath
    ) -> None:
        self._document: SubscriptionDocument | None = document
        self.expression = expression
        analysis = _analyze(expression)
        self._dependencies = analysis.records
        self._maintainable = analysis.maintainable
        self._hits: dict[Path, t.Any] = self._snapshot(document.proxy())

    def snapshot(self, data: t.Any) -> dict[Path, t.Any]:
        """Recompute the hit set against *data* and return it."""

        return self._snapshot(data)

    @property
    def maintainable(self) -> bool:
        """Whether local dependency tracking is used for this expression."""

        return self._maintainable

    @property
    def hits(self) -> dict[Path, t.Any]:
        """A copy of the current hit set keyed by concrete path."""

        return dict(self._hits)

    def _release(self) -> None:
        self._document = None
        self._hits.clear()
        self._dependencies.clear()

    def _snapshot(self, data: t.Any) -> dict[Path, t.Any]:
        hits: dict[Path, t.Any] = {}
        for datum in self.expression.find(data):
            hits[_match_path(datum)] = copy.deepcopy(datum.value)
        return hits

    def _refresh(
        self,
        document: SubscriptionDocument,
        new_data: t.Any,
        operations: list[PatchOperation],
        new_version: int,
    ) -> SubscriptionPatch:
        from_version = new_version - 1
        proxy = _wrap(document, (), copy.deepcopy(new_data))
        if not self._maintainable or _is_invalidated(
            self._dependencies, operations
        ):
            new_hits = self._snapshot(proxy)
        else:
            new_hits = self._maintain(proxy, operations)

        patch_ops = _diff_hits(self._hits, new_hits)
        self._hits = new_hits
        return SubscriptionPatch(
            self.expression, from_version, new_version, patch_ops
        )

    def _maintain(
        self,
        new_data: t.Any,
        operations: list[PatchOperation],
    ) -> dict[Path, t.Any]:
        """Apply localized updates to the previously recorded hit set.

        Only hits whose concrete path lies at or below a changed anchor are
        inspected; every other entry is retained without touching the tree.
        Newly appearing matches are discovered through the same anchor set.
        """

        hits = dict(self._hits)
        anchors = [_operation_anchor(operation) for operation in operations]
        if not anchors:
            return hits

        def relevant(path: Path) -> bool:
            return any(
                path == anchor or _is_relative(path, anchor)
                for anchor in anchors
            )

        if not any(relevant(path) for path in hits):
            # Nothing this subscription matches was touched; avoid any
            # traversal so unaffected branches are never visited.
            return hits

        for path, value in list(hits.items()):
            if not any(
                path == anchor or _is_relative(path, anchor)
                for anchor in anchors
            ):
                continue
            current = _navigate(new_data, path)
            if current is _MISSING:
                del hits[path]
            elif not _json_equal(current, value):
                hits[path] = copy.deepcopy(current)
        for anchor in anchors:
            container = _navigate(new_data, anchor)
            if not isinstance(container, (dict, list)):
                continue
            for datum in self.expression.find(new_data):
                path = _match_path(datum)
                if (
                    path not in hits
                    and (path == anchor or _is_relative(path, anchor))
                ):
                    hits[path] = copy.deepcopy(datum.value)
        return hits


_MISSING = object()


def _operation_anchor(operation: PatchOperation) -> Path:
    """Return the container path that a patch operation structurally affects."""

    path = operation.path
    if operation.op == "add" and path:
        # Insertions and new members change the parent container.
        return path[:-1]
    return path


def _is_relative(path: Path, anchor: Path) -> bool:
    """Return whether *path* is strictly below *anchor*."""

    return len(path) > len(anchor) and path[: len(anchor)] == anchor


def _navigate(root: t.Any, path: Path) -> t.Any:
    """Follow *path* into *root*, returning ``_MISSING`` when absent."""

    current = root
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return _MISSING
    return current


def _is_structural(operation: PatchOperation) -> bool:
    """Return whether an operation changes container membership or positions."""

    return operation.op in ("add", "remove", "move")


def _array_shift(
    operations: list[PatchOperation], container: Path
) -> bool:
    """Return whether any operation shifts element positions in *container*."""

    for operation in operations:
        if operation.op == "move":
            if operation.path[:-1] == container or (
                operation.from_path is not None
                and operation.from_path[:-1] == container
            ):
                return True
            continue
        if operation.path[:-1] != container:
            continue
        if operation.op == "add":
            return True
        if operation.op == "remove" and isinstance(operation.path[-1], int):
            return True
    return False


def _touches(path: Path, other: Path) -> bool:
    """Return whether *path* is *other* itself or lies below it."""

    return path == other or _is_relative(path, other)


def _is_invalidated(
    dependencies: list[_Dependency], operations: list[PatchOperation]
) -> bool:
    """Return whether any dependency forces a full recomputation."""

    for dependency in dependencies:
        if dependency.kind == "field":
            # A structural change that creates or removes the field (or a
            # descendant container) invalidates; a leaf value replacement is
            # maintained locally by hit value comparison.
            for operation in operations:
                if not _is_structural(operation):
                    continue
                if _touches(operation.path, dependency.path) or _touches(
                    dependency.path, _operation_anchor(operation)
                ):
                    return True
        elif dependency.kind == "index":
            container = dependency.path[:-1]
            index = dependency.path[-1]
            for operation in operations:
                if _is_relative(operation.path, dependency.path):
                    return True
                if (
                    operation.path[:-1] == container
                    and isinstance(operation.path[-1], int)
                    and operation.path[-1] >= index
                ):
                    return True
            if _array_shift(operations, container):
                return True
        elif dependency.kind in ("wildcard", "slice"):
            container = dependency.path
            for operation in operations:
                if operation.path == container:
                    # Replacing the whole container changes its membership.
                    return True
                if _is_structural(operation) and _touches(
                    operation.path, container
                ):
                    return True
            if _array_shift(operations, container):
                return True
        elif dependency.kind == "filter":
            container = dependency.path
            # A filter predicate reads element values, so any change inside
            # the container (structural or a leaf value) may flip membership.
            for operation in operations:
                if _touches(operation.path, container):
                    return True
            if _array_shift(operations, container):
                return True
            # A filter that references the root is invalidated whenever one
            # of its referenced absolute paths (or a subtree of it) changes,
            # regardless of whether the change is structural.
            for root_path in dependency.root_paths:
                for operation in operations:
                    if _touches(operation.path, root_path) or _touches(
                        root_path, _operation_anchor(operation)
                    ):
                        return True
    return False


def _diff_hits(
    old: dict[Path, t.Any], new: dict[Path, t.Any]
) -> list[PatchOperation]:
    """Compute add/remove/replace/move operations between two hit sets.

    Values are compared as multisets: a value that leaves one path and
    reappears at another (even when other hits shift in between) is reported
    as a single ``move`` rather than a paired remove/add or a replace.
    """

    # Values that disappear from a path, and values that newly appear at one.
    vanished: list[tuple[Path, t.Any]] = [
        (path, old[path]) for path in old if path not in new
    ]
    appeared: list[tuple[Path, t.Any]] = [
        (path, new[path]) for path in new if path not in old
    ]
    # Values that changed at a path that exists in both sets.
    for path in old:
        if path in new and not _json_equal(old[path], new[path]):
            vanished.append((path, old[path]))
            appeared.append((path, new[path]))

    # Pair vanished values with identical appeared values to form moves.
    moved: list[PatchOperation] = []
    remaining_appeared = list(appeared)
    unmatched_vanished: list[tuple[Path, t.Any]] = []
    for old_path, old_value in vanished:
        index = next(
            (
                i
                for i, (_, new_value) in enumerate(remaining_appeared)
                if _json_equal(old_value, new_value)
            ),
            None,
        )
        if index is None:
            unmatched_vanished.append((old_path, old_value))
        else:
            new_path, new_value = remaining_appeared.pop(index)
            moved.append(
                PatchOperation(
                    "move",
                    new_path,
                    copy.deepcopy(new_value),
                    from_path=old_path,
                )
            )

    # Whatever is left is a genuine removal or addition.  A vanished value
    # and an appeared value at the *same* path collapse into a replace.
    removals = {path: value for path, value in unmatched_vanished}
    additions = {path: value for path, value in remaining_appeared}

    operations: list[PatchOperation] = []
    for path, value in removals.items():
        if path in additions:
            operations.append(
                PatchOperation("replace", path, copy.deepcopy(additions[path]))
            )
        else:
            operations.append(PatchOperation("remove", path))
    for path, value in additions.items():
        if path not in removals:
            operations.append(PatchOperation("add", path, copy.deepcopy(value)))
    operations.extend(moved)
    operations.sort(key=lambda operation: operation.path)
    return operations
