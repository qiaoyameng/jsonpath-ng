"""
Locator generation for JSON documents.

Given a document, a normalized path to a target node, and a list of field
names that are allowed to act as identity fields, ``generate_locator``
produces a JSONPath expression that keeps pointing at the same node even
when the arrays along the path are reordered.

Array segments are expressed as filter expressions over identity fields
whenever a combination of those fields uniquely identifies the target
element among its siblings.  Only when no combination of the allowed
fields yields a unique identity does the generator fall back to a
positional index, and every such segment is reported in
``Locator.unstable_segments``.

All field names and values are encoded so that the generated expression
cannot break out of its string literals or field brackets; the finished
expression is verified with the real query engine (exactly one hit, at
exactly the requested path) before it is returned.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Optional, Sequence, Tuple, Union

from .exceptions import JSONPathError
from .ext.parser import parse as _parse
from .jsonpath import Fields, Index

__all__ = ['Locator', 'LocatorError', 'generate_locator', 'encode_key', 'encode_value']


class LocatorError(JSONPathError):
    """Raised when a locator cannot be generated or fails verification."""


class Locator:
    """The result of :func:`generate_locator`.

    ``expression`` is the JSONPath string locating the target.
    ``unstable_segments`` holds the normalized paths of every array
    segment that had to fall back to a positional index; those segments
    may point at a different element once the array is reordered.
    """

    __slots__ = ('expression', 'unstable_segments')

    def __init__(self, expression, unstable_segments=()):
        self.expression = expression
        self.unstable_segments = tuple(unstable_segments)

    @property
    def stable(self):
        """True when no segment fell back to a positional index."""
        return not self.unstable_segments

    def __str__(self):
        return self.expression

    def __repr__(self):
        return '%s(%r, unstable_segments=%r)' % (
            self.__class__.__name__, self.expression, self.unstable_segments)

    def __eq__(self, other):
        return (isinstance(other, Locator)
                and self.expression == other.expression
                and self.unstable_segments == other.unstable_segments)


# The extended lexer only accepts floats of the form `-?\d+\.\d+`.
_FLOAT_LITERAL_RE = re.compile(r'-?\d+\.\d+$')


def _quote_string(value: str) -> str:
    """Encode a string as a single-quoted JSONPath string literal.

    Only ``\\`` and ``'`` can terminate or escape inside the literal, so
    escaping exactly those two characters makes breakout impossible.
    """
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"


def encode_key(key: str) -> str:
    """Encode a mapping key as a JSONPath child segment, e.g. ``['a']``."""
    if not isinstance(key, str):
        raise LocatorError('Mapping keys must be strings, got %r' % (key,))
    return '[' + _quote_string(key) + ']'


def encode_value(value: Any) -> Optional[str]:
    """Encode a scalar as a JSONPath filter literal.

    Returns ``None`` for values this JSONPath dialect cannot express as a
    comparison literal (``None``, containers, non-finite or
    non-representable floats); such fields cannot serve as identity.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if not _FLOAT_LITERAL_RE.match(text) or float(text) != value:
            return None
        return text
    if isinstance(value, str):
        return _quote_string(value)
    return None


def _filter_segment(conditions: Sequence[Tuple[str, Any]]) -> str:
    parts = ['@%s == %s' % (encode_key(field), encode_value(value))
             for field, value in conditions]
    return '[?(' + ' & '.join(parts) + ')]'


def _combo_is_unique(siblings, element, combo) -> bool:
    """Check with the real query engine that ``combo`` picks ``element`` only."""
    condition = _filter_segment([(field, element[field]) for field in combo])
    matches = _parse('$' + condition).find(siblings)
    return len(matches) == 1 and matches[0].value is element


def _find_identity_combination(siblings, index, identity_fields,
                               max_combination_size) -> Optional[Tuple[str, ...]]:
    element = siblings[index]
    if not isinstance(element, dict):
        return None
    usable = [field for field in identity_fields
              if isinstance(field, str)
              and field in element
              and encode_value(element[field]) is not None]
    if not usable:
        return None
    if max_combination_size is None:
        max_size = len(usable)
    else:
        max_size = min(max_combination_size, len(usable))
    # Fixed selection rule: smallest combination first; ties are broken by
    # the order in which the fields appear in ``identity_fields``.
    for size in range(1, max_size + 1):
        for combo in combinations(usable, size):
            if _combo_is_unique(siblings, element, combo):
                return combo
    return None


def _normalize_target_path(target_path) -> Tuple[Union[str, int], ...]:
    if isinstance(target_path, (str, bytes)):
        raise LocatorError(
            'target_path must be a sequence of segments, not a string')
    segments = []
    for segment in target_path:
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise LocatorError(
                'Path segments must be strings or integers, got %r' % (segment,))
        segments.append(segment)
    return tuple(segments)


def _datum_normalized_path(datum) -> Tuple[Union[str, int], ...]:
    parts = []
    node = datum
    while node is not None:
        path = node.path
        if isinstance(path, Index):
            if len(path.indices) != 1:
                raise LocatorError('Unexpected multi-index path %r' % (path,))
            parts.append(path.indices[0])
        elif isinstance(path, Fields):
            if len(path.fields) != 1:
                raise LocatorError('Unexpected multi-field path %r' % (path,))
            parts.append(path.fields[0])
        node = node.context
    return tuple(reversed(parts))


def _verify(expression, document, expected_path) -> None:
    matches = _parse(expression).find(document)
    if len(matches) != 1:
        raise LocatorError(
            'Generated locator %r matched %d nodes instead of exactly one'
            % (expression, len(matches)))
    actual_path = _datum_normalized_path(matches[0])
    if actual_path != expected_path:
        raise LocatorError(
            'Generated locator %r resolved to path %r, expected %r'
            % (expression, actual_path, expected_path))


def generate_locator(document, target_path, identity_fields,
                     max_combination_size=None) -> Locator:
    """Generate a JSONPath locator for the node at ``target_path``.

    ``target_path`` is a normalized path: a sequence of mapping keys
    (strings) and array indices (integers) leading from ``document`` to
    the target node.  ``identity_fields`` lists the field names allowed
    to identify array elements; combinations are tried smallest first,
    in the order the fields are listed.  The returned locator is
    verified against ``document`` with the query engine before being
    returned.
    """
    if not isinstance(identity_fields, (list, tuple)):
        raise LocatorError('identity_fields must be a list or tuple of field names')
    segments = _normalize_target_path(target_path)

    expression_parts = ['$']
    unstable_segments = []
    prefix: list = []
    node = document

    for segment in segments:
        if isinstance(segment, str):
            if not isinstance(node, dict) or segment not in node:
                raise LocatorError(
                    'No key %r at path %r' % (segment, tuple(prefix)))
            expression_parts.append(encode_key(segment))
            node = node[segment]
            prefix.append(segment)
        else:
            if not isinstance(node, (list, tuple)):
                raise LocatorError(
                    'Expected an array at path %r, found %s'
                    % (tuple(prefix), type(node).__name__))
            index = segment + len(node) if segment < 0 else segment
            if not 0 <= index < len(node):
                raise LocatorError(
                    'Index %d out of range at path %r' % (segment, tuple(prefix)))
            element = node[index]
            combo = _find_identity_combination(
                node, index, identity_fields, max_combination_size)
            if combo is None:
                expression_parts.append('[%d]' % index)
                unstable_segments.append(tuple(prefix) + (index,))
            else:
                expression_parts.append(
                    _filter_segment([(field, element[field]) for field in combo]))
            node = element
            prefix.append(index)

    expression = ''.join(expression_parts)
    _verify(expression, document, tuple(prefix))
    return Locator(expression, tuple(unstable_segments))
