# -*- coding: utf-8 -*-
"""Stable locator generation for JSON documents.

Given a document, a normalized target path and an allowlist of identity
fields, :func:`generate_locator` produces a JSONPath locator that identifies
array elements by their identity fields (``[?(@.id == ...)]`` filters)
instead of their position.  Positional indices are only used as a last
resort and are reported as unstable segments.

Selection rules are deterministic: among all field combinations that
uniquely identify an element, the one with the fewest fields wins; ties are
broken by the order of the identity-field allowlist.  Every generated
locator is verified with the jsonpath_ng query engine before it is
returned, and must match the target object, and only the target object.
"""

import itertools
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from .jsonpath import Child, Fields, Index, Root
from .parser import parse as _parse

__all__ = [
    'LocatorError',
    'LocatorResult',
    'LocatorSegment',
    'generate_locator',
]

# Only floats that survive a repr() round-trip through the extended lexer's
# FLOAT token (``-?\d+\.\d+``) may be used as filter literals.
_FLOAT_LITERAL_RE = re.compile(r'-?\d+\.\d+\Z')

_MISSING = object()


class LocatorError(Exception):
    """Raised when a locator cannot be generated for the given target."""


@dataclass(frozen=True)
class LocatorSegment:
    """Describes how one segment of the target path is addressed."""

    position: int
    segment: Union[str, int]
    kind: str  # 'field' | 'identity' | 'index'
    stable: bool
    identity: Optional[Tuple[Tuple[str, Any], ...]] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class LocatorResult:
    """A generated locator plus its stability report."""

    locator: str
    segments: Tuple[LocatorSegment, ...]

    @property
    def stable(self) -> bool:
        """True when no segment relies on a positional array index."""
        return all(segment.stable for segment in self.segments)

    @property
    def unstable_segments(self) -> Tuple[LocatorSegment, ...]:
        """Segments that fell back to positional indices."""
        return tuple(segment for segment in self.segments if not segment.stable)

    def find(self, document) -> list:
        """Evaluate the locator against *document* with the query engine."""
        return _ext_parse(self.locator).find(document)

    def __str__(self) -> str:
        return self.locator


def _ext_parse(text):
    # Imported lazily so that importing jsonpath_ng.locator never triggers
    # the extended parser (and its dependencies) unless it is needed.
    from .ext.parser import parse
    return parse(text)


def _segments_from_ast(expr) -> List[Union[str, int]]:
    segments: List[Union[str, int]] = []
    node = expr
    while isinstance(node, Child):
        right = node.right
        if isinstance(right, Fields) and len(right.fields) == 1:
            segments.append(right.fields[0])
        elif isinstance(right, Index) and len(right.indices) == 1:
            segments.append(right.indices[0])
        else:
            raise ValueError(
                'target path is not a normalized path: %r' % (expr,))
        node = node.left
    if not isinstance(node, Root):
        raise ValueError(
            'target path is not a normalized path: %r' % (expr,))
    segments.reverse()
    return segments


def _normalize_target_path(
        target_path: Union[str, Sequence[Union[str, int]]]
) -> List[Union[str, int]]:
    if isinstance(target_path, str):
        text = target_path.strip()
        if not text:
            raise ValueError('target path must not be empty')
        if not text.startswith('$'):
            text = '$' + ('' if text.startswith('[') else '.') + text
        return _segments_from_ast(_parse(text))
    segments = []
    for segment in target_path:
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise TypeError(
                'path segments must be str or int, got %r' % (segment,))
        segments.append(segment)
    return segments


def _encode_string(value: str) -> str:
    # The lexer only understands \<char> escapes (the escaped character is
    # taken literally), so escaping the backslash and the quote itself is
    # both sufficient and exact -- every other byte stays as-is.
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"


def _encode_literal(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return _encode_string(value)


def _encode_field_access(name: str) -> str:
    return '[' + _encode_string(name) + ']'


def _encode_filter(identity: Tuple[Tuple[str, Any], ...]) -> str:
    expressions = ' & '.join(
        '@[%s] == %s' % (_encode_string(field), _encode_literal(value))
        for field, value in identity)
    return '[?(%s)]' % expressions


def _usable_literal(value: Any) -> bool:
    """Whether *value* can be expressed exactly as a filter literal."""
    if value is None:
        return False
    if isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return _FLOAT_LITERAL_RE.match(repr(value)) is not None
    return False


def _literal_matches(literal: Any, candidate: Any) -> bool:
    """Mirror the equality semantics of the engine's filter expressions."""
    if candidate is _MISSING:
        return False
    if isinstance(literal, bool):
        return candidate == literal
    if isinstance(literal, int):
        # The engine coerces the candidate with int() for int literals.
        try:
            candidate = int(candidate)
        except (ValueError, TypeError):
            return False
        return candidate == literal
    return candidate == literal


def _combo_matches(identity, sibling) -> bool:
    if not isinstance(sibling, dict):
        # Filters never match non-object array elements.
        return False
    return all(
        _literal_matches(value, sibling.get(field, _MISSING))
        for field, value in identity)


def _find_identity(element: dict, siblings: list,
                   identity_fields: Sequence[str]
                   ) -> Optional[Tuple[Tuple[str, Any], ...]]:
    candidates = []
    seen = set()
    for field in identity_fields:
        if not isinstance(field, str):
            raise TypeError(
                'identity fields must be strings, got %r' % (field,))
        if field in seen or field not in element:
            continue
        seen.add(field)
        value = element[field]
        if _usable_literal(value):
            candidates.append((field, value))
    # Deterministic choice: fewest fields first, then allowlist order.
    for size in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, size):
            if sum(_combo_matches(combo, sibling)
                   for sibling in siblings) == 1:
                return combo
    return None


def _verify_unique(locator: str, document, target) -> bool:
    """Check with the real query engine that *locator* hits only *target*."""
    try:
        matches = _ext_parse(locator).find(document)
    except Exception:
        return False
    return len(matches) == 1 and matches[0].value is target


def generate_locator(
        document,
        target_path: Union[str, Sequence[Union[str, int]]],
        identity_fields: Iterable[str] = (),
) -> LocatorResult:
    """Generate a stable JSONPath locator for *target_path* in *document*.

    *target_path* is a normalized path, either as a string
    (``$.users[3].name`` or ``$['users'][3]['name']``) or as a sequence of
    str/int segments.  *identity_fields* is the allowlist of field names
    that may be used to identify array elements, in priority order.

    Array elements are addressed with ``[?(@.field == value)]`` filters
    built from the smallest uniquely-identifying combination of allowed
    fields; segments where no unique identity exists fall back to their
    positional index and are flagged as unstable in the result.
    """
    identity_fields = tuple(identity_fields or ())
    segments = _normalize_target_path(target_path)

    parts = ['$']
    infos: List[LocatorSegment] = []
    current = document
    for position, segment in enumerate(segments):
        if isinstance(segment, str):
            if segment == '*':
                raise LocatorError(
                    "cannot generate a locator for field name '*' at "
                    'position %d: it cannot be distinguished from the '
                    'wildcard operator' % position)
            if not isinstance(current, dict) or segment not in current:
                raise LocatorError(
                    'target path does not exist in the document: '
                    'missing field %r at position %d' % (segment, position))
            parts.append(_encode_field_access(segment))
            infos.append(LocatorSegment(
                position=position, segment=segment,
                kind='field', stable=True))
            current = current[segment]
            continue

        if not isinstance(current, list):
            raise LocatorError(
                'target path does not exist in the document: '
                'index %d at position %d is not inside an array'
                % (segment, position))
        index = segment + len(current) if segment < 0 else segment
        if not 0 <= index < len(current):
            raise LocatorError(
                'target path does not exist in the document: '
                'index %d out of range at position %d'
                % (segment, position))

        element = current[index]
        identity = None
        reason = None
        if isinstance(element, dict):
            identity = _find_identity(element, current, identity_fields)
            if identity is not None:
                candidate = ''.join(parts) + _encode_filter(identity)
                if not _verify_unique(candidate, document, element):
                    identity = None
                    reason = ('identity candidate rejected by query '
                              'engine verification')
            if identity is None and reason is None:
                reason = 'no unique identity field combination'
        else:
            reason = 'array element is not an object'

        if identity is not None:
            parts.append(_encode_filter(identity))
            infos.append(LocatorSegment(
                position=position, segment=index,
                kind='identity', stable=True, identity=identity))
        else:
            parts.append('[%d]' % index)
            infos.append(LocatorSegment(
                position=position, segment=index,
                kind='index', stable=False, reason=reason))
        current = element

    locator = ''.join(parts)
    if not _verify_unique(locator, document, current):
        raise LocatorError(
            'generated locator failed verification: %r' % locator)
    return LocatorResult(locator=locator, segments=tuple(infos))
