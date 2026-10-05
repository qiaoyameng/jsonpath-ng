"""
Static containment checking for JSONPath selectors.

A permission configuration is a set of JSONPath selectors describing the
readable locations of a document.  When a new selector is submitted, this
module statically decides its relation to the existing coverage:

* ``Relation.CONTAINED`` -- every structural path the new selector may match
  is also matched by the allowed set.
* ``Relation.NOT_CONTAINED`` -- the new selector may match a structural path
  outside the allowed set.  A small counterexample (a JSON document plus the
  out-of-bounds path) is provided and can be replayed with ``find()``.
* ``Relation.UNKNOWN`` -- the selector uses constructs outside the supported
  subset, so no safe conclusion is reported.  Callers should reject the
  selector rather than treat the analysis as an authorization decision.

Supported subset
----------------
* the root ``$`` and ``this``,
* object fields (``$.a``, ``$['a b']``, ``$.a,b``) and the wildcard ``*``,
* descendants ``..``,
* non-negative array indices (``$[0]``, ``$[0,2]``),
* slices with non-negative bounds and a finite, positive step
  (``$[*]``, ``$[1:]``, ``$[0:10:2]``),
* unions of the above (``$.a | $.b``).

Anything else -- filters, ``where``, ``parent``, intersections, negative
indices, zero/negative slice steps, ``$`` in a non-initial position, and the
extensions from ``jsonpath_ng.ext`` -- yields ``Relation.UNKNOWN``.

Semantics
---------
Selectors are compared by the *set of structural paths* they may match,
where a structural path is a sequence of object fields and array indices
(e.g. ``$.a`` vs ``$.ab`` are unrelated: field names are compared as
atoms, never as text prefixes).  Array-length boundaries are modeled
exactly: an index ``i`` requires length > i, and a slice denotes the
arithmetic progression of indices it can realize over arrays of any
length.  A selector with an empty denotation (e.g. ``$.a[2:1]``) is
contained in any allowed set, and the empty selector ``$`` matches only
the root path.

When the relation is ``NOT_CONTAINED`` the returned counterexample is
verified by actually running ``find()``: the path is produced by the new
selector on the document and by none of the allowed selectors.
"""

from __future__ import annotations

import itertools
import re
import string
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import gcd
from typing import Any, Iterable, List, Optional, Tuple

from jsonpath_ng.exceptions import JSONPathError
from jsonpath_ng.jsonpath import (
    JSONPath,
    Child,
    Descendants,
    Fields,
    Index,
    Root,
    Slice,
    This,
    Union,
)
from jsonpath_ng.parser import parse as _parse

__all__ = [
    "Relation",
    "RelationResult",
    "Counterexample",
    "check_relation",
    "is_covered",
    "replay",
]


class Relation(Enum):
    """The three possible outcomes of a static containment check."""

    CONTAINED = "contained"
    NOT_CONTAINED = "not_contained"
    UNKNOWN = "unknown"


@dataclass
class Counterexample:
    """
    A witness that a selector escapes the allowed coverage.

    ``document`` is a small JSON document and ``path`` the out-of-bounds
    structural path: the new selector matches ``path`` on ``document``
    while no allowed selector does.  ``verified`` is True when the witness
    was replayed with ``find()`` before being returned.
    """

    document: Any
    path: str
    verified: bool = True
    _elements: Tuple[tuple, ...] = field(default=(), repr=False, compare=False)

    @property
    def path_ast(self) -> "JSONPath":
        """The out-of-bounds path as a JSONPath AST.

        Compares equal to the ``full_path`` of a ``find()`` match, e.g.
        ``counterexample.path_ast in [m.full_path for m in sel.find(doc)]``.
        """
        return _path_jsonpath(list(self._elements))

    def replay(self, selector, allowed: Iterable = ()) -> bool:
        """Re-run ``find()`` to confirm this counterexample still reproduces."""
        return replay(self, selector, allowed)


@dataclass
class RelationResult:
    """The outcome of :func:`check_relation`."""

    relation: Relation
    reason: str = ""
    counterexample: Optional[Counterexample] = None

    @property
    def contained(self) -> bool:
        return self.relation is Relation.CONTAINED

    @property
    def unknown(self) -> bool:
        return self.relation is Relation.UNKNOWN

    def __bool__(self) -> bool:
        # Only a definite CONTAINED is a safe "yes"; both NOT_CONTAINED and
        # UNKNOWN are falsy so callers fail closed.
        return self.contained


# ---------------------------------------------------------------------------
# Sets of field names (finite or cofinite)
# ---------------------------------------------------------------------------


class _Names:
    """A set of object field names: finite, or cofinite (all but finitely many)."""

    __slots__ = ("cofinite", "names")

    def __init__(self, cofinite: bool, names=()):
        self.cofinite = cofinite
        self.names = frozenset(names)

    @classmethod
    def all(cls) -> "_Names":
        return cls(True, ())

    @classmethod
    def of(cls, names) -> "_Names":
        return cls(False, names)

    def is_empty(self) -> bool:
        return not self.cofinite and not self.names

    def intersect(self, other: "_Names") -> "_Names":
        if self.cofinite and other.cofinite:
            return _Names(True, self.names | other.names)
        if self.cofinite:
            return _Names(False, other.names - self.names)
        if other.cofinite:
            return _Names(False, self.names - other.names)
        return _Names(False, self.names & other.names)

    def subtract(self, other: "_Names") -> "_Names":
        if self.cofinite:
            if other.cofinite:
                # (ALL \ s) \ (ALL \ o) == o \ s
                return _Names(False, other.names - self.names)
            return _Names(True, self.names | other.names)
        if other.cofinite:
            return _Names(False, self.names & other.names)
        return _Names(False, self.names - other.names)

    def pick(self) -> str:
        """Return a concrete member, deterministically."""
        if not self.cofinite:
            return min(self.names)
        candidates = itertools.chain(
            string.ascii_lowercase, ("k%d" % i for i in itertools.count())
        )
        for candidate in candidates:
            if candidate not in self.names:
                return candidate
        raise AssertionError("unreachable")  # pragma: no cover

    def __eq__(self, other):
        return (
            isinstance(other, _Names)
            and self.cofinite == other.cofinite
            and self.names == other.names
        )

    def __hash__(self):
        return hash((self.cofinite, self.names))

    def __repr__(self):
        if self.cofinite:
            if not self.names:
                return "_Names.all()"
            return "_Names(all - %r)" % sorted(self.names)
        return "_Names.of(%r)" % sorted(self.names)


# ---------------------------------------------------------------------------
# Sets of non-negative array indices (unions of arithmetic progressions)
# ---------------------------------------------------------------------------

# A segment is a tuple (start, end, step) with 0 <= start, step >= 1 and
# end either None (unbounded) or an integer > start.  It denotes
# {start + k * step : k >= 0 and start + k * step < end}.

_MAX_SEGMENTS = 512


class _TooHard(Exception):
    """Raised when the analysis exceeds its resource bounds."""


def _egcd(a: int, b: int) -> Tuple[int, int]:
    """Return (x, y) such that a*x + b*y == gcd(a, b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_s, old_t


def _seg_intersect(seg1, seg2):
    """Intersect two segments; return a segment or None."""
    a1, e1, m1 = seg1
    a2, e2, m2 = seg2
    g = gcd(m1, m2)
    if (a2 - a1) % g:
        return None
    lcm = m1 // g * m2
    p, _ = _egcd(m1, m2)  # p*m1 == g (mod m2)
    k0 = ((a2 - a1) // g * p) % (m2 // g)
    c = (a1 + m1 * k0) % lcm
    lo = max(a1, a2)
    if e1 is None:
        hi = e2
    elif e2 is None:
        hi = e1
    else:
        hi = min(e1, e2)
    if lo <= c:
        x0 = c
    else:
        x0 = c + lcm * ((lo - c + lcm - 1) // lcm)
    if hi is not None and x0 >= hi:
        return None
    return (x0, hi, lcm)


def _seg_complement(seg) -> List[tuple]:
    """Complement of a segment within the non-negative integers."""
    a, e, s = seg
    out = []
    if a > 0:
        out.append((0, a, 1))
    for r in range(1, s):
        start = a + r
        if e is None or start < e:
            out.append((start, e, s))
    if e is not None:
        out.append((e, None, 1))
    return out


class _Indices:
    """A set of non-negative array indices: a union of arithmetic segments."""

    __slots__ = ("segments",)

    def __init__(self, segments=()):
        kept = []
        for seg in segments:
            start, end, step = seg
            if end is None or start < end:
                kept.append((start, end, step))
            if len(kept) > _MAX_SEGMENTS:
                raise _TooHard("index set too complex")
        self.segments = tuple(kept)

    @classmethod
    def all(cls) -> "_Indices":
        return cls(((0, None, 1),))

    @classmethod
    def singleton(cls, index: int) -> "_Indices":
        return cls(((index, index + 1, 1),))

    @classmethod
    def progression(cls, start: int, end: Optional[int], step: int) -> "_Indices":
        return cls(((start, end, step),))

    def is_empty(self) -> bool:
        return not self.segments

    def pick(self) -> int:
        """Return the smallest member, deterministically."""
        return min(start for start, _, _ in self.segments)

    def intersect(self, other: "_Indices") -> "_Indices":
        out = []
        for seg1 in self.segments:
            for seg2 in other.segments:
                seg = _seg_intersect(seg1, seg2)
                if seg is not None:
                    out.append(seg)
        return _Indices(out)

    def subtract(self, other: "_Indices") -> "_Indices":
        result = list(self.segments)
        for seg in other.segments:
            complement = _seg_complement(seg)
            nxt = []
            for own in result:
                for piece in complement:
                    inter = _seg_intersect(own, piece)
                    if inter is not None:
                        nxt.append(inter)
            result = nxt
            if not result:
                break
        return _Indices(result)

    def __eq__(self, other):
        return isinstance(other, _Indices) and self.segments == other.segments

    def __hash__(self):
        return hash(self.segments)

    def __repr__(self):
        return "_Indices(%r)" % (self.segments,)


# ---------------------------------------------------------------------------
# Patterns: the denotation of a supported selector
# ---------------------------------------------------------------------------

# A pattern is a tuple (absolute, guard, steps) where steps is a tuple of
# steps.  A step is a _FieldStep, an _IndexStep or _DESC.  A pattern denotes
# a set of structural paths (sequences of object fields and array indices):
# _DESC matches any (possibly empty) sequence of elements.  ``absolute``
# records that a ``$`` appeared inside the expression and resets matching to
# the document root (e.g. ``$.a.$`` matches the root, not ``$.a``); ``guard``
# is the set of step sequences that must also match the document for the
# reset to fire (the prefix before the ``$``).  Patterns with a non-trivial
# guard match conditionally, which the path-set model cannot express, so
# they are rejected as unsupported after normalization.


class _FieldStep:
    __slots__ = ("names",)

    def __init__(self, names: _Names):
        self.names = names

    def __eq__(self, other):
        return isinstance(other, _FieldStep) and self.names == other.names

    def __hash__(self):
        return hash(("field", self.names))

    def __repr__(self):
        return "_FieldStep(%r)" % (self.names,)


class _IndexStep:
    __slots__ = ("indices",)

    def __init__(self, indices: _Indices):
        self.indices = indices

    def __eq__(self, other):
        return isinstance(other, _IndexStep) and self.indices == other.indices

    def __hash__(self):
        return hash(("index", self.indices))

    def __repr__(self):
        return "_IndexStep(%r)" % (self.indices,)


class _DescStep:
    __slots__ = ()

    def __eq__(self, other):
        return isinstance(other, _DescStep)

    def __hash__(self):
        return hash("desc")

    def __repr__(self):
        return "_DESC"


_DESC = _DescStep()

_MAX_PATTERNS = 512


class _Unsupported(Exception):
    """Raised when a selector uses constructs outside the supported subset."""


def _is_valid_index(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalize(expr) -> frozenset:
    """Denote a parsed selector as a set of (absolute, guard, steps) patterns."""
    if isinstance(expr, Root):
        return frozenset({(True, frozenset(), ())})
    if isinstance(expr, This):
        return frozenset({(False, frozenset(), ())})
    if isinstance(expr, Fields):
        if not all(isinstance(name, str) for name in expr.fields):
            raise _Unsupported("non-string field name")
        if "*" in expr.fields:
            names = _Names.all()
        else:
            names = _Names.of(expr.fields)
        return frozenset({(False, frozenset(), (_FieldStep(names),))})
    if isinstance(expr, Index):
        if not all(_is_valid_index(index) for index in expr.indices):
            raise _Unsupported("negative or non-integer array index")
        indices = _Indices((index, index + 1, 1) for index in expr.indices)
        return frozenset({(False, frozenset(), (_IndexStep(indices),))})
    if isinstance(expr, Slice):
        for value in (expr.start, expr.end):
            if value is not None and not _is_valid_index(value):
                raise _Unsupported("negative slice bound")
        if expr.step is None:
            step = 1
        elif (
            isinstance(expr.step, int)
            and not isinstance(expr.step, bool)
            and expr.step >= 1
        ):
            step = expr.step
        else:
            raise _Unsupported("slice step must be a positive integer")
        start = expr.start if expr.start is not None else 0
        step_pattern = _IndexStep(_Indices.progression(start, expr.end, step))
        return frozenset({(False, frozenset(), (step_pattern,))})
    if isinstance(expr, Child):
        left = _normalize(expr.left)
        right = _normalize(expr.right)
        return _concat_patterns(left, right)
    if isinstance(expr, Descendants):
        left = _normalize(expr.left)
        right = _normalize(expr.right)
        return _concat_patterns(left, right, descendant=True)
    if isinstance(expr, Union):
        combined = _normalize(expr.left) | _normalize(expr.right)
        if len(combined) > _MAX_PATTERNS:
            raise _TooHard("too many union branches")
        return frozenset(combined)
    raise _Unsupported("unsupported construct: %s" % type(expr).__name__)


def _steps_possibly_nonempty(steps) -> bool:
    """Whether a step sequence can match at least one structural path."""
    return all(_step_label_non_empty(step) for step in steps)


def _step_label_non_empty(step) -> bool:
    if isinstance(step, _FieldStep):
        return not step.names.is_empty()
    if isinstance(step, _IndexStep):
        return not step.indices.is_empty()
    return True


def _concat_patterns(left: frozenset, right: frozenset, descendant: bool = False) -> frozenset:
    """
    Concatenate two pattern sets (optionally with a descendant step between).

    An absolute right-hand pattern discards the left-hand steps entirely (a
    ``$`` resets matching to the root) and records the left-hand steps as a
    guard: the reset only fires on documents the left-hand side matches.
    """
    if len(left) * max(1, len(right)) > _MAX_PATTERNS:
        raise _TooHard("too many pattern combinations")
    combined = set()
    for left_abs, left_guard, left_steps in left:
        for right_abs, right_guard, right_steps in right:
            guard = left_guard | right_guard
            if right_abs:
                # The reset fires only where the left-hand side matches; a
                # left-hand side that can never match makes the reset dead.
                if _steps_possibly_nonempty(left_steps):
                    combined.add((True, guard | {left_steps}, right_steps))
            elif descendant:
                combined.add((left_abs, guard, left_steps + (_DESC,) + right_steps))
            else:
                combined.add((left_abs, guard, left_steps + right_steps))
    if len(combined) > _MAX_PATTERNS:
        raise _TooHard("too many pattern combinations")
    return frozenset(combined)


def _pattern_key(pattern) -> tuple:
    """A deterministic sort key for (absolute, guard, steps) patterns."""
    absolute, _, steps = pattern
    key = []
    for step in steps:
        if isinstance(step, _FieldStep):
            names = step.names
            key.append((0, 0 if names.cofinite else 1, sorted(names.names)))
        elif isinstance(step, _IndexStep):
            key.append((1, tuple(tuple(seg) for seg in step.indices.segments)))
        else:
            key.append((2,))
    return (not absolute, tuple(key))


# ---------------------------------------------------------------------------
# Symbolic step labels and region algebra
# ---------------------------------------------------------------------------

# A region is a tuple ("field", _Names) or ("index", _Indices): the set of
# concrete path elements matching it.  A label is a tuple of regions, read
# as their union.  Field and index elements are disjoint by construction.

_FIELD = "field"
_INDEX = "index"

_NAMES_ALL = _Names.all()
_INDICES_ALL = _Indices.all()
_DESC_LABEL = ((_FIELD, _NAMES_ALL), (_INDEX, _INDICES_ALL))


def _step_label(step) -> tuple:
    if isinstance(step, _FieldStep):
        regions = ((_FIELD, step.names),)
    elif isinstance(step, _IndexStep):
        regions = ((_INDEX, step.indices),)
    else:
        regions = _DESC_LABEL
    return tuple(region for region in regions if not region[1].is_empty())


def _region_intersect(region1, region2):
    kind1, set1 = region1
    kind2, set2 = region2
    if kind1 != kind2:
        return None
    inter = set1.intersect(set2)
    if inter.is_empty():
        return None
    return (kind1, inter)


def _region_subtract(region1, region2):
    kind1, set1 = region1
    kind2, set2 = region2
    if kind1 != kind2:
        return (region1,)
    diff = set1.subtract(set2)
    if diff.is_empty():
        return ()
    return ((kind1, diff),)


def _label_intersect(label1, label2) -> tuple:
    out = []
    for region1 in label1:
        for region2 in label2:
            inter = _region_intersect(region1, region2)
            if inter is not None:
                out.append(inter)
    return tuple(out)


def _label_subtract(label1, label2) -> tuple:
    out = []
    for region in label1:
        pieces = (region,)
        for other in label2:
            pieces = tuple(
                piece
                for current in pieces
                for piece in _region_subtract(current, other)
            )
            if not pieces:
                break
        out.extend(pieces)
    return tuple(out)


_MAX_CELLS = 512


def _split_cells(new_label, allowed_labels) -> List[Tuple[tuple, frozenset]]:
    """
    Partition ``new_label`` into cells on which the set of matching allowed
    labels is constant.  Returns (cell, allowed-label-indices) pairs.
    """
    cells = [(new_label, ())]
    for position, allowed_label in enumerate(allowed_labels):
        nxt = []
        for regions, mask in cells:
            inter = _label_intersect(regions, allowed_label)
            if inter:
                nxt.append((inter, mask + (position,)))
            diff = _label_subtract(regions, allowed_label)
            if diff:
                nxt.append((diff, mask))
        cells = nxt
        if not cells:
            break
        if len(cells) > _MAX_CELLS:
            raise _TooHard("too many overlapping selector branches")
    return [(regions, frozenset(mask)) for regions, mask in cells]


# ---------------------------------------------------------------------------
# Pattern automata and the containment search
# ---------------------------------------------------------------------------

# Each pattern set is viewed as an NFA whose states are (pattern, position)
# pairs.  A descendant step is a self-loop labeled with "any element" plus
# an epsilon move to the next position (eliminated on the fly).

_MAX_STATES = 20000


def _nfa_edges(patterns, state) -> tuple:
    pat_index, pos = state
    pattern = patterns[pat_index]
    if pos >= len(pattern):
        return ()
    step = pattern[pos]
    if step is _DESC:
        # Self-loop consuming any element, plus the epsilon move's edges.
        return ((_DESC_LABEL, state),) + _nfa_edges(patterns, (pat_index, pos + 1))
    return ((_step_label(step), (pat_index, pos + 1)),)


def _nfa_final(patterns, state) -> bool:
    pat_index, pos = state
    pattern = patterns[pat_index]
    if pos >= len(pattern):
        return True
    if pattern[pos] is _DESC:
        return _nfa_final(patterns, (pat_index, pos + 1))
    return False


def _witness_cells(new_patterns, allowed_patterns):
    """
    Breadth-first search for a path matched by ``new_patterns`` but by no
    allowed pattern.  Yields candidate witnesses (lists of symbolic cells,
    shortest first); stops yielding once the language difference is
    provably empty.
    """
    allowed_start = frozenset((i, 0) for i in range(len(allowed_patterns)))
    edge_cache = {}

    def allowed_edges(subset):
        if subset not in edge_cache:
            edges = []
            for state in subset:
                edges.extend(_nfa_edges(allowed_patterns, state))
            edge_cache[subset] = edges
        return edge_cache[subset]

    def allowed_final(subset):
        return any(_nfa_final(allowed_patterns, state) for state in subset)

    starts = [((i, 0), allowed_start) for i in range(len(new_patterns))]
    visited = set(starts)
    queue = deque((state, []) for state in starts)
    while queue:
        (new_state, allowed_subset), cells = queue.popleft()
        if _nfa_final(new_patterns, new_state) and not allowed_final(allowed_subset):
            yield cells
            continue
        edges = allowed_edges(allowed_subset)
        allowed_labels = [label for label, _ in edges]
        for new_label, next_new_state in _nfa_edges(new_patterns, new_state):
            if not new_label:
                continue
            for cell, mask in _split_cells(new_label, allowed_labels):
                next_subset = frozenset(edges[i][1] for i in mask)
                key = (next_new_state, next_subset)
                if key in visited:
                    continue
                visited.add(key)
                if len(visited) > _MAX_STATES:
                    raise _TooHard("state space too large")
                queue.append((key, cells + [cell]))


# ---------------------------------------------------------------------------
# Counterexample concretization and replay
# ---------------------------------------------------------------------------


def _pick_element(cell) -> tuple:
    """Pick a concrete path element from a symbolic cell, deterministically."""
    for kind, values in cell:
        if kind == _FIELD and not values.is_empty():
            return ("f", values.pick())
        if kind == _INDEX and not values.is_empty():
            return ("i", values.pick())
    raise AssertionError("empty cell")  # pragma: no cover


def _build_document(elements) -> Any:
    """Build the smallest JSON document embedding the given structural path."""
    document = None
    for kind, value in reversed(elements):
        if kind == "f":
            document = {value: document}
        else:
            container = [None] * (value + 1)
            container[value] = document
            document = container
    return document


def _path_jsonpath(elements) -> JSONPath:
    """Rebuild the structural path as a JSONPath AST (used for replay checks)."""
    path = Root()
    for kind, value in elements:
        path = path.child(Fields(value)) if kind == "f" else path.child(Index(value))
    return path


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_@][A-Za-z0-9_@-]*$")


def _format_path(elements) -> str:
    """Render a structural path in canonical JSONPath notation."""
    parts = ["$"]
    for kind, value in elements:
        if kind == "f":
            if _IDENTIFIER_RE.match(value):
                parts.append("." + value)
            else:
                parts.append("[%r]" % value)
        else:
            parts.append("[%d]" % value)
    return "".join(parts)


def _find_paths(ast, document) -> Optional[list]:
    """All full paths matched by ``ast`` on ``document`` (None if it errors)."""
    try:
        return [datum.full_path for datum in ast.find(document)]
    except Exception:
        return None


def _verify_counterexample(new_ast, allowed_asts, document, path_ast) -> bool:
    """Replay the witness with find(): matched by new, covered by no allowed."""
    new_paths = _find_paths(new_ast, document)
    if new_paths is None or path_ast not in new_paths:
        return False
    for allowed_ast in allowed_asts:
        allowed_paths = _find_paths(allowed_ast, document)
        if allowed_paths is not None and path_ast in allowed_paths:
            return False
    return True


def _strip_patterns(patterns) -> tuple:
    """
    Reduce (absolute, guard, steps) patterns to plain step sequences.

    Raises _Unsupported if any pattern's root reset is guarded by a
    non-trivial prefix (conditional matching is out of scope).
    """
    stripped = []
    for _, guard, steps in sorted(patterns, key=_pattern_key):
        if any(len(prefix) > 0 for prefix in guard):
            raise _Unsupported("'$' in a non-initial position with a non-trivial prefix")
        stripped.append(steps)
    return tuple(stripped)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _as_ast(selector) -> JSONPath:
    if isinstance(selector, JSONPath):
        return selector
    if isinstance(selector, str):
        return _parse(selector)
    raise JSONPathError("expected a JSONPath expression or string, got %r" % type(selector).__name__)


def check_relation(selector, allowed: Iterable = ()) -> RelationResult:
    """
    Statically compare ``selector`` against the set of allowed selectors.

    ``selector`` and each entry of ``allowed`` may be a JSONPath string or an
    already-parsed ``JSONPath`` expression.  Returns a :class:`RelationResult`
    whose ``relation`` is CONTAINED, NOT_CONTAINED (with a replayable
    counterexample) or UNKNOWN (for unsupported constructs).
    """
    try:
        new_ast = _as_ast(selector)
    except JSONPathError as exc:
        return RelationResult(Relation.UNKNOWN, "cannot parse selector: %s" % exc)

    allowed = list(allowed)
    allowed_asts = []
    for entry in allowed:
        try:
            allowed_asts.append(_as_ast(entry))
        except JSONPathError as exc:
            return RelationResult(
                Relation.UNKNOWN, "cannot parse allowed selector %r: %s" % (entry, exc)
            )

    try:
        new_patterns = _normalize(new_ast)
    except (_Unsupported, _TooHard) as exc:
        return RelationResult(Relation.UNKNOWN, "selector %s: %s" % (selector, exc))

    allowed_patterns = []
    for entry, entry_ast in zip(allowed, allowed_asts):
        try:
            allowed_patterns.extend(_normalize(entry_ast))
        except (_Unsupported, _TooHard) as exc:
            return RelationResult(Relation.UNKNOWN, "allowed selector %r: %s" % (entry, exc))

    # At the top level both relative and absolute patterns are simply sets
    # of paths from the document root, so the absoluteness flag is dropped.
    # Patterns whose root reset is guarded by a non-trivial prefix match
    # conditionally, which the path-set model cannot express: reject them.
    try:
        new_patterns = _strip_patterns(new_patterns)
    except _Unsupported as exc:
        return RelationResult(Relation.UNKNOWN, "selector %s: %s" % (selector, exc))
    try:
        allowed_patterns = _strip_patterns(allowed_patterns)
    except _Unsupported as exc:
        return RelationResult(Relation.UNKNOWN, "allowed selectors: %s" % exc)

    try:
        candidates = _witness_cells(new_patterns, allowed_patterns)
        first_elements = None
        for attempt, cells in enumerate(candidates):
            elements = tuple(_pick_element(cell) for cell in cells)
            if first_elements is None:
                first_elements = elements
            document = _build_document(elements)
            path_ast = _path_jsonpath(list(elements))
            if _verify_counterexample(new_ast, allowed_asts, document, path_ast):
                return RelationResult(
                    Relation.NOT_CONTAINED,
                    "selector can match %s, which is outside the allowed set"
                    % _format_path(elements),
                    Counterexample(document, _format_path(elements), True, elements),
                )
            if attempt >= 7:
                break
    except _TooHard as exc:
        return RelationResult(Relation.UNKNOWN, "analysis too complex: %s" % exc)

    if first_elements is None:
        return RelationResult(
            Relation.CONTAINED,
            "every structural path the selector may match is covered by the allowed set",
        )

    # The language difference is non-empty but no candidate could be
    # replay-verified (defensive fallback; should not happen).
    return RelationResult(
        Relation.NOT_CONTAINED,
        "selector can match %s, which is outside the allowed set"
        " (counterexample could not be replay-verified)" % _format_path(first_elements),
        Counterexample(_build_document(first_elements), _format_path(first_elements), False, first_elements),
    )


def is_covered(selector, allowed: Iterable = ()) -> bool:
    """
    True only when ``selector`` is provably contained in the allowed set.

    Both NOT_CONTAINED and UNKNOWN yield False, so authorization callers
    fail closed.
    """
    return check_relation(selector, allowed).contained


def replay(counterexample: Counterexample, selector, allowed: Iterable = ()) -> bool:
    """
    Replay a counterexample with find(): returns True when ``selector``
    matches ``counterexample.path`` on ``counterexample.document`` and no
    allowed selector does.
    """
    new_ast = _as_ast(selector)
    allowed_asts = [_as_ast(entry) for entry in allowed]
    path_ast = _path_jsonpath(list(counterexample._elements)) if counterexample._elements else _parse(counterexample.path)
    return _verify_counterexample(new_ast, allowed_asts, counterexample.document, path_ast)
