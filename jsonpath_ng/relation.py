"""
Static containment checking for JSONPath selectors.

This module answers the authorization question: "does a candidate selector
stay within the locations covered by a set of already-allowed selectors?"

Two selectors are compared by the *sets of structural paths they may
match*.  A structural path is a sequence of object field names and array
indices rooted at ``$``.  A selector ``S`` denotes the set ``L(S)`` of
structural paths that ``S.find(document)`` can produce for *some* JSON
document, and the candidate is considered covered when ``L(candidate)`` is
a subset of the union of ``L(allowed)`` over every allowed selector.  This
is a genuine set comparison: ``$.a`` does *not* cover ``$.ab`` merely
because one concrete string is a prefix of the other.

Supported selector constructs (anything else yields ``Relation.UNKNOWN``):

* the root ``$`` (only at the start of a selector) and `` `this` ``,
* object fields, including quoted names and ``*`` (any field),
* the descendant operator ``..``,
* non-negative array indices (``[0]``, ``[1,3]``),
* slices with non-negative bounds and a finite, positive step
  (``[*]``, ``[1:]``, ``[:3]``, ``[0:8:2]``),
* unions ``|`` and comma-separated fields/indices.

Filters (``[?(...)]``), ``where``/``wherenot``, intersections ``&``,
`` `parent` ``, negative indices, negative or zero slice steps, and every
extension (arithmetic, string functions, ...) are deliberately out of
scope: they produce ``Relation.UNKNOWN`` so that callers can deny the
request instead of trusting an analyzer guess.

Three outcomes are possible, see :class:`Relation`:

* ``CONTAINED`` -- the candidate provably stays within the allowed union.
* ``NOT_CONTAINED`` -- the candidate may escape the allowed union.  The
  result carries a small ``witness_document`` and a ``witness_path`` such
  that replaying ``candidate.find(witness_document)`` reaches
  ``witness_path`` while no allowed selector does.
* ``UNKNOWN`` -- the expression exceeds the supported fragment (or an
  internal resource limit was hit).  Callers should treat this as a
  denial, not as an analyzer guess.

Caveat: JSON text values are treated as atomic scalars.  ``jsonpath_ng``
happens to let ``Index`` reach *into* strings (``$[0]`` on ``"ab"``); that
quirk is not modeled.
"""

from __future__ import annotations

import enum
import re
from collections import deque
from dataclasses import dataclass
from math import gcd
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .exceptions import JSONPathError
from .jsonpath import (
    Child,
    Descendants,
    Fields,
    Index,
    JSONPath,
    Root,
    Slice,
    This,
    Union,
)
from .parser import parse

__all__ = [
    "Relation",
    "RelationResult",
    "check_relation",
    "check_containment",
    "is_covered",
]


# Resource limits.  Exceeding any of them aborts the analysis with
# ``Relation.UNKNOWN`` rather than burning unbounded time or memory.
_MAX_PATTERNS = 512        # path patterns a single selector may expand to
_MAX_PATTERN_LENGTH = 128  # steps in a single expanded path pattern
_MAX_FRONTIER = 64         # distinct allowed matchers per automaton frontier
_MAX_STATE_SETS = 4096     # automaton state sets explored per descendant star
_MAX_MEMO_ENTRIES = 20000  # total memoized subproblems
_MAX_LCM = 4096            # lcm of slice steps for index-set arithmetic
_MAX_INDEX_SCAN = 1000000  # bound for brute-force index-set scans


class Relation(enum.Enum):
    """The three possible outcomes of a static containment check."""

    CONTAINED = "contained"
    NOT_CONTAINED = "not_contained"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RelationResult:
    """
    The outcome of :func:`check_relation`.

    ``witness_document`` and ``witness_path`` are only set for
    ``Relation.NOT_CONTAINED``: replaying the candidate with
    ``candidate.find(witness_document)`` reaches ``witness_path`` while no
    allowed selector does.
    """

    relation: Relation
    reason: str = ""
    witness_document: Any = None
    witness_path: Optional[JSONPath] = None

    @property
    def contained(self) -> bool:
        """True only when containment was positively proven."""
        return self.relation is Relation.CONTAINED

    def __bool__(self) -> bool:
        return self.contained


class _Unsupported(Exception):
    """Raised internally when an expression exceeds the supported fragment."""


# ---------------------------------------------------------------------------
# Step matchers
# ---------------------------------------------------------------------------
#
# A compiled selector is a list of *patterns*.  A pattern is a tuple of
# steps; each step is a ``_FieldSet``, the ``_ANY_FIELD`` singleton, a
# ``_IndexSet``, or the ``_STAR`` singleton (an arbitrary sequence of
# descendant steps, produced by ``..``).  ``_ANY_STEP`` only appears as a
# frontier matcher of the allowed-side automaton.


class _Singleton:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


_ANY_FIELD = _Singleton("ANY_FIELD")  # any object field name
_ANY_STEP = _Singleton("ANY_STEP")    # any single field or index step
_STAR = _Singleton("STAR")            # any sequence of steps (from ``..``)


class _FieldSet:
    """Matches exactly one of a fixed set of object field names."""

    __slots__ = ("names",)

    def __init__(self, names: Iterable[str]):
        self.names = frozenset(names)

    def __eq__(self, other):
        return isinstance(other, _FieldSet) and self.names == other.names

    def __hash__(self):
        return hash(self.names)

    def __repr__(self):
        return "fields%r" % (sorted(self.names),)


class _IndexSet:
    """
    A set of non-negative array indices, expressed as a union of arithmetic
    progressions ``(start, step, end)`` meaning ``start + k*step < end``
    (``end=None`` is unbounded).  Slices and indices both compile to this:
    a slice denotes exactly the indices it can select for *some* array
    length, which is what the path-set semantics compares.
    """

    __slots__ = ("aps",)

    def __init__(self, aps: Iterable[Tuple[int, int, Optional[int]]] = ()):
        cleaned = []
        for start, step, end in aps:
            if end is not None and start >= end:
                continue
            cleaned.append((start, step, end))
        self.aps = tuple(sorted(cleaned))

    def __eq__(self, other):
        return isinstance(other, _IndexSet) and self.aps == other.aps

    def __hash__(self):
        return hash(self.aps)

    def __repr__(self):
        return "indices%r" % (self.aps,)

    @property
    def is_empty(self) -> bool:
        return not self.aps

    def matches(self, index: int) -> bool:
        return any(
            start <= index
            and (end is None or index < end)
            and (index - start) % step == 0
            for start, step, end in self.aps
        )

_NATURALS = _IndexSet(((0, 1, None),))


def _lcm(a: int, b: int) -> int:
    result = a // gcd(a, b) * b
    if result > _MAX_LCM:
        raise _Unsupported("slice steps are too large to compare exactly")
    return result


def _matches(matcher: Any, step: Any) -> bool:
    """Whether an atomic matcher accepts the concrete step ``step``."""
    if isinstance(matcher, _FieldSet):
        return step in matcher.names
    if matcher is _ANY_FIELD:
        return isinstance(step, str)
    if matcher is _ANY_STEP:
        return True
    return isinstance(step, int) and matcher.matches(step)


# ---------------------------------------------------------------------------
# Compiling selectors into path patterns
# ---------------------------------------------------------------------------


def _check_caps(patterns: List[tuple]) -> List[tuple]:
    if len(patterns) > _MAX_PATTERNS:
        raise _Unsupported("selector expands to too many path patterns")
    for pattern in patterns:
        if len(pattern) > _MAX_PATTERN_LENGTH:
            raise _Unsupported("selector expands to too many path steps")
    return patterns


def _compile_slice(expr: Slice) -> _IndexSet:
    for value in (expr.start, expr.end, expr.step):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise _Unsupported("slice bounds must be integers")
    start, end, step = expr.start, expr.end, expr.step
    if start is not None and start < 0:
        raise _Unsupported("negative slice starts are not supported")
    if end is not None and end < 0:
        raise _Unsupported("negative slice ends are not supported")
    if step is not None and step < 1:
        raise _Unsupported("slice steps must be positive integers")
    start = 0 if start is None else start
    step = 1 if step is None else step
    if end is not None and start >= end:
        return _IndexSet(())
    return _IndexSet(((start, step, end),))


def _product(left: List[tuple], right: List[tuple], star: bool = False) -> List[tuple]:
    middle = (_STAR,) if star else ()
    return _check_caps([l + middle + r for l in left for r in right])


def _compile(expr: JSONPath, allow_root: bool = True) -> List[tuple]:
    """
    Compile a selector into a list of path patterns.

    ``allow_root`` tracks whether ``$`` may still denote the document root:
    it does on the left spine of a selector (where the expression is only
    ever evaluated against the context-free top datum), but not inside the
    right-hand side of a child or descendant, where ``Root.find`` would
    climb the match context instead.
    """
    if isinstance(expr, Root):
        if not allow_root:
            raise _Unsupported("`$` is only supported at the start of a selector")
        return [()]
    if isinstance(expr, This):
        return [()]
    if isinstance(expr, Fields):
        if "*" in expr.fields:
            return [(_ANY_FIELD,)]
        return [(_FieldSet(expr.fields),)]
    if isinstance(expr, Index):
        indices = []
        for index in expr.indices:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise _Unsupported(
                    "array index %r is not a non-negative integer" % (index,)
                )
            indices.append(index)
        return [(_IndexSet((i, 1, i + 1) for i in indices),)]
    if isinstance(expr, Slice):
        return [(_compile_slice(expr),)]
    if isinstance(expr, Child):
        left = _compile(expr.left, allow_root=allow_root)
        right = _compile(expr.right, allow_root=False)
        return _product(left, right)
    if isinstance(expr, Union):
        return _check_caps(
            _compile(expr.left, allow_root=allow_root)
            + _compile(expr.right, allow_root=allow_root)
        )
    if isinstance(expr, Descendants):
        left = _compile(expr.left, allow_root=allow_root)
        right = _compile(expr.right, allow_root=False)
        return _product(left, right, star=True)
    raise _Unsupported("expressions of type %s are not supported" % type(expr).__name__)


# ---------------------------------------------------------------------------
# The allowed-side automaton
# ---------------------------------------------------------------------------


class _CoverageAutomaton:
    """
    A small NFA over the allowed patterns.  States are ``(pattern_index,
    position)`` pairs; a ``_STAR`` step consumes any single step in place
    and may also be skipped.  A concrete path is covered iff the automaton
    reaches an accepting state (the end of any pattern) after consuming it.
    """

    def __init__(self, patterns: Sequence[tuple]):
        self.patterns = [tuple(p) for p in patterns]
        self.initial = self._closure(
            frozenset((idx, 0) for idx in range(len(self.patterns)))
        )

    def _closure(self, states):
        result = set(states)
        stack = list(states)
        while stack:
            pattern_idx, pos = stack.pop()
            pattern = self.patterns[pattern_idx]
            if pos < len(pattern) and pattern[pos] is _STAR:
                nxt = (pattern_idx, pos + 1)
                if nxt not in result:
                    result.add(nxt)
                    stack.append(nxt)
        return frozenset(result)

    def accepts(self, states) -> bool:
        return any(pos == len(self.patterns[idx]) for idx, pos in states)

    def frontier(self, states) -> List[Any]:
        """Distinct atomic matchers that can consume the next step."""
        matchers = []
        for idx, pos in states:
            pattern = self.patterns[idx]
            if pos >= len(pattern):
                continue
            element = pattern[pos]
            matchers.append(_ANY_STEP if element is _STAR else element)
        return list(dict.fromkeys(matchers))

    def delta(self, states, step):
        advanced = set()
        for idx, pos in states:
            pattern = self.patterns[idx]
            if pos >= len(pattern):
                continue
            element = pattern[pos]
            if element is _STAR:
                advanced.add((idx, pos))
            elif _matches(element, step):
                advanced.add((idx, pos + 1))
        return self._closure(frozenset(advanced))


# ---------------------------------------------------------------------------
# Region representatives
# ---------------------------------------------------------------------------
#
# To check a candidate step against the allowed frontier we partition the
# candidate step's domain by *signature*: the tuple of frontier matchers
# that accept a given concrete step.  The automaton transition only depends
# on the signature, so checking one representative concrete step per
# signature decides the whole domain exactly.


def _fresh_name(reserved: set) -> str:
    counter = 0
    while True:
        name = "__fresh_%d__" % counter
        if name not in reserved:
            return name
        counter += 1


def _region_representatives(frontier, candidate_matcher, reserved) -> List[Any]:
    """One representative concrete step per achievable frontier signature."""
    if len(frontier) > _MAX_FRONTIER:
        raise _Unsupported("too many distinct allowed matchers at one position")
    field_sets = [m for m in frontier if isinstance(m, _FieldSet)]
    index_sets = [m for m in frontier if isinstance(m, _IndexSet)]

    representatives: List[Any] = []

    # Field-name steps.  ``_ANY_FIELD``/``_ANY_STEP`` frontier matchers
    # accept every name and index sets accept none, so a signature is
    # determined by which ``_FieldSet`` matchers contain the name.
    if isinstance(candidate_matcher, _FieldSet):
        names_to_try = sorted(candidate_matcher.names)
    elif candidate_matcher is _ANY_FIELD or candidate_matcher is _ANY_STEP:
        pool: set = set()
        for field_set in field_sets:
            pool |= field_set.names
        # A synthesized name stands in for every name no frontier field
        # set mentions.
        names_to_try = sorted(pool) + [_fresh_name(reserved | pool)]
    else:
        names_to_try = []
    seen_name_signatures = set()
    for name in names_to_try:
        signature = tuple(name in field_set.names for field_set in field_sets)
        if signature not in seen_name_signatures:
            seen_name_signatures.add(signature)
            representatives.append(name)

    # Array-index steps.  Field matchers accept no index and ``_ANY_STEP``
    # accepts them all, so a signature is determined by which frontier
    # index sets contain the index.  Frontier memberships are periodic
    # beyond the largest frontier start/end bound, so scanning the
    # candidate elements below that bound plus one period of each
    # candidate progression finds every achievable signature.
    if isinstance(candidate_matcher, _IndexSet):
        candidate_indices = candidate_matcher
    elif candidate_matcher is _ANY_STEP:
        candidate_indices = _NATURALS
    else:
        candidate_indices = None
    if candidate_indices is not None and candidate_indices.aps:
        frontier_bound = 0
        frontier_period = 1
        for index_set in index_sets:
            for start, step, end in index_set.aps:
                frontier_bound = max(frontier_bound, start, end if end is not None else 0)
                frontier_period = _lcm(frontier_period, step)
        seen_index_signatures = set()
        scanned = 0

        def visit(index: int) -> None:
            signature = tuple(index_set.matches(index) for index_set in index_sets)
            if signature not in seen_index_signatures:
                seen_index_signatures.add(signature)
                representatives.append(index)

        for start, step, end in candidate_indices.aps:
            bounded_end = frontier_bound if end is None else min(end, frontier_bound)
            value = start
            while value < bounded_end:
                scanned += 1
                if scanned > _MAX_INDEX_SCAN:
                    raise _Unsupported("array index sets are too large to compare")
                visit(value)
                value += step
            tail_start = max(start, frontier_bound)
            if end is None or tail_start < end:
                skip = (tail_start - start + step - 1) // step
                value = start + skip * step
                period = frontier_period // gcd(step, frontier_period)
                for _ in range(period):
                    if end is not None and value >= end:
                        break
                    visit(value)
                    value += step

    return representatives


# ---------------------------------------------------------------------------
# The containment search
# ---------------------------------------------------------------------------


def _uncovered_steps(patterns, pattern_idx, pos, states, automaton, reserved, memo):
    """
    ``None`` if every concrete path accepted by ``patterns[pattern_idx][pos:]``
    is covered by the automaton from ``states``; otherwise a concrete
    counterexample path (tuple of steps), minimal in a breadth-first sense.
    """
    key = (pattern_idx, pos, states)
    if key in memo:
        return memo[key]
    if len(memo) > _MAX_MEMO_ENTRIES:
        raise _Unsupported("containment search exploded; giving up")
    pattern = patterns[pattern_idx]
    if pos == len(pattern):
        result = None if automaton.accepts(states) else ()
    elif pattern[pos] is _STAR:
        # The descendant star generates an arbitrary step sequence, so the
        # rest of the pattern must be covered from *every* automaton state
        # set reachable by consuming arbitrary steps.  There are finitely
        # many such sets; enumerate them breadth-first.
        result = None
        seen = {states}
        queue = deque([(states, ())])
        while queue and result is None:
            current, prefix = queue.popleft()
            tail = _uncovered_steps(
                patterns, pattern_idx, pos + 1, current, automaton, reserved, memo
            )
            if tail is not None:
                result = prefix + tail
                break
            for rep in _region_representatives(
                automaton.frontier(current), _ANY_STEP, reserved
            ):
                nxt = automaton.delta(current, rep)
                if nxt not in seen:
                    seen.add(nxt)
                    if len(seen) > _MAX_STATE_SETS:
                        raise _Unsupported("too many automaton states; giving up")
                    queue.append((nxt, prefix + (rep,)))
    else:
        result = None
        for rep in _region_representatives(
            automaton.frontier(states), pattern[pos], reserved
        ):
            tail = _uncovered_steps(
                patterns, pattern_idx, pos + 1,
                automaton.delta(states, rep), automaton, reserved, memo,
            )
            if tail is not None:
                result = (rep,) + tail
                break
    memo[key] = result
    return result


def _find_counterexample(candidate_patterns, allowed_patterns, reserved):
    """A concrete path matched by the candidate but not covered, if one exists."""
    automaton = _CoverageAutomaton(allowed_patterns)
    memo: dict = {}
    for idx, pattern in enumerate(candidate_patterns):
        witness = _uncovered_steps(
            candidate_patterns, idx, 0, automaton.initial, automaton, reserved, memo
        )
        if witness is not None:
            return list(witness)
    return None


# ---------------------------------------------------------------------------
# Witnesses
# ---------------------------------------------------------------------------


def _build_document(steps: Sequence) -> Any:
    """The smallest JSON document providing the concrete path ``steps``."""
    document: Any = None
    for step in reversed(steps):
        if isinstance(step, str):
            document = {step: document}
        else:
            document = [None] * step + [document]
    return document


def _path_expression(steps: Sequence) -> JSONPath:
    """A concrete JSONPath matching exactly the location ``steps`` points to."""
    expr: JSONPath = Root()
    for step in steps:
        expr = expr.child(Fields(step) if isinstance(step, str) else Index(step))
    return expr


def _render_steps(steps: Sequence) -> str:
    rendered = ["$"]
    for step in steps:
        if isinstance(step, str):
            if re.match(r"^[A-Za-z_@][A-Za-z0-9_@-]*$", step):
                rendered.append("." + step)
            else:
                rendered.append("[%r]" % step)
        else:
            rendered.append("[%d]" % step)
    return "".join(rendered)


def _verify_witness(candidate_exprs, allowed_exprs, document, path) -> bool:
    """
    Replay the counterexample with ``find``: the candidate must actually
    reach ``path`` in ``document`` while no allowed selector does.  This
    grounds the ``NOT_CONTAINED`` verdict in the real matching semantics.
    """

    def reaches(expr) -> bool:
        try:
            return any(datum.full_path == path for datum in expr.find(document))
        except Exception:
            return False

    if not any(reaches(expr) for expr in candidate_exprs):
        return False
    return not any(reaches(expr) for expr in allowed_exprs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _as_list(value) -> list:
    if isinstance(value, (str, JSONPath)):
        return [value]
    try:
        return list(value)
    except TypeError:
        raise _Unsupported("selectors must be strings or JSONPath expressions")


def _parse_one(item) -> JSONPath:
    if isinstance(item, JSONPath):
        return item
    if not isinstance(item, str):
        raise _Unsupported(
            "selector %r is neither a string nor a JSONPath" % (item,)
        )
    try:
        return parse(item)
    except JSONPathError:
        pass
    try:
        from .ext import parse as ext_parse

        return ext_parse(item)
    except JSONPathError:
        raise _Unsupported("selector %r cannot be parsed" % (item,))


def _reserved_names(*pattern_lists) -> set:
    names = set()
    for patterns in pattern_lists:
        for pattern in patterns:
            for element in pattern:
                if isinstance(element, _FieldSet):
                    names |= element.names
    return names


def check_relation(candidate, allowed=()) -> RelationResult:
    """
    Statically check whether ``candidate`` stays within ``allowed``.

    ``candidate`` is a JSONPath selector (string or parsed expression, or
    an iterable of those, treated as a union).  ``allowed`` is the
    collection of already-permitted selectors.  Returns a
    :class:`RelationResult`; see :class:`Relation` for the three outcomes.
    Callers making authorization decisions should only proceed on
    ``Relation.CONTAINED``.
    """
    try:
        candidate_exprs = [_parse_one(item) for item in _as_list(candidate)]
        candidate_patterns: List[tuple] = []
        for expr in candidate_exprs:
            candidate_patterns.extend(_compile(expr))
    except _Unsupported as exc:
        return RelationResult(Relation.UNKNOWN, reason="candidate: %s" % exc)

    allowed_exprs = []
    allowed_patterns: List[tuple] = []
    unsupported_allowed = 0
    try:
        allowed_items = _as_list(allowed)
    except _Unsupported as exc:
        return RelationResult(Relation.UNKNOWN, reason="allowed: %s" % exc)
    for item in allowed_items:
        try:
            expr = _parse_one(item)
            patterns = _compile(expr)
        except _Unsupported:
            unsupported_allowed += 1
            continue
        allowed_exprs.append(expr)
        allowed_patterns.extend(patterns)

    reserved = _reserved_names(candidate_patterns, allowed_patterns)
    try:
        witness = _find_counterexample(candidate_patterns, allowed_patterns, reserved)
    except _Unsupported as exc:
        return RelationResult(Relation.UNKNOWN, reason=str(exc))

    if witness is None:
        return RelationResult(
            Relation.CONTAINED,
            reason="candidate is covered by the allowed selectors",
        )

    if unsupported_allowed:
        # The candidate escapes every allowed selector we could model, but
        # the coverage of the unsupported ones is unknowable statically.
        return RelationResult(
            Relation.UNKNOWN,
            reason=(
                "candidate is not covered by the supported allowed selectors and "
                "%d allowed selector(s) could not be analyzed" % unsupported_allowed
            ),
        )

    document = _build_document(witness)
    path = _path_expression(witness)
    if not _verify_witness(candidate_exprs, allowed_exprs, document, path):
        return RelationResult(
            Relation.UNKNOWN,
            reason="counterexample could not be replayed; refusing to guess",
        )
    return RelationResult(
        Relation.NOT_CONTAINED,
        reason=(
            "candidate can reach %s which no allowed selector covers"
            % _render_steps(witness)
        ),
        witness_document=document,
        witness_path=path,
    )


def check_containment(candidate, allowed=()) -> RelationResult:
    """Alias for :func:`check_relation`."""
    return check_relation(candidate, allowed)


def is_covered(candidate, allowed=()) -> bool:
    """True only when ``candidate`` is provably covered by ``allowed``."""
    return check_relation(candidate, allowed).contained
