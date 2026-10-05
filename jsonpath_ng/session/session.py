"""Query sessions with incremental maintenance across JSON Patch batches."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from jsonpath_ng.ext import parse as parse

from ..jsonpath import DatumInContext
from .analysis import Step, UnsupportedQuery, analyze
from . import engine
from .patch import PatchBatchError  # re-exported
from .patch import apply_batch, json_equal
from .pointer import AccessCounter, Pointer, format_pointer, is_prefix, resolve


def _is_prefix(prefix: Pointer, path: Pointer) -> bool:
    return is_prefix(prefix, path)

# Re-export so callers can import the exception from the package root.


@dataclass
class MatchDelta:
    """The change in one query's match set for a single batch."""

    expression: str
    added: List[Tuple[str, Any]] = field(default_factory=list)
    removed: List[Tuple[str, Any]] = field(default_factory=list)
    value_changed: List[Tuple[str, Any, Any]] = field(default_factory=list)
    moved: List[Tuple[str, str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (
            self.added or self.removed or self.value_changed or self.moved
        )


@dataclass
class SessionStats:
    """Counters for one :meth:`QuerySession.apply_patches` call."""

    batches_applied: int = 0
    operations_applied: int = 0
    full_recomputations: int = 0
    incremental_updates: int = 0
    nodes_visited: int = 0
    fallback_queries: Dict[str, str] = field(default_factory=dict)
    fallback_used: List[str] = field(default_factory=list)
    conservative_reasons: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "batches_applied": self.batches_applied,
            "operations_applied": self.operations_applied,
            "full_recomputations": self.full_recomputations,
            "incremental_updates": self.incremental_updates,
            "nodes_visited": self.nodes_visited,
            "fallback_queries": dict(self.fallback_queries),
            "fallback_used_this_batch": list(self.fallback_used),
            "conservative_reasons": list(self.conservative_reasons),
        }


class SessionQuery:
    """A registered query and its current match snapshot."""

    def __init__(self, expression: str, expr: Any, steps: Optional[List[Step]]):
        self.expression = expression
        self.expr = expr
        self.steps = steps
        self.supported = steps is not None
        self.fallback_reason: Optional[str] = None
        if not self.supported and isinstance(steps, UnsupportedQuery):
            self.fallback_reason = str(steps)
        # pointer -> stored value (an independent snapshot)
        self.matches: Dict[Pointer, Any] = {}

    def matches_as_pointers(self) -> List[str]:
        return [format_pointer(p) for p in sorted(self.matches, key=_sort_key)]

    def values(self) -> List[Any]:
        return [self.matches[p] for p in sorted(self.matches, key=_sort_key)]


def _sort_key(ptr: Pointer):
    return [("/i/" + str(t) if isinstance(t, int) else "/s/" + str(t)) for t in ptr]


class QuerySession:
    """Holds a document and one or more queries, maintained incrementally.

    Parameters
    ----------
    document:
        The JSON document. It is owned by the session and mutated in place.
    expressions:
        One or more JSONPath expressions (extended dialect).
    """

    def __init__(self, document: Any, *expressions: str):
        self.document = document
        self.stats = SessionStats()
        self._counter = AccessCounter()
        self.queries: List[SessionQuery] = []
        for expression in expressions:
            self.register(expression)
        self._initial_find()

    # -- registration --------------------------------------------------------

    def register(self, expression: str) -> SessionQuery:
        expr = parse(expression)
        steps: Any
        try:
            steps = analyze(expr)
        except UnsupportedQuery as exc:
            steps = exc
            query = SessionQuery(expression, expr, None)
            query.fallback_reason = str(exc)
            self.stats.fallback_queries[expression] = str(exc)
            self.queries.append(query)
            return query
        query = SessionQuery(expression, expr, steps)
        self.queries.append(query)
        return query

    def _initial_find(self):
        for query in self.queries:
            query.matches = self._full_find(query)

    def _full_find(self, query: SessionQuery) -> Dict[Pointer, Any]:
        self._counter.reset()
        wrapped = self._counter.wrap(self.document)
        try:
            found = query.expr.find(
                DatumInContext(wrapped, path=None, context=None)
            )
        finally:
            self.stats.nodes_visited += self._counter.visited
        matches: Dict[Pointer, Any] = {}
        for datum in found:
            pointer = engine.datum_pointer(datum)
            value = self._counter.unwrap(datum.value)
            matches[pointer] = deepcopy(value)
        return matches

    # -- public API ----------------------------------------------------------

    @property
    def doc(self) -> Any:
        return self.document

    def matches(self, expression: str) -> List[Any]:
        query = self._require_query(expression)
        return query.values()

    def match_pointers(self, expression: str) -> List[str]:
        query = self._require_query(expression)
        return query.matches_as_pointers()

    def apply_patches(self, patches: List[dict]) -> Dict[str, MatchDelta]:
        """Validate then apply *patches*; publish deltas only on full success."""
        events = apply_batch(self.document, patches)
        return self._publish(events, len(patches))

    # -- incremental update --------------------------------------------------

    def _publish(
        self, events: List[tuple], operation_count: int
    ) -> Dict[str, MatchDelta]:
        deltas: Dict[str, MatchDelta] = {}
        conservative = False
        conservative_reason = None

        descriptors = engine.normalize_event_paths(events)
        simulations, nested = engine.structural_state(events, self.document)
        moves = engine.move_index_registry(events)
        move_links = self._move_links(events)

        if nested:
            conservative = True
            conservative_reason = "nested structurally edited arrays in one batch"

        if not conservative and move_links:
            struct_lists = {
                e[1] for e in events if e[0] in ("insert", "remove")
            }
            for src, dst in move_links.items():
                if any(
                    _is_prefix(src, lp) or _is_prefix(dst, lp)
                    for lp in struct_lists
                ):
                    conservative = True
                    conservative_reason = (
                        "move combined with array edits in the moved subtree"
                    )
                    break

        for query in self.queries:
            delta = MatchDelta(query.expression)
            if not query.supported or conservative:
                self._recompute(
                    query, delta,
                    events if query.supported else None,
                    simulations, moves, move_links,
                )
                if not query.supported:
                    reason = query.fallback_reason or "unsupported query"
                    self.stats.fallback_used.append(query.expression)
                    self.stats.conservative_reasons.append(
                        f"{query.expression}: {reason}"
                    )
                else:
                    self.stats.conservative_reasons.append(
                        f"{query.expression}: {conservative_reason}"
                    )
            else:
                self._incremental(
                    query, descriptors, events, simulations, moves, move_links, delta
                )
                self.stats.incremental_updates += 1
            deltas[query.expression] = delta

        self.stats.batches_applied += 1
        self.stats.operations_applied += operation_count
        return deltas

    def _incremental(
        self,
        query: SessionQuery,
        descriptors,
        events,
        simulations,
        moves,
        move_links,
        delta: MatchDelta,
    ):
        probes = engine.plan_probes(query.steps, descriptors)
        old_candidates: Dict[Pointer, Any] = {}
        new_candidates: Dict[Pointer, Any] = {}
        for kind, anchor, suffix in probes:
            old_anchor, old_node = engine.old_anchor_for(
                anchor, events, self.document, simulations, moves
            )
            if old_anchor is not None:
                self._eval_probe(
                    old_anchor, list(suffix), old_node, old_candidates, count=True
                )
            new_node = resolve(self.document, anchor, None)
            self._eval_probe(anchor, list(suffix), new_node, new_candidates, count=True)

        old_pointers = set(query.matches)
        covered_old = set(old_candidates)

        translated: Dict[Pointer, Optional[Pointer]] = {}
        for pointer in old_pointers:
            translated[pointer] = engine.translate_through(pointer, simulations, moves)
            for src, dst in move_links.items():
                if pointer == src:
                    translated[pointer] = dst

        # New snapshot: untouched matches survive at their translated location.
        new_matches: Dict[Pointer, Any] = {}
        for pointer, value in query.matches.items():
            mapped = translated.get(pointer, pointer)
            if pointer not in covered_old and mapped is not None:
                new_matches[mapped] = value

        reachable_new = {m for m in translated.values() if m is not None}
        changed_pairs: List[Tuple[Pointer, Pointer]] = []

        for old_pointer, old_value in old_candidates.items():
            mapped = translated.get(old_pointer)
            if mapped is not None and mapped in new_candidates:
                new_value = new_candidates[mapped]
                new_matches[mapped] = deepcopy(new_value)
                if not json_equal(old_value, new_value):
                    changed_pairs.append((old_pointer, mapped))
                    delta.value_changed.append((
                        format_pointer(mapped),
                        deepcopy(old_value),
                        deepcopy(new_value),
                    ))
                continue
            delta.removed.append((format_pointer(old_pointer), deepcopy(old_value)))

        for pointer, value in new_candidates.items():
            new_matches.setdefault(pointer, deepcopy(value))
            if pointer not in reachable_new:
                delta.added.append((format_pointer(pointer), deepcopy(value)))

        # Structural deletions outside probe coverage.
        for pointer in old_pointers - covered_old:
            mapped = translated.get(pointer, pointer)
            if mapped is None:
                delta.removed.append(
                    (format_pointer(pointer), deepcopy(query.matches[pointer]))
                )
                new_matches.pop(pointer, None)

        # Pure positional moves: value preserved, only the pointer changed.
        changed_set = set(changed_pairs)
        for old_pointer, new_pointer in translated.items():
            if (
                new_pointer is not None
                and old_pointer != new_pointer
                and (old_pointer, new_pointer) not in changed_set
                and old_pointer in query.matches
                and new_pointer in new_candidates
                and json_equal(
                    query.matches[old_pointer], new_candidates[new_pointer]
                )
            ):
                delta.moved.append((
                    format_pointer(old_pointer),
                    format_pointer(new_pointer),
                    deepcopy(new_candidates[new_pointer]),
                ))

        query.matches = new_matches
        self._order_delta(delta)

    def _recompute(
        self, query: SessionQuery, delta: MatchDelta,
        events=None, simulations=None, moves=None, move_links=None,
    ):
        old = query.matches
        new = self._full_find(query)
        old_ptrs = set(old)
        new_ptrs = set(new)

        translated: Dict[Pointer, Optional[Pointer]] = {p: p for p in old_ptrs}
        if events is not None and simulations is not None:
            for pointer in old_ptrs:
                translated[pointer] = engine.translate_through(
                    pointer, simulations, moves or {}
                )
                for src, dst in (move_links or {}).items():
                    if pointer == src:
                        translated[pointer] = dst

        surviving_old = set()
        claimed_new = set()
        for old_pointer in old_ptrs:
            mapped = translated[old_pointer]
            if mapped is not None and mapped in new_ptrs:
                surviving_old.add(old_pointer)
                claimed_new.add(mapped)
                if not json_equal(old[old_pointer], new[mapped]):
                    delta.value_changed.append((
                        format_pointer(mapped),
                        deepcopy(old[old_pointer]),
                        deepcopy(new[mapped]),
                    ))
                elif old_pointer != mapped:
                    delta.moved.append((
                        format_pointer(old_pointer),
                        format_pointer(mapped),
                        deepcopy(new[mapped]),
                    ))
        for pointer in sorted(new_ptrs - claimed_new, key=_sort_key):
            delta.added.append((format_pointer(pointer), deepcopy(new[pointer])))
        for pointer in sorted(old_ptrs - surviving_old, key=_sort_key):
            delta.removed.append((format_pointer(pointer), deepcopy(old[pointer])))

        query.matches = new
        self.stats.full_recomputations += 1
        self._order_delta(delta)

    def _eval_probe(self, anchor, suffix, node_value, sink: Dict[Pointer, Any], count: bool):
        found = engine.evaluate_probe(
            anchor, suffix, node_value, self._counter if count else None
        )
        if count:
            self.stats.nodes_visited += self._counter.visited
            self._counter.reset()
        for pointer, value in found.items():
            sink[pointer] = value

    @staticmethod
    def _move_links(events):
        links = {}
        for event in events:
            if event[0] == "move_link":
                links[event[1]] = event[2]
        return links

    @staticmethod
    def _order_delta(delta: MatchDelta):
        delta.added.sort(key=lambda item: item[0])
        delta.removed.sort(key=lambda item: item[0])
        delta.value_changed.sort(key=lambda item: item[0])
        delta.moved.sort(key=lambda item: (item[0], item[1]))

    def _require_query(self, expression: str) -> SessionQuery:
        for query in self.queries:
            if query.expression == expression:
                return query
        raise KeyError(f"no registered query {expression!r}")
