"""Finite, incremental subscription sessions over versioned documents.

A :class:`SubscriptionSession` owns a bounded number of subscriptions to
parsed JSONPath expressions.  Instead of re-running every expression after a
small change, each subscription records the concrete paths it actually
depends on, and patches are propagated as add/remove/replace events against
the subscription's current hit set, including array movement.

First-version scope:

* field selectors (``$.a.b``), array indices (``[0]``) and wildcards
  (``[*]`` / ``.*``);
* explicit scalar filters (``[?@.score >= 10]``), including conjunctions and
  filters that reference the root document (``[?$.threshold == 5]``).

Anything outside that scope (descendants, slices, unions, arithmetic,
query-to-query comparisons, ...) is still supported: the subscription is
marked non-incremental and explicitly re-evaluated in full after each commit.
"""

from __future__ import annotations

import copy
import typing

from jsonpath_ng import Child
from jsonpath_ng import Fields
from jsonpath_ng import Index
from jsonpath_ng import JSONPath
from jsonpath_ng import Root
from jsonpath_ng import This
from jsonpath_ng import Wildcard
from jsonpath_ng.ext.filter import And
from jsonpath_ng.ext.filter import Expression
from jsonpath_ng.ext.filter import Filter
from jsonpath_ng.ext.filter import Or
from jsonpath_ng.ext.filter import _BooleanOperation
from jsonpath_ng.ext.patch import ABSENT
from jsonpath_ng.ext.patch import ConcretePath
from jsonpath_ng.ext.patch import OperationEvent
from jsonpath_ng.ext.patch import Segment
from jsonpath_ng.ext.patch import VersionedDocument


class SubscriptionLimitError(RuntimeError):
    """Raised when a session cannot register another subscription."""


class SubscriptionClosedError(RuntimeError):
    """Raised when a closed session or subscription is used."""


#: Synthetic path segment marking a node inserted by the current patch.
_INSERTED_MARKER = object()


# ---------------------------------------------------------------------------
# Compiled plans
# ---------------------------------------------------------------------------


class Predicate:
    """One scalar comparison inside a filter segment.

    Attributes:
        expression: The original library :class:`Expression`, reused for
            evaluation so filter semantics stay identical to ``find()``.
        read_paths: Concrete relative paths read on each candidate, with an
            empty tuple standing for the candidate itself.
        root_read_paths: Concrete paths read from the document root.
    """

    __slots__ = ("expression", "read_paths", "root_read_paths")

    def __init__(
        self,
        expression: Expression,
        read_paths: list[tuple[Segment, ...]],
        root_read_paths: list[tuple[Segment, ...]],
    ) -> None:
        self.expression = expression
        self.read_paths = read_paths
        self.root_read_paths = root_read_paths


class PlanSegment:
    """One locally maintainable step of a compiled expression."""

    __slots__ = ("kind", "keys", "indices", "filter_node", "predicates")

    def __init__(
        self,
        kind: str,
        keys: tuple[str, ...] | None = None,
        indices: tuple[int, ...] | None = None,
        filter_node: Filter | None = None,
        predicates: list[Predicate] | None = None,
    ) -> None:
        self.kind = kind
        self.keys = keys
        self.indices = indices
        self.filter_node = filter_node
        self.predicates = predicates


class Plan:
    """A compiled expression: either locally maintainable or full recompute."""

    __slots__ = ("expression", "segments", "incremental", "references_root")

    def __init__(
        self,
        expression: JSONPath,
        segments: list[PlanSegment],
        incremental: bool,
        references_root: bool,
    ) -> None:
        self.expression = expression
        self.segments = segments
        self.incremental = incremental
        self.references_root = references_root


def _chain_segments(
    node: JSONPath,
) -> tuple[list[tuple[str, tuple]], list[tuple[Segment, ...]], bool] | None:
    """Flatten a ``This``/``Root`` anchored chain into concrete read paths.

    Returns ``(segments, read_paths, relative)`` where each read path is a
    concrete relative (candidate-relative) or absolute (root-relative) path.
    """

    relative: bool | None = None
    parts: list[JSONPath] = []
    current: JSONPath | None = node
    while isinstance(current, Child):
        parts.append(current.right)
        current = current.left
    if isinstance(current, This):
        relative = True
    elif isinstance(current, Root):
        relative = False
    else:
        return None

    resolved: list[tuple[str, tuple]] = []
    for part in reversed(parts):
        if isinstance(part, Fields):
            resolved.append(("key", part.fields))
        elif isinstance(part, Index):
            resolved.append(("index", part.indices))
        elif isinstance(part, Wildcard):
            resolved.append(("wildcard", ()))
        else:
            return None

    paths: list[tuple[Segment, ...]] = [()]
    for kind, values in resolved:
        if kind == "wildcard":
            paths = [path + ("*",) for path in paths]
        else:
            paths = [path + (value,) for path in paths for value in values]
    return resolved, paths, relative


def _compile_predicate(node: Expression) -> Predicate | None:
    """Compile one filter expression, or return ``None`` to force recompute."""

    # Existence tests and scalar comparisons only; the compared operand must
    # be a plain scalar, not another query.
    if node.op not in {None, "!", "==", "=", "!=", "<", "<=", ">", ">=", "=~"}:
        return None
    if isinstance(node.value, JSONPath):
        return None
    if node.op == "=~" and not isinstance(node.value, str):
        return None

    flattened = _chain_segments(node.target)
    if flattened is None:
        return None
    _segments, read_paths, relative = flattened
    if relative:
        return Predicate(node, read_paths, [])
    return Predicate(node, [], read_paths)


def _compile_filter(node: Filter) -> tuple[PlanSegment | None, bool]:
    """Compile a filter node; returns (segment, references_root)."""

    predicates: list[Predicate] = []
    references_root = False
    for expression in node.expressions:
        if isinstance(expression, _BooleanOperation):
            # Conjunctions beyond a plain filter-level AND (e.g. OR) cannot
            # be maintained locally in the first version.
            if isinstance(expression, Or):
                return None, False
            assert isinstance(expression, And)
            for operand in expression.operands:
                if not isinstance(operand, Expression):
                    return None, False
                predicate = _compile_predicate(operand)
                if predicate is None:
                    return None, False
                predicates.append(predicate)
                references_root = references_root or bool(
                    predicate.root_read_paths
                )
            continue
        if not isinstance(expression, Expression):
            return None, False
        predicate = _compile_predicate(expression)
        if predicate is None:
            return None, False
        predicates.append(predicate)
        references_root = references_root or bool(predicate.root_read_paths)
    if not predicates:
        return None, False
    return (
        PlanSegment(
            "filter", filter_node=node, predicates=predicates
        ),
        references_root,
    )


def compile_plan(expression: JSONPath) -> Plan:
    """Compile a parsed expression into a maintenance plan.

    Unsupported AST nodes yield a plan whose ``incremental`` flag is false;
    such subscriptions are re-evaluated wholesale after every commit.
    """

    nodes: list[JSONPath] = []
    current: JSONPath | None = expression
    while isinstance(current, Child):
        nodes.append(current.right)
        current = current.left
    if not isinstance(current, Root):
        return Plan(expression, [], False, False)

    segments: list[PlanSegment] = []
    references_root = False
    for node in reversed(nodes):
        if isinstance(node, Fields):
            if node._deprecated_wildcard_use:
                return Plan(expression, [], False, False)
            segments.append(PlanSegment("key", keys=node.fields))
        elif isinstance(node, Index):
            segments.append(PlanSegment("index", indices=node.indices))
        elif isinstance(node, Wildcard):
            segments.append(PlanSegment("wildcard"))
        elif isinstance(node, Filter):
            segment, root_ref = _compile_filter(node)
            if segment is None:
                return Plan(expression, [], False, False)
            references_root = references_root or root_ref
            segments.append(segment)
        else:
            return Plan(expression, [], False, False)

    return Plan(expression, segments, True, references_root)


# ---------------------------------------------------------------------------
# Tracing navigator
# ---------------------------------------------------------------------------


def _try_resolve(
    document: typing.Any, path: ConcretePath
) -> tuple[bool, typing.Any]:
    value = document
    for segment in path:
        if isinstance(value, list):
            if not isinstance(segment, int):
                return False, None
            index = segment if segment >= 0 else segment + len(value)
            if not -len(value) <= index < len(value):
                return False, None
            value = value[index]
        elif isinstance(value, dict):
            if segment not in value:
                return False, None
            value = value[segment]
        else:
            return False, None
    return True, value


class _Trace:
    """Accumulates the concrete paths one maintenance pass actually touches."""

    __slots__ = (
        "container_paths",
        "input_paths",
        "indexed_inputs",
        "value_paths",
        "predicate_anchors",
        "predicate_candidates",
        "root_paths",
        "visits",
    )

    def __init__(self) -> None:
        self.container_paths: set[ConcretePath] = set()
        # Concrete paths each segment reads as its input container.  These
        # are the only membership dependencies; resolved child values live in
        # value_paths and never double as containers.
        self.input_paths: set[ConcretePath] = set()
        # (segment depth, input container path) for fixed key/index segments.
        # These need re-navigation on structural child changes only.
        self.indexed_inputs: set[tuple[int, ConcretePath]] = set()
        self.value_paths: set[ConcretePath] = set()
        # (filter segment depth, filter input container path) pairs.
        self.predicate_anchors: set[tuple[int, ConcretePath]] = set()
        # (filter segment depth, individual candidate path) pairs, used to
        # re-evaluate a single candidate when only its predicate reads change.
        self.predicate_candidates: set[tuple[int, ConcretePath]] = set()
        self.root_paths: set[ConcretePath] = set()
        self.visits = 0

    def visit(self, path: ConcretePath) -> None:
        self.visits += 1


def _record_predicate_reads(
    trace: _Trace,
    candidate_path: ConcretePath,
    candidate: typing.Any,
    root: typing.Any,
    segment: PlanSegment,
    filter_depth: int,
) -> None:
    assert segment.predicates is not None
    trace.predicate_anchors.add((filter_depth, candidate_path[:-1] or ()))
    trace.predicate_candidates.add((filter_depth, candidate_path))
    for predicate in segment.predicates:
        for relative in predicate.read_paths:
            expanded = _expand_wildcards(candidate, relative)
            for concrete in expanded:
                full_path = candidate_path + concrete
                trace.visits += 1
                _try_resolve(root, full_path)
        for root_path in predicate.root_read_paths:
            expanded = _expand_wildcards(root, root_path)
            for concrete in expanded:
                trace.root_paths.add(concrete)
                trace.visits += 1
                _try_resolve(root, concrete)


def _expand_wildcards(
    value: typing.Any, pattern: tuple[Segment, ...]
) -> list[tuple[Segment, ...]]:
    """Expand ``*`` pattern segments against the current document shape."""

    if not pattern:
        return [()]
    token, rest = pattern[0], pattern[1:]
    if token != "*":
        found, child = _try_resolve(value, (token,))
        if not found:
            return []
        return [(token,) + suffix for suffix in _expand_wildcards(child, rest)]
    if isinstance(value, list):
        keys: typing.Iterable[Segment] = range(len(value))
    elif isinstance(value, dict):
        keys = tuple(value.keys())
    else:
        return []
    results: list[tuple[Segment, ...]] = []
    for key in keys:
        for suffix in _expand_wildcards(value[key], rest):
            results.append((key,) + suffix)
    return results


def _materialize(
    plan: Plan,
    document: typing.Any,
    trace: _Trace,
    entry_paths: typing.Iterable[ConcretePath] | None = None,
    entry_depth: int = 0,
) -> dict[ConcretePath, typing.Any]:
    """Run plan segments from ``entry_depth`` over the given entry paths.

    A full materialization starts from ``[()]`` (the document root); anchor
    reconciliation reuses the same machinery from a deeper segment.
    """

    if entry_paths is None:
        entry_paths = [()]
    fronts: list[tuple[ConcretePath, typing.Any]] = []
    for path in entry_paths:
        found, value = _try_resolve(document, path)
        if found:
            fronts.append((path, value))
    for depth, segment in enumerate(plan.segments[entry_depth:], start=entry_depth):
        next_fronts: list[tuple[ConcretePath, typing.Any]] = []
        for path, value in fronts:
            trace.visit(path)
            # Only wildcard/filter segments iterate their input container,
            # so only their inputs are membership dependencies.  Fixed
            # key/index segments address one node directly and follow array
            # shifts via phase 1 instead; recording their resolved input or
            # output here would go stale after inserts and removals.
            if segment.kind in {"wildcard", "filter"} and isinstance(
                value, (list, dict)
            ):
                trace.input_paths.add(path)
            elif segment.kind in {"key", "index"} and isinstance(
                value, (list, dict)
            ):
                trace.indexed_inputs.add((depth, path))
            if segment.kind == "key":
                if not isinstance(value, dict):
                    continue
                assert segment.keys is not None
                for key in segment.keys:
                    if key in value:
                        child_path = path + (key,)
                        trace.value_paths.add(child_path)
                        trace.visits += 1
                        next_fronts.append((child_path, value[key]))
            elif segment.kind == "index":
                if not isinstance(value, list):
                    continue
                assert segment.indices is not None
                for index in segment.indices:
                    if -len(value) <= index < len(value):
                        normalized = index if index >= 0 else index + len(value)
                        child_path = path + (normalized,)
                        trace.value_paths.add(child_path)
                        trace.visits += 1
                        next_fronts.append((child_path, value[normalized]))
            elif segment.kind == "wildcard":
                if isinstance(value, list):
                    items: typing.Iterable[tuple[Segment, typing.Any]] = tuple(
                        enumerate(value)
                    )
                elif isinstance(value, dict):
                    items = tuple(value.items())
                else:
                    continue
                for key, child in items:
                    child_path = path + (key,)
                    trace.value_paths.add(child_path)
                    trace.visits += 1
                    next_fronts.append((child_path, child))
            else:  # filter
                if isinstance(value, list):
                    items = tuple(enumerate(value))
                elif isinstance(value, dict):
                    items = tuple(value.items())
                else:
                    continue
                matched_keys = _filter_container(
                    segment, value, path, document, trace, depth
                )
                for key, child in items:
                    child_path = path + (key,)
                    trace.visits += 1
                    if key in matched_keys:
                        trace.value_paths.add(child_path)
                        next_fronts.append((child_path, child))
        fronts = next_fronts
    # Deep-copy matched values so later in-place document mutations never
    # silently rewrite the "old" side of a hit-set diff.
    return {path: copy.deepcopy(value) for path, value in fronts}


def _materialize_filter_candidate(
    plan: Plan,
    document: typing.Any,
    trace: _Trace,
    candidate_path: ConcretePath,
    filter_depth: int,
) -> dict[ConcretePath, typing.Any]:
    """Re-evaluate one filter candidate, then run the remaining segments.

    Only the candidate at *candidate_path* is tested against the filter
    segment at *filter_depth*; every other candidate is left untouched.
    """

    segment = plan.segments[filter_depth]
    found, candidate = _try_resolve(document, candidate_path)
    if not found:
        return {}
    container_path = candidate_path[:-1]
    found_container, container = _try_resolve(document, container_path)
    if not found_container:
        return {}
    candidate_key = candidate_path[-1]
    matched = _filter_container(
        segment,
        container,
        container_path,
        document,
        trace,
        filter_depth,
        only_candidate=candidate_key,
    )
    if candidate_path[-1] not in matched:
        return {}
    fronts: list[tuple[ConcretePath, typing.Any]] = [(candidate_path, candidate)]
    for depth, next_segment in enumerate(
        plan.segments[filter_depth + 1:], start=filter_depth + 1
    ):
        next_fronts: list[tuple[ConcretePath, typing.Any]] = []
        for path, value in fronts:
            trace.visit(path)
            if next_segment.kind == "key":
                if isinstance(value, dict):
                    assert next_segment.keys is not None
                    for key in next_segment.keys:
                        if key in value:
                            child_path = path + (key,)
                            trace.value_paths.add(child_path)
                            trace.visits += 1
                            next_fronts.append((child_path, value[key]))
            elif next_segment.kind == "index" and isinstance(value, list):
                assert next_segment.indices is not None
                for index in next_segment.indices:
                    if -len(value) <= index < len(value):
                        normalized = index if index >= 0 else index + len(value)
                        child_path = path + (normalized,)
                        trace.value_paths.add(child_path)
                        trace.visits += 1
                        next_fronts.append((child_path, value[normalized]))
            elif next_segment.kind == "wildcard":
                if isinstance(value, list):
                    items: typing.Iterable[tuple[Segment, typing.Any]] = tuple(
                        enumerate(value)
                    )
                elif isinstance(value, dict):
                    items = tuple(value.items())
                else:
                    items = ()
                for key, child in items:
                    child_path = path + (key,)
                    trace.value_paths.add(child_path)
                    trace.visits += 1
                    next_fronts.append((child_path, child))
            else:
                matched_keys = _filter_container(
                    next_segment, value, path, document, trace, depth
                )
                if isinstance(value, list):
                    items = tuple(enumerate(value))
                elif isinstance(value, dict):
                    items = tuple(value.items())
                else:
                    items = ()
                for key, child in items:
                    child_path = path + (key,)
                    if key in matched_keys:
                        trace.value_paths.add(child_path)
                        next_fronts.append((child_path, child))
        fronts = next_fronts
    return {path: copy.deepcopy(value) for path, value in fronts}


def _filter_container(
    segment: PlanSegment,
    container: typing.Any,
    container_path: ConcretePath,
    root: typing.Any,
    trace: _Trace,
    filter_depth: int,
    only_candidate: Segment | None = None,
) -> set[Segment]:
    """Evaluate a filter segment over one container.

    When *only_candidate* is given, just that one candidate is evaluated and
    its predicate reads are recorded in isolation; otherwise every candidate
    in the container is evaluated.
    """

    assert segment.filter_node is not None
    if isinstance(container, list):
        items: typing.Iterable[tuple[Segment, typing.Any]] = tuple(
            enumerate(container)
        )
    elif isinstance(container, dict):
        items = tuple(container.items())
    else:
        return set()

    # Record the concrete reads each predicate performs on every candidate.
    for key, child in items:
        if only_candidate is not None and key != only_candidate:
            continue
        candidate_path = container_path + (key,)
        _record_predicate_reads(
            trace, candidate_path, child, root, segment, filter_depth
        )

    # Reuse the library's own filter semantics, including ``$`` handling.
    # Candidates are built exactly like ``Filter.find()`` builds them: the
    # candidate's context is the container, and the container's context chain
    # reaches the document root so root references resolve correctly.
    from jsonpath_ng import DatumInContext

    references_root = segment.filter_node._references_root()
    parent = (
        DatumInContext(container, context=_root_context(root))
        if references_root
        else DatumInContext(container)
    )
    matched: set[Segment] = set()
    is_list = isinstance(container, list)
    for key, child in items:
        if only_candidate is not None and key != only_candidate:
            continue
        path_node = (
            Index(key)
            if is_list
            else Fields._compliant_constructor(str(key))
        )
        candidate = DatumInContext(child, path=path_node, context=parent)
        if all(
            expression.find(candidate) for expression in segment.filter_node.expressions
        ):
            normalized = key if not is_list or key >= 0 else key + len(container)
            matched.add(normalized)
    return matched


def _root_context(root: typing.Any) -> typing.Any:
    """Build the minimal context chain whose ``$`` resolves to *root*."""

    from jsonpath_ng import DatumInContext

    return DatumInContext(root)


# ---------------------------------------------------------------------------
# Public change records
# ---------------------------------------------------------------------------


class MatchChange:
    """A change to one entry of a subscription's hit set."""

    __slots__ = ("kind", "path", "old_path", "old_value", "new_value")

    def __init__(
        self,
        kind: str,
        path: ConcretePath,
        old_value: typing.Any = ABSENT,
        new_value: typing.Any = ABSENT,
        old_path: ConcretePath | None = None,
    ) -> None:
        self.kind = kind
        self.path = path
        self.old_path = old_path
        self.old_value = old_value
        self.new_value = new_value

    def __repr__(self) -> str:
        if self.kind == "move":
            return f"MatchChange('move', {self.old_path!r} -> {self.path!r})"
        return f"MatchChange({self.kind!r}, {self.path!r})"

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, MatchChange)
            and self.kind == other.kind
            and self.path == other.path
            and self.old_path == other.old_path
            and self.old_value == other.old_value
            and self.new_value == other.new_value
        )


class Notification:
    """The versioned patch delivered to one subscription after a commit."""

    __slots__ = (
        "subscription_id",
        "base_version",
        "version",
        "changes",
        "recomputed",
        "visits",
    )

    def __init__(
        self,
        subscription_id: int,
        base_version: int,
        version: int,
        changes: list[MatchChange],
        recomputed: bool,
        visits: int,
    ) -> None:
        self.subscription_id = subscription_id
        self.base_version = base_version
        self.version = version
        self.changes = changes
        self.recomputed = recomputed
        self.visits = visits

    @property
    def added(self) -> list[MatchChange]:
        return [change for change in self.changes if change.kind == "add"]

    @property
    def removed(self) -> list[MatchChange]:
        return [change for change in self.changes if change.kind == "remove"]

    @property
    def moved(self) -> list[MatchChange]:
        return [change for change in self.changes if change.kind == "move"]

    @property
    def replaced(self) -> list[MatchChange]:
        return [change for change in self.changes if change.kind == "replace"]

    def __repr__(self) -> str:
        return (
            f"Notification(subscription={self.subscription_id}, "
            f"version={self.version}, changes={len(self.changes)})"
        )


# ---------------------------------------------------------------------------
# Subscriptions and sessions
# ---------------------------------------------------------------------------


def _is_prefix(prefix: ConcretePath, path: ConcretePath) -> bool:
    return path[: len(prefix)] == prefix


def _remap_path(path: ConcretePath, event: OperationEvent) -> ConcretePath | None:
    """Apply an array shift to a concrete path; ``None`` means it was removed."""

    for old, new in zip(event.shifted_from, event.shifted_to):
        if _is_prefix(old, path):
            return new + path[len(old):]
    if event.op == "remove" and _is_prefix(event.path, path):
        return None
    return path


def _event_to_original_coordinates(
    index: int, events: list[OperationEvent]
) -> OperationEvent:
    """Rewrite event *index* into the document coordinates before the patch.

    Later operations use indices shifted by earlier inserts and removals, so
    their shifts are inverted in reverse order.  An event that modifies a
    node inserted earlier in the same patch is tagged with
    :data:`_INSERTED_MARKER`; such a path has no pre-patch anchor and is
    handled separately during anchor collection.
    """

    event = events[index]
    path = event.path
    for later in reversed(events[index + 1:]):
        path = _invert_shift(path, later)

    inserted = False
    for earlier in reversed(events[:index]):
        if earlier.op != "add" or not earlier.shifted_from:
            path = _invert_shift(path, earlier)
            continue
        first_old = earlier.shifted_from[0]
        if (
            len(path) > len(first_old)
            and path[: len(first_old) - 1] == first_old[:-1]
            and path[len(first_old) - 1] == first_old[-1]
        ):
            # This event modifies the element inserted by *earlier*.
            path = first_old[:-1] + (_INSERTED_MARKER,) + path[len(first_old):]
            inserted = True
            break
        path = _invert_shift(path, earlier)

    if not inserted and event.op != "add":
        path = _invert_shift(path, event)

    if path == event.path:
        return event
    return OperationEvent(event.op, path, event.old_value, event.new_value)


def _invert_shift(path: ConcretePath, event: OperationEvent) -> ConcretePath:
    """Reverse one array shift on a path expressed in post-shift indices."""

    for old, new in zip(event.shifted_from, event.shifted_to):
        if _is_prefix(new, path):
            return old + path[len(new):]
    return path


class Subscription:
    """A registered expression and its locally maintained hit set."""

    def __init__(
        self,
        subscription_id: int,
        expression: JSONPath,
        document: VersionedDocument,
    ) -> None:
        self.id = subscription_id
        self.expression = expression
        self.plan = compile_plan(expression)
        self._document = document
        self._closed = False
        self._trace = _Trace()
        if self.plan.incremental:
            self._matches = _materialize(self.plan, document.document, self._trace)
        else:
            # Unsupported expressions are maintained by explicit recompute.
            self._matches = self._full_find(document.document)
        self.total_visits = self._trace.visits

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def incremental(self) -> bool:
        return self.plan.incremental

    @property
    def matches(self) -> dict[ConcretePath, typing.Any]:
        """The current hit set, keyed by concrete materialized path."""

        return dict(self._matches)

    @property
    def values(self) -> list[typing.Any]:
        """Current matched values in path (document) order."""

        return [self._matches[path] for path in sorted(self._matches)]

    @property
    def dependencies(self) -> dict[str, list[ConcretePath]]:
        """The concrete paths the last maintenance pass actually depended on.

        Four dependency kinds are recorded: traversed containers, selected
        values, filter input containers (with the filter's segment depth) and
        concrete reads made against the document root.
        """

        return {
            "containers": sorted(self._trace.container_paths),
            "inputs": sorted(self._trace.input_paths),
            "values": sorted(self._trace.value_paths),
            "predicate_anchors": sorted(
                (depth, path) for depth, path in self._trace.predicate_anchors
            ),
            "predicate_candidates": sorted(
                (depth, path)
                for depth, path in self._trace.predicate_candidates
            ),
            "root_reads": sorted(self._trace.root_paths),
        }

    def _full_find(self, document: typing.Any) -> dict[ConcretePath, typing.Any]:
        matches: dict[ConcretePath, typing.Any] = {}
        for datum in self.expression.find(document):
            matches[_datum_path(datum)] = copy.deepcopy(datum.value)
        return matches

    def close(self) -> None:
        self._closed = True
        self._document = None  # type: ignore[assignment]
        self._matches = {}
        self._trace = None  # type: ignore[assignment]

    # -- incremental propagation -----------------------------------------

    def propagate(
        self,
        events: list[OperationEvent],
        base_version: int,
        version: int,
    ) -> Notification:
        if self._closed:
            msg = f"Subscription {self.id} is closed"
            raise SubscriptionClosedError(msg)
        trace = _Trace()
        assert self._document is not None
        document = self._document.document

        if not self.plan.incremental or any(not event.path for event in events):
            changes, recomputed = self._recompute(document, events)
            self._record_trace(trace)
            self.total_visits += trace.visits
            return Notification(
                self.id, base_version, version, changes, recomputed, trace.visits
            )

        changes = self._propagate_incremental(events, document, trace)
        self.total_visits += trace.visits
        return Notification(
            self.id, base_version, version, changes, False, trace.visits
        )

    def _record_trace(self, trace: _Trace) -> None:
        # A full recompute re-materializes from the root to refresh deps.
        refreshed = _Trace()
        if self.plan.incremental:
            _materialize(self.plan, self._document.document, refreshed)
        self._trace = refreshed
        trace.container_paths.update(refreshed.container_paths)
        trace.input_paths.update(refreshed.input_paths)
        trace.indexed_inputs.update(refreshed.indexed_inputs)
        trace.value_paths.update(refreshed.value_paths)
        trace.predicate_anchors.update(refreshed.predicate_anchors)
        trace.predicate_candidates.update(refreshed.predicate_candidates)
        trace.root_paths.update(refreshed.root_paths)
        trace.visits += refreshed.visits

    def _recompute(
        self, document: typing.Any, events: list[OperationEvent]
    ) -> tuple[list[MatchChange], bool]:
        old_matches = self._matches
        if self.plan.incremental:
            trace = _Trace()
            new_matches = _materialize(self.plan, document, trace)
        else:
            new_matches = self._full_find(document)
        self._matches = new_matches
        return self._diff(old_matches, new_matches, {}), True

    def _propagate_incremental(
        self,
        events: list[OperationEvent],
        document: typing.Any,
        trace: _Trace,
    ) -> list[MatchChange]:
        # Phase 1: apply array shifts to the maintained match map, keeping the
        # cumulative origin of every shifted match for move reporting.
        snapshot = dict(self._matches)
        origins: dict[ConcretePath, ConcretePath] = {
            path: path for path in self._matches
        }
        for event in events:
            remapped: dict[ConcretePath, typing.Any] = {}
            remapped_origins: dict[ConcretePath, ConcretePath] = {}
            for path, value in self._matches.items():
                updated = _remap_path(path, event)
                if updated is None:
                    continue
                remapped[updated] = value
                remapped_origins[updated] = origins.get(path, path)
            self._matches = remapped
            origins = remapped_origins
            # Keep the recorded dependency paths consistent with the shifts,
            # dropping references to removed nodes so stale indices (such as
            # a negative index's old target) cannot anchor later changes.
            self._remap_trace(event)

        # Phase 2: reconcile affected subtrees at their segment anchors.
        # Event paths use the indices in force when each operation ran, while
        # maintained dependencies still use pre-patch indices.  Rewrite every
        # event back through the shifts introduced by later operations so all
        # anchors are computed against the original document coordinates.
        pre_patch_events = [
            _event_to_original_coordinates(index, events)
            for index, event in enumerate(events)
        ]
        anchors = self._collect_anchors(pre_patch_events)
        # Run deeper anchors first: once a subtree is re-navigated from a
        # deeper segment, any shallower anchor fully inside it is redundant.
        handled: list[tuple[int, ConcretePath]] = []
        for depth, anchor_path in sorted(
            anchors, key=lambda item: item[0], reverse=True
        ):
            if anchor_path is None:
                old_matches = dict(self._matches)
                self._matches = _materialize(self.plan, document, trace)
                # The fresh trace is authoritative for future change
                # propagation, so replace rather than merely absorb it.
                self._trace = trace
                return self._diff(snapshot, self._matches, origins)
            if any(
                handled_depth <= depth and _is_prefix(handled_path, anchor_path)
                for handled_depth, handled_path in handled
            ):
                continue
            handled.append((depth, anchor_path))
            is_candidate_anchor = (
                (depth, anchor_path) in self._trace.predicate_candidates
            )
            # Drop stale dependency records inside this subtree before the
            # re-navigation repopulates them, so removals never leave
            # references to containers that no longer exist.
            self._prune_trace(anchor_path)
            for path in [
                path for path in self._matches if _is_prefix(anchor_path, path)
            ]:
                del self._matches[path]
            found, _anchor_value = _try_resolve(document, anchor_path)
            if found:
                if is_candidate_anchor:
                    fresh = _materialize_filter_candidate(
                        self.plan, document, trace, anchor_path, depth
                    )
                else:
                    fresh = _materialize(
                        self.plan, document, trace, [anchor_path], depth
                    )
                self._matches.update(fresh)
            self._absorb_trace(trace)

        self._apply_deep_leaf_replacements(pre_patch_events, document)
        self._correct_fixed_indices(document, trace)
        changes = self._diff(snapshot, self._matches, origins)
        # Dependencies are maintained incrementally: each anchor re-run above
        # recorded exactly the containers and reads it touched via *trace*,
        # and _absorb_trace() merged them into the persistent dependency set.
        # No full re-materialization is needed, which keeps unrelated
        # branches untouched by a localized change.
        return changes

    def _correct_fixed_indices(
        self, document: typing.Any, trace: _Trace
    ) -> None:
        """Re-authorize matches produced by fixed index segments.

        A fixed index (including a negative one) addresses one array slot,
        which after inserts and removals is cheap to re-resolve directly.
        Recomputing just those segments is O(number of index segments) and
        guarantees a single, current match instead of a stale shifted entry.
        """

        if not any(segment.kind == "index" for segment in self.plan.segments):
            return
        correction_trace = _Trace()
        fresh = _materialize(self.plan, document, correction_trace)
        trace.visits += correction_trace.visits
        trace.container_paths.update(correction_trace.container_paths)
        trace.input_paths.update(correction_trace.input_paths)
        trace.value_paths.update(correction_trace.value_paths)
        # Replace every match that flows through an index segment with the
        # freshly resolved set; non-index matches are left untouched.
        corrected = {
            path: value
            for path, value in self._matches.items()
            if not self._path_uses_index_segment(path)
        }
        corrected.update(fresh)
        self._matches = corrected

    def _path_uses_index_segment(self, path: ConcretePath) -> bool:
        for depth, segment in enumerate(self.plan.segments):
            if segment.kind == "index" and depth < len(path):
                return True
        return False

    def _apply_deep_leaf_replacements(
        self,
        events: list[OperationEvent],
        document: typing.Any,
    ) -> None:
        """Refresh matched values whose interior was replaced in place.

        When an expression matches a whole subtree (for example a wildcard
        over objects), a leaf change below the final selected node produces
        no structural anchor.  The maintained copy of the matched value is
        refreshed here so the hit-set diff reports a value replacement.
        """

        n_segments = len(self.plan.segments)
        for event in events:
            if event.op != "replace" or len(event.path) <= n_segments:
                continue
            match_path = event.path[:n_segments]
            if match_path in self._matches:
                found, current = _try_resolve(document, match_path)
                if found:
                    self._matches[match_path] = copy.deepcopy(current)

    def _absorb_trace(self, trace: _Trace) -> None:
        self._trace.container_paths.update(trace.container_paths)
        self._trace.input_paths.update(trace.input_paths)
        self._trace.indexed_inputs.update(trace.indexed_inputs)
        self._trace.value_paths.update(trace.value_paths)
        self._trace.predicate_anchors.update(trace.predicate_anchors)
        self._trace.predicate_candidates.update(trace.predicate_candidates)
        self._trace.root_paths.update(trace.root_paths)

    def _prune_trace(self, prefix: ConcretePath) -> None:
        """Remove dependency records that live below *prefix*."""

        self._trace.container_paths = {
            path
            for path in self._trace.container_paths
            if not _is_prefix(prefix, path)
        }
        self._trace.input_paths = {
            path
            for path in self._trace.input_paths
            if not _is_prefix(prefix, path)
        }
        self._trace.indexed_inputs = {
            (depth, path)
            for depth, path in self._trace.indexed_inputs
            if not _is_prefix(prefix, path)
        }
        self._trace.value_paths = {
            path for path in self._trace.value_paths if not _is_prefix(prefix, path)
        }
        self._trace.predicate_anchors = {
            (depth, path)
            for depth, path in self._trace.predicate_anchors
            if not _is_prefix(prefix, path)
        }
        self._trace.predicate_candidates = {
            (depth, path)
            for depth, path in self._trace.predicate_candidates
            if not _is_prefix(prefix, path)
        }
        self._trace.root_paths = {
            path for path in self._trace.root_paths if not _is_prefix(prefix, path)
        }

    def _remap_trace(self, event: OperationEvent) -> None:
        """Apply one array shift/removal to every recorded dependency path."""

        def remap(path: ConcretePath) -> ConcretePath | None:
            return _remap_path(path, event)

        def remap_set(paths: set[ConcretePath]) -> set[ConcretePath]:
            result: set[ConcretePath] = set()
            for path in paths:
                updated = remap(path)
                if updated is not None:
                    result.add(updated)
            return result

        self._trace.container_paths = remap_set(self._trace.container_paths)
        self._trace.input_paths = remap_set(self._trace.input_paths)
        self._trace.value_paths = remap_set(self._trace.value_paths)
        self._trace.root_paths = remap_set(self._trace.root_paths)
        indexed: set[tuple[int, ConcretePath]] = set()
        for depth, path in self._trace.indexed_inputs:
            updated = remap(path)
            if updated is not None:
                indexed.add((depth, updated))
        self._trace.indexed_inputs = indexed
        anchors: set[tuple[int, ConcretePath]] = set()
        for depth, path in self._trace.predicate_anchors:
            updated = remap(path)
            if updated is not None:
                anchors.add((depth, updated))
        self._trace.predicate_anchors = anchors
        candidates: set[tuple[int, ConcretePath]] = set()
        for depth, path in self._trace.predicate_candidates:
            updated = remap(path)
            if updated is not None:
                candidates.add((depth, updated))
        self._trace.predicate_candidates = candidates

    def _collect_anchors(
        self, events: list[OperationEvent]
    ) -> set[tuple[int, ConcretePath | None]]:
        """Map each change to the segment input containers it invalidates.

        Segment *d* reads an input container at concrete depth *d*.  A change
        invalidates segment *d* for container *C* when it can alter what
        segment *d* selects from *C*:

        * a direct child of *C* is added, removed or replaced (the child
          membership or value the segment looks at changed);
        * *C* itself is replaced;
        * for a filter segment, a candidate field the predicate reads under
          *C* changed;
        * a root-referencing predicate read changed, which invalidates every
          filter container and therefore forces a full recompute.

        Changes strictly inside an already selected subtree below the final
        segment need no anchor: matched values are deep-copied, so the hit-set
        diff reports them directly.
        """

        anchors: set[tuple[int, ConcretePath | None]] = set()
        n_segments = len(self.plan.segments)
        for event in events:
            path = event.path
            if not path:
                anchors.add((0, None))
                continue

            root_affected = any(
                segment.kind == "filter"
                and any(
                    _pattern_hit(pattern, path)
                    for predicate in (segment.predicates or [])
                    for pattern in predicate.root_read_paths
                )
                for segment in self.plan.segments
            )
            if root_affected:
                anchors.add((0, None))
                continue

            if _INSERTED_MARKER in path:
                # A modification inside a node inserted by this same patch.
                # The insert's own structural anchor covers the container, so
                # only re-run a filter whose input is that container.
                marker_at = path.index(_INSERTED_MARKER)
                container_path = path[:marker_at]
                for depth, recorded in self._trace.predicate_anchors:
                    if recorded == container_path:
                        anchors.add((depth, container_path))
                continue

            # Structural / value changes against recorded input containers.
            for container_path in self._trace.input_paths:
                depth = len(container_path)
                if depth >= n_segments:
                    continue
                if not _is_prefix(container_path, path):
                    if _is_prefix(path, container_path):
                        # The recorded container (or an ancestor of it) was
                        # itself replaced/removed.
                        parent_path = path[:-1]
                        anchor_depth = min(len(parent_path), depth)
                        anchors.add((anchor_depth, parent_path[:anchor_depth]))
                    continue
                if len(path) <= len(container_path):
                    # The input container itself was replaced.
                    if event.op == "replace":
                        anchors.add((depth, container_path))
                    continue
                # The change is below the input container.  Only a direct
                # child change can alter what this segment selects; deeper
                # changes either feed a deeper segment/filter or sit inside
                # an already matched value.
                if len(path) == len(container_path) + 1:
                    anchors.add((depth, container_path))

            # Fixed key/index segments follow array shifts in phase 1, but a
            # structural add/remove of a direct child, or replacement of the
            # addressed child itself, changes what the fixed address resolves
            # to, so re-run that segment.  Leaf replacements strictly below
            # the addressed node are handled by the deep-copy diff.
            for depth, container_path in self._trace.indexed_inputs:
                if _is_prefix(container_path, path) and len(
                    path
                ) == len(container_path) + 1:
                    anchors.add((depth, container_path))

            # Filter candidate reads.  When the change stays inside one
            # recorded candidate, re-evaluate just that candidate; otherwise
            # fall back to re-running the whole filter input container.
            candidate_hit = False
            for depth, candidate_path in self._trace.predicate_candidates:
                if _is_prefix(candidate_path, path):
                    anchors.add((depth, candidate_path))
                    candidate_hit = True
            if not candidate_hit:
                for depth, container_path in self._trace.predicate_anchors:
                    if _is_prefix(container_path, path) and len(path) > len(
                        container_path
                    ):
                        anchors.add((depth, container_path))
            # A candidate-level filter anchor already re-evaluates that
            # candidate and every segment after the filter, so suppress
            # deeper anchors whose subtree the candidate fully contains.
            candidate_anchor_paths = {
                candidate_path
                for depth, candidate_path in self._trace.predicate_candidates
                if _is_prefix(candidate_path, path)
            }
            if candidate_anchor_paths:
                anchors = {
                    (depth, anchor_path)
                    for depth, anchor_path in anchors
                    if anchor_path is None
                    or not any(
                        depth > candidate_depth
                        and _is_prefix(candidate_path, anchor_path)
                        for candidate_path in candidate_anchor_paths
                        for candidate_depth in (
                            d
                            for d, p in self._trace.predicate_candidates
                            if p == candidate_path
                        )
                    )
                }
        return anchors


    def _diff(
        self,
        old_matches: dict[ConcretePath, typing.Any],
        new_matches: dict[ConcretePath, typing.Any],
        origins: dict[ConcretePath, ConcretePath],
    ) -> list[MatchChange]:
        changes: list[MatchChange] = []
        old_paths = set(old_matches)
        new_paths = set(new_matches)

        # Moves first: a new entry whose pre-shift origin was an old entry
        # with an equal value is a move, not remove + add.
        move_targets: set[ConcretePath] = set()
        move_origins: set[ConcretePath] = set()
        for new_path, origin in origins.items():
            if (
                new_path in new_paths
                and new_path != origin
                and origin in old_paths
                and new_path not in old_paths
                and old_matches[origin] == new_matches[new_path]
            ):
                changes.append(
                    MatchChange(
                        "move",
                        new_path,
                        old_value=old_matches[origin],
                        new_value=new_matches[new_path],
                        old_path=origin,
                    )
                )
                move_targets.add(new_path)
                move_origins.add(origin)

        for path in sorted(new_paths - old_paths - move_targets):
            changes.append(
                MatchChange("add", path, new_value=new_matches[path])
            )
        for path in sorted(old_paths - new_paths - move_origins):
            changes.append(
                MatchChange("remove", path, old_value=old_matches[path])
            )
        for path in sorted(old_paths & new_paths):
            if old_matches[path] != new_matches[path]:
                changes.append(
                    MatchChange(
                        "replace",
                        path,
                        old_value=old_matches[path],
                        new_value=new_matches[path],
                    )
                )
        return changes


def _pattern_hit(pattern: tuple[Segment, ...], path: ConcretePath) -> bool:
    """Whether a concrete change path matches a wildcard dependency pattern."""

    if len(pattern) != len(path):
        return False
    for token, concrete in zip(pattern, path):
        if token != "*" and token != concrete:
            return False
    return True


def _datum_path(datum: typing.Any) -> ConcretePath:
    """Reconstruct a concrete materialized path from a ``DatumInContext``."""

    segments: list[Segment] = []
    current = datum
    while current is not None and not isinstance(current.path, (Root, This)):
        path = current.path
        if isinstance(path, Fields):
            segments.append(path.fields[0])
        elif isinstance(path, Index):
            segments.append(path.indices[0])
        else:
            msg = f"Unsupported datum path segment {path!r}"
            raise SubscriptionLimitError(msg)
        current = current.context
    segments.reverse()
    return tuple(segments)


class SubscriptionSession:
    """A bounded registry of subscriptions over one versioned document."""

    def __init__(
        self,
        document: typing.Any,
        max_subscriptions: int = 64,
    ) -> None:
        self._store = VersionedDocument(document)
        self._max = max_subscriptions
        self._subscriptions: dict[int, Subscription] = {}
        self._next_id = 0
        self._closed = False

    @property
    def document(self) -> typing.Any:
        return self._store.document

    @property
    def version(self) -> int:
        return self._store.version

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    def subscribe(self, expression: JSONPath | str) -> Subscription:
        if self._closed:
            msg = "Cannot subscribe on a closed session"
            raise SubscriptionClosedError(msg)
        if len(self._subscriptions) >= self._max:
            msg = (
                f"Session subscription limit of {self._max} reached; "
                "close a subscription before registering another"
            )
            raise SubscriptionLimitError(msg)
        if isinstance(expression, str):
            from jsonpath_ng.ext.parser import parse

            expression = parse(expression)
        subscription = Subscription(self._next_id, expression, self._store)
        self._subscriptions[self._next_id] = subscription
        self._next_id += 1
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        if subscription.id in self._subscriptions:
            del self._subscriptions[subscription.id]
            subscription.close()

    def commit(
        self,
        patch: typing.Sequence[dict[str, typing.Any]],
        base_version: int | None = None,
    ) -> dict[int, Notification]:
        """Apply a patch atomically and notify every live subscription.

        On failure (bad base version or illegal path) the document and every
        subscription keep their previous state, and the underlying
        :class:`PatchError` propagates unchanged.
        """

        if self._closed:
            msg = "Cannot commit on a closed session"
            raise SubscriptionClosedError(msg)
        old_version = self._store.version
        events = self._store.apply_patch(patch, base_version)
        notifications: dict[int, Notification] = {}
        for subscription in self._subscriptions.values():
            notifications[subscription.id] = subscription.propagate(
                events, old_version, self._store.version
            )
        return notifications

    def close(self) -> None:
        """Release the document and every subscription reference."""

        for subscription in self._subscriptions.values():
            subscription.close()
        self._subscriptions.clear()
        self._store = None  # type: ignore[assignment]
        self._closed = True

    def __enter__(self) -> SubscriptionSession:
        return self

    def __exit__(self, *exc: typing.Any) -> None:
        self.close()
