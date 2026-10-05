"""Incremental matching engine.

The engine compares two *localized* views of the matches:

* old candidates - every pre-batch match lying in a region the batch touched,
  evaluated by reconstructing that region as it was before the batch;
* new candidates - the same regions evaluated against the current document.

Regions ("anchors") are derived by walking the analyzed spine against the
patch event paths. Subtrees untouched by the batch are never evaluated, so a
patch against an unrelated branch costs nothing beyond resolving the anchor.

List insert/remove positions are replayed with symbolic labels so that
elements which merely shifted are distinguished from elements whose value
changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from jsonpath_ng.jsonpath import (
    Child,
    DatumInContext,
    Descendants,
    Fields,
    Index,
    Root,
    Slice,
    This,
)
from jsonpath_ng.ext.filter import Expression, Filter

from .analysis import Step
from .pointer import Pointer, is_prefix, resolve

# ---------------------------------------------------------------------------
# Structural simulation
# ---------------------------------------------------------------------------


def simulate_list(events: List[tuple], final_len: int):
    """Replay one list's structural events.

    Returns ``(fwd, back)`` where *fwd* maps a pre-batch index to its final
    index (survivors only) and *back* maps a final index to its pre-batch
    index (surviving originals only). Inserted/moved elements get symbolic
    labels in *back*: ``("move", id)`` or ``("new", n)``.
    """
    start_len = final_len - sum(1 for e in events if e[0] == "insert")
    start_len += sum(1 for e in events if e[0] == "remove")
    seq: List[Any] = list(range(start_len))
    new_counter = 0
    for event in events:
        if event[0] == "insert":
            _, _, index, _value, move_id = event
            if move_id is not None:
                seq.insert(index, ("move", move_id))
            else:
                seq.insert(index, ("new", new_counter))
                new_counter += 1
        else:
            _, _, index, _old, _move_id = event
            seq.pop(index)
    fwd: Dict[int, int] = {}
    back: Dict[int, Any] = {}
    for final_index, label in enumerate(seq):
        back[final_index] = label
        if isinstance(label, int):
            fwd[label] = final_index
    return fwd, back


def structural_state(events: List[tuple], doc: Any):
    """Build per-final-list simulations and detect nested structural edits."""
    per_list: Dict[Pointer, List[tuple]] = {}
    for event in events:
        if event[0] in ("insert", "remove"):
            per_list.setdefault(event[1], []).append(event)

    simulations: Dict[Pointer, dict] = {}
    for list_ptr, evs in per_list.items():
        final_list = resolve(doc, list_ptr)
        fwd, back = simulate_list(evs, len(final_list))
        simulations[list_ptr] = {"fwd": fwd, "back": back}

    nested = any(
        left != right and (is_prefix(left, right) or is_prefix(right, left))
        for left in simulations
        for right in simulations
    )
    return simulations, nested



def reconstruct_old_anchor(anchor: Pointer, events: List[tuple], doc: Any):
    """Rebuild the pre-batch node at *anchor*.

    The final anchor subtree is copied and all batch events are undone in
    reverse execution order. For each touched list a *length* is maintained
    for the state currently represented by the copy; undo operations and the
    (execution-time) indices of later value events are interpreted against it.
    Same-list moves are undone as relocations; cross-list moves as transfers.
    """
    # A replace of the anchor itself: its pre-batch value is stored directly.
    for event in events:
        if event[0] == "value" and event[1] == anchor:
            return deepcopy_value(event[2])

    node = deepcopy_value(resolve(doc, anchor, None))

    move_pairs = move_index_registry(events)
    same_list = {
        mid for mid, (src, dst) in move_pairs.items() if src[0] == dst[0]
    }
    cross_list = set(move_pairs) - same_list

    # Current length of each touched list in the state represented by `node`.
    lengths: Dict[Pointer, int] = {}
    for event in events:
        if event[0] in ("insert", "remove") and is_prefix(anchor, event[1]):
            final_list = resolve(doc, event[1], None)
            if isinstance(final_list, list):
                lengths[event[1]] = len(final_list)

    def rel_list(list_ptr):
        return _ensure_list(node, list_ptr[len(anchor):])

    def map_value_path(path: Pointer) -> Optional[Pointer]:
        """Map a value event's execution-time path onto the current view."""
        tokens = list(path)
        for list_ptr in sorted(lengths, key=len, reverse=True):
            depth = len(list_ptr)
            if tuple(tokens[:depth]) != list_ptr:
                continue
            if depth < len(tokens) and isinstance(tokens[depth], int):
                idx = tokens[depth]
                # Later structural ops on this list have already been undone
                # (they appear earlier in the reversed traversal), so `lengths`
                # now reflects the state at this event's execution time. The
                # execution-time index is directly valid.
                if idx >= lengths.get(list_ptr, idx + 1):
                    return None
        return tuple(tokens)

    for event in reversed(events):
        kind = event[0]

        if kind == "insert":
            _, list_ptr, index, value, move_id = event
            if not is_prefix(anchor, list_ptr):
                continue
            target = rel_list(list_ptr)
            if move_id in same_list or move_id in cross_list:
                # Destination of a move: element came from elsewhere.
                if index < len(target):
                    target.pop(index)
            else:
                if index < len(target):
                    target.pop(index)
            lengths[list_ptr] = lengths.get(list_ptr, len(target)) - 1 if (
                list_ptr in lengths
            ) else len(target)
            continue

        if kind == "remove":
            _, list_ptr, index, old, move_id = event
            if not is_prefix(anchor, list_ptr):
                continue
            target = rel_list(list_ptr)
            if move_id in same_list:
                # Restore to original source index.
                src_index = move_pairs[move_id][0][1]
                if src_index <= len(target):
                    target.insert(src_index, deepcopy_value(old))
            else:
                # Real deletion or cross-list move source: restore element.
                if index <= len(target):
                    target.insert(index, deepcopy_value(old))
            lengths[list_ptr] = lengths.get(list_ptr, len(target)) + 1 if (
                list_ptr in lengths
            ) else len(target)
            continue

        # value event
        path = event[1]
        if not is_prefix(anchor, path) or path == anchor:
            continue
        mapped = map_value_path(path)
        if mapped is None:
            continue
        rel = mapped[len(anchor):]
        try:
            _set_path(node, rel, deepcopy_value(event[2]))
        except (KeyError, IndexError, TypeError):
            # The element is absent from the state currently reconstructed
            # (it is restored by a structural undo appearing later in the
            # reversed traversal); its pre-batch value is carried by that
            # undo's snapshot instead.
            continue

    return node


def old_anchor_for(
    anchor: Pointer,
    events: List[tuple],
    doc: Any,
    simulations: Dict[Pointer, dict],
    moves: dict,
) -> Tuple[Optional[Pointer], Any]:
    """Resolve a *final* probe anchor back to its pre-batch location."""
    old_anchor = _back_adjust(anchor, simulations, moves)
    if old_anchor is None:
        return None, None
    old_node = reconstruct_old_anchor(old_anchor, events, doc)
    return old_anchor, old_node


def _back_adjust(
    anchor: Pointer,
    simulations: Dict[Pointer, dict],
    moves: dict,
) -> Optional[Pointer]:
    """Map a final anchor back to its pre-batch coordinates."""
    tokens = list(anchor)
    # Cross-list moves: the destination element did not exist there pre-batch.
    for move_id, ((src_list, src_index), (dst_list, dst_index)) in moves.items():
        depth = len(dst_list)
        if (
            tuple(tokens[:depth]) == dst_list
            and len(tokens) > depth
            and tokens[depth] == dst_index
            and src_list != dst_list
        ):
            tokens = list(src_list) + [src_index] + tokens[depth + 1:]
    # In-list back maps, outermost list first.
    for list_ptr in sorted(simulations, key=len):
        depth = len(list_ptr)
        if tuple(tokens[:depth]) != list_ptr:
            continue
        if depth >= len(tokens) or not isinstance(tokens[depth], int):
            continue
        label = simulations[list_ptr]["back"].get(tokens[depth])
        if isinstance(label, int):
            tokens[depth] = label
        elif isinstance(label, tuple) and label[0] == "move":
            source = moves.get(label[1])
            if source is None:
                return None
            (src_list, src_index), (dst_list, dst_index) = source
            if src_list == list_ptr:
                tokens[depth] = src_index
            else:
                tokens = (
                    list(src_list) + [src_index] + tokens[depth + 1:]
                )
        else:
            return None
    return tuple(tokens)

def deepcopy_value(value: Any):
    from copy import deepcopy

    return deepcopy(value)


def _set_path(node: Any, rel: Pointer, value: Any):
    container = node
    for token in rel[:-1]:
        container = container[token]
    token = rel[-1]
    if value is None and not (
        isinstance(container, dict) and token in container and container[token] is None
    ):
        # Removed member.
        if isinstance(container, dict):
            container.pop(token, None)
            return
    if (
        isinstance(container, list)
        and isinstance(token, int)
        and 0 <= token <= len(container)
        and token == len(container)
    ):
        container.append(value)
    else:
        container[token] = value


def _ensure_list(node: Any, rel: Pointer):
    current = node
    for token in rel:
        current = current[token]
    return current


# ---------------------------------------------------------------------------
# Suffix AST construction and localized evaluation
# ---------------------------------------------------------------------------


def step_node(step: Step):
    kind = step.kind
    if kind == "field":
        _mode, names = step.payload
        return Fields(*sorted(names))
    if kind == "index":
        return Index(*step.payload)
    if kind == "slice":
        start, end, st = step.payload
        return Slice(start, end, st)
    if kind == "filter":
        return Filter(
            [Expression(Child(This(), Fields(field)), op, value)
             for field, op, value in step.payload]
        )
    raise ValueError(f"cannot build node for {kind}")


def suffix_ast(steps: List[Step]):
    node: Any = This()
    for step in steps:
        selector = step.payload if step.kind != "desc" else step.payload
        if step.kind == "desc":
            node = Descendants(node, step_node(step.payload))
        else:
            node = Child(node, step_node(step))
    return node




def _build_anchor(anchor: Pointer, node_value: Any) -> DatumInContext:
    # bottom-up: start a root-like datum, then attach tokens.
    # root context:
    root_ctx = DatumInContext(None, path=Root(), context=None)
    # leaf datum whose context chain is root + each preceding token.
    context: Any = None
    for depth, token in enumerate(anchor):
        path_node = Index(token) if isinstance(token, int) else Fields(token)
        if depth == len(anchor) - 1:
            return DatumInContext(node_value, path=path_node, context=context)
        context = DatumInContext(None, path=path_node, context=context)
    raise AssertionError("unreachable")


def datum_pointer(datum: DatumInContext) -> Pointer:
    tokens: List[Any] = []
    _collect(datum.full_path, tokens)
    return tuple(tokens)


def _collect(node: Any, tokens: List[Any]):
    if isinstance(node, (Root, This)):
        return
    if isinstance(node, (Child, Descendants)):
        _collect(node.left, tokens)
        _collect(node.right, tokens)
        return
    if isinstance(node, Fields):
        tokens.append(node.fields[0])
    elif isinstance(node, Index):
        tokens.append(node.indices[0])


def evaluate_probe(
    anchor: Pointer,
    suffix: List[Step],
    node_value: Any,
    counter,
) -> Dict[Pointer, Any]:
    """Evaluate *suffix* at an anchor; return ``{pointer: value}``."""
    if not suffix:
        if counter is not None:
            counter.wrap(node_value)
        return {anchor: node_value}
    wrapped = counter.wrap(node_value) if counter is not None else node_value
    datum = _build_anchor(anchor, wrapped)
    ast = suffix_ast(suffix)
    results: Dict[Pointer, Any] = {}
    for found in ast.find(datum):
        ptr = datum_pointer(found)
        value = counter.unwrap(found.value) if counter is not None else found.value
        results[ptr] = value
    return results


# ---------------------------------------------------------------------------
# Probe planning
# ---------------------------------------------------------------------------


# Event view: tuple (path, is_structural, struct_kind)
# struct_kind: None for value events; 'insert' / 'remove' for list changes.



def normalize_event_paths(
    events: List[tuple]
) -> List[Tuple[Pointer, str]]:
    """Map every event path to final-batch coordinates.

    Each value event tracks its element through *later* structural operations
    (inserts/removes/moves). An element genuinely deleted loses its descriptor;
    a moved element is followed to its destination.
    """
    normalized: List[Tuple[Pointer, str]] = []
    for idx, event in enumerate(events):
        kind = event[0]
        if kind == "value":
            final_path = _track_element(events, idx)
            _, _path, old, new_value = event
            if final_path is None:
                continue
            if old is None and new_value is not None:
                change = "value-add"
            elif new_value is None and old is not None:
                change = "value-del"
            else:
                change = "value-set"
            normalized.append((final_path, change))
        elif kind == "insert":
            normalized.append((event[1], "insert"))
        elif kind == "remove":
            normalized.append((event[1], "remove"))
    return normalized


def _track_element(events: List[tuple], value_idx: int) -> Optional[Pointer]:
    """Follow a value event's element to its final location."""
    path = events[value_idx][1]
    tokens = list(path)
    in_transit = False
    for later in events[value_idx + 1:]:
        if later[0] not in ("insert", "remove"):
            continue
        list_ptr = later[1]
        depth = len(list_ptr)
        if not in_transit and tuple(tokens[:depth]) != list_ptr:
            continue
        if depth >= len(tokens) or (
            not in_transit and not isinstance(tokens[depth], int)
        ):
            continue
        index = tokens[depth] if isinstance(tokens[depth], int) else None
        if later[0] == "insert":
            dst_index = later[2]
            if in_transit:
                # This insert is the move that carries our element.
                tokens = list(list_ptr) + [dst_index] + tokens[depth + 1:]
                in_transit = False
            elif dst_index <= index:
                tokens[depth] = index + 1
        else:  # remove
            src_index = later[2]
            if later[4] is not None:
                # Start of a move: our element leaves this list.
                if not in_transit and src_index == index:
                    in_transit = True
                    continue
            if in_transit:
                continue
            if src_index == index:
                return None
            if src_index < index:
                tokens[depth] = index - 1
    if in_transit:
        return None
    return tuple(tokens)


def _selector_accepts(step: Step, token: Any) -> bool:
    kind = step.kind
    if kind == "field":
        if not isinstance(token, str):
            return False
        mode, names = step.payload
        return mode == "*" or token in names
    if kind == "index":
        return isinstance(token, int) and token in step.payload
    if kind == "slice":
        return isinstance(token, int)
    if kind == "filter":
        return True
    return False




def precise_states(steps: List[Step], path: Pointer) -> Set[Tuple[int, int]]:
    """Like :func:`reachable_states` but with concrete token selectivity.

    A non-full slice only counts as consumed when the concrete index can be in
    the window; otherwise the transition is dropped so no probe is planned.
    """
    states = {(0, 0)}
    progressed = True
    while progressed:
        progressed = False
        for k, d in list(states):
            if d >= len(path) or k > len(steps):
                continue
            token = path[d]
            step = steps[k] if k < len(steps) else None
            if step is None:
                continue
            if step.kind == "desc":
                gap = (k, d + 1)
                if gap not in states:
                    states.add(gap)
                    progressed = True
                if _precisely_accepts(step.payload, token, path, d):
                    hit = (k + 1, d + 1)
                    if hit not in states:
                        states.add(hit)
                        progressed = True
            elif _precisely_accepts(step, token, path, d):
                nxt = (k + 1, d + 1)
                if nxt not in states:
                    states.add(nxt)
                    progressed = True
    return states


def _precisely_accepts(step: Step, token: Any, path: Pointer, d: int) -> bool:
    kind = step.kind
    if kind == "desc":
        return _precisely_accepts(step.payload, token, path, d)
    if kind == "filter" or kind == "field":
        return _selector_accepts(step, token)
    if kind == "index":
        return _selector_accepts(step, token)
    if kind == "slice":
        start, end, st = step.payload
        if not isinstance(token, int):
            return False
        if st in (None, 1) and start is None and end is None:
            return True
        # The window depends on list length; a positive concrete index is in a
        # [0:n] / [:n] style window iff it satisfies the explicit bounds.
        if (st in (None, 1)) and end is not None and start in (None, 0):
            return 0 <= token < end
        if st in (None, 1) and start is not None and end is None:
            return token >= start
        if st not in (None, 1):
            # Precise only for the canonical positive-step case starting at 0;
            # other stepped windows are conservatively handled via windowing.
            if (start in (None, 0) and end is None and st > 0):
                return token % st == 0
            return True
    return _selector_accepts(step, token)


def plan_probes(steps: List[Step], descriptors) -> List[Tuple[str, Pointer, Tuple[Step, ...]]]:
    """Compute deduplicated localized probes for one batch.

    Probe kinds:
      * ``"node"``      - evaluate a suffix (possibly empty) at one anchor;
      * ``"container"`` - a structurally changed list, evaluate a suffix over
                          its whole pre/post state.
    """
    probes: Dict[tuple, tuple] = {}

    def add(kind: str, anchor: Pointer, suffix: List[Step]):
        key = (kind, anchor, tuple(suffix))
        probes.setdefault(key, (kind, anchor, tuple(suffix)))

    for path, change in descriptors:
        states = _states_with_filter_guard(steps, path)
        structural = change in ("insert", "remove")
        for k, d, guard_depth, window_depth in states:
            prefix = tuple(path[:d])
            suffix = list(steps[k:])
            guarded = guard_depth >= 0
            windowed = window_depth >= 0
            if k == len(steps) and not guarded and not windowed:
                add("node", prefix, [])
            if d == len(path):
                if structural:
                    if suffix and suffix[0].kind in (
                        "index", "slice", "filter", "desc",
                    ):
                        add("container", path, suffix)
                elif guarded:
                    guard_anchor = tuple(path[:guard_depth])
                    add("container", guard_anchor, list(steps[guard_depth:]))
                elif windowed and not structural:
                    # A change inside a bounded-slice element may move it out
                    # of the window or alter window membership; re-evaluate the
                    # window over its container.
                    win_anchor = tuple(path[:window_depth])
                    add("container", win_anchor, list(steps[window_depth:]))
                else:
                    add("node", path, suffix)
                continue
            if suffix and suffix[0].kind == "filter" and not structural:
                add("container", prefix, suffix)
    return list(probes.values())


def _states_with_filter_guard(steps: List[Step], path: Pointer):
    """Reachability states annotated with the innermost pending filter index.

    ``guarded`` is the spine index of the filter that dynamically decides the
    current element's membership (or ``len(steps)`` when no filter is active).
    A node reached *through* a filter is still guarded until the path moves to
    the filter's container (one token shallower).
    """
    # state: (k, d, guard_depth, window_depth). `guard_depth` marks the
    # innermost filter container; `window_depth` the innermost bounded
    # (non-full) slice/index container; -1 means none active.
    states = {(0, 0, -1, -1)}
    progressed = True
    while progressed:
        progressed = False
        for k, d, guard, window in list(states):
            if d >= len(path) or k > len(steps):
                continue
            token = path[d]
            step = steps[k] if k < len(steps) else None
            if step is None:
                continue
            accepted = _precisely_accepts(step, token, path, d)
            if step.kind == "desc":
                gap = (k, d + 1, guard, window)
                if gap not in states:
                    states.add(gap); progressed = True
                if accepted:
                    ng = d if step.payload.kind == "filter" else guard
                    nw = _window_depth(step.payload, d, window)
                    nxt = (k + 1, d + 1, ng, nw)
                    if nxt not in states:
                        states.add(nxt); progressed = True
            elif accepted:
                ng = d if step.kind == "filter" else guard
                nw = _window_depth(step, d, window)
                nxt = (k + 1, d + 1, ng, nw)
                if nxt not in states:
                    states.add(nxt); progressed = True
    return states


def _window_depth(step: Step, token_depth: int, current: int) -> int:
    """Record a bounded selector's container depth (full slices don't bind)."""
    if step.kind == "slice":
        start, end, st = step.payload
        if start is None and end is None and st in (None, 1):
            return current
        return token_depth  # container is one token shallower than the element
    if step.kind == "index":
        return token_depth
    return current


def move_index_registry(events: List[tuple]):
    """Array-to-array moves keyed by move id: src/dst list and index."""
    removes = {}
    inserts = {}
    for event in events:
        if event[0] == "remove" and event[4] is not None:
            removes[event[4]] = (event[1], event[2])
        elif event[0] == "insert" and event[4] is not None:
            inserts[event[4]] = (event[1], event[2])
    return {
        move_id: (removes[move_id], inserts[move_id])
        for move_id in removes
        if move_id in inserts
    }


def translate_through(
    ptr: Pointer,
    simulations: Dict[Pointer, dict],
    moves: dict,
) -> Optional[Pointer]:
    """Pre-batch pointer -> post-batch pointer (``None`` if deleted)."""
    tokens = list(ptr)
    remaining = dict(simulations)
    # Each structurally changed list appears at most once along a concrete
    # pointer; apply its index mapping at the token immediately after its
    # prefix, deepest-prefix-first so earlier edits do not invalidate later
    # prefix matching.
    for _ in range(len(remaining) + 2):
        matched: List[Tuple[int, str]] = []
        for list_ptr in remaining:
            depth = len(list_ptr)
            if tuple(tokens[:depth]) != list_ptr:
                continue
            if depth < len(tokens) and isinstance(tokens[depth], int):
                matched.append((depth, list_ptr))
        if not matched:
            break
        depth, list_ptr = min(matched, key=lambda item: item[0])
        fwd = simulations[list_ptr]["fwd"]
        old_index = tokens[depth]
        if old_index in fwd:
            tokens[depth] = fwd[old_index]
            # Consume this list so the next iteration handles deeper/other ones.
            remaining.pop(list_ptr, None)
            continue
        # The element was removed here. A move carries it to another list.
        carried = None
        for move_id, ((src_list, src_index), (dst_list, dst_index)) in moves.items():
            if src_list == list_ptr and src_index == old_index:
                carried = (dst_list, dst_index)
                break
        if carried is None:
            return None
        dst_list, dst_index = carried
        tokens = list(dst_list) + [dst_index] + tokens[depth + 1:]
        remaining.pop(list_ptr, None)
    return tuple(tokens)
