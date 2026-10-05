"""End-to-end behaviour of incremental query sessions."""

import copy

import pytest
from jsonpath_ng.ext import parse as ext_parse

from jsonpath_ng.session import (
    MatchDelta,
    PatchBatchError,
    QuerySession,
)


STORE_DOC = {
    "store": {
        "book": [
            {"category": "fiction", "title": "Sayings", "price": 8},
            {"category": "cook", "title": "Sword", "price": 15},
            {"category": "fiction", "title": "Moby", "price": 5},
        ]
    },
    "noise": [{"v": i} for i in range(100)],
}


def find_pointers(doc, expression):
    """Reference implementation: a fresh full ``find`` each time."""
    from jsonpath_ng.session import engine

    result = {}
    for datum in ext_parse(expression).find(doc):
        tokens = []
        engine._collect(datum.full_path, tokens)
        result[tuple(tokens)] = datum.value
    return result


def assert_matches_equal(session, expression, doc):
    expected = find_pointers(doc, expression)
    actual = {
        tuple(
            __import__(
                "jsonpath_ng.session.pointer", fromlist=["parse_pointer"]
            ).parse_pointer(p)
        ): v
        for p, v in zip(
            session.match_pointers(expression),
            session.matches(expression),
        )
    }
    assert actual == expected


# ---------------------------------------------------------------------------
# Basic value changes on fields / indices / slices
# ---------------------------------------------------------------------------


def test_field_value_change_is_reported():
    doc = copy.deepcopy(STORE_DOC)
    session = QuerySession(doc, "$.store.book[0].title")
    delta = session.apply_patches(
        [{"op": "replace", "path": "/store/book/0/title", "value": "Other"}]
    )["$.store.book[0].title"]
    assert delta.value_changed == [("/store/book/0/title", "Sayings", "Other")]
    assert delta.added == []
    assert delta.removed == []
    assert delta.moved == []
    assert session.matches("$.store.book[0].title") == ["Other"]


def test_unrelated_field_change_is_empty():
    doc = copy.deepcopy(STORE_DOC)
    session = QuerySession(doc, "$.store.book[0].title")
    delta = session.apply_patches(
        [{"op": "replace", "path": "/store/book/1/title", "value": "X"}]
    )["$.store.book[0].title"]
    assert delta.empty


def test_index_selector_tracks_replacement():
    doc = {"items": [10, 20, 30]}
    session = QuerySession(doc, "$.items[2]")
    delta = session.apply_patches(
        [{"op": "replace", "path": "/items/2", "value": 99}]
    )["$.items[2]"]
    assert delta.value_changed == [("/items/2", 30, 99)]


# ---------------------------------------------------------------------------
# Stepwise array insert / delete with move vs value-change distinction
# ---------------------------------------------------------------------------


def test_insert_into_wildcard_reports_add_and_positional_moves():
    doc = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
    session = QuerySession(doc, "$.items[*].id")
    delta = session.apply_patches(
        [{"op": "add", "path": "/items/0", "value": {"id": 0}}]
    )["$.items[*].id"]
    assert delta.added == [("/items/0/id", 0)]
    assert delta.removed == []
    assert delta.value_changed == []
    assert sorted(delta.moved) == sorted(
        [
            ("/items/0/id", "/items/1/id", 1),
            ("/items/1/id", "/items/2/id", 2),
            ("/items/2/id", "/items/3/id", 3),
        ]
    )


def test_stepwise_delete_reports_remove_and_positional_moves():
    doc = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
    session = QuerySession(doc, "$.items[*].id")
    delta = session.apply_patches(
        [{"op": "remove", "path": "/items/1"}]
    )["$.items[*].id"]
    assert delta.removed == [("/items/1/id", 2)]
    assert delta.added == []
    assert delta.value_changed == []
    assert sorted(delta.moved) == sorted(
        [
            ("/items/2/id", "/items/1/id", 3),
        ]
    )


def test_slice_window_distinguishes_shift_from_window_entry():
    doc = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
    session = QuerySession(doc, "$.items[0:2].id")
    delta = session.apply_patches(
        [{"op": "remove", "path": "/items/0"}]
    )["$.items[0:2].id"]
    # Element 2 shifts inside the window (a positional move), element 3
    # enters the window as a genuine addition, element 1 is removed.
    assert delta.removed == [("/items/0/id", 1)]
    assert delta.added == [("/items/1/id", 3)]
    assert delta.moved == [("/items/1/id", "/items/0/id", 2)]
    assert delta.value_changed == []


def test_fixed_index_reports_remove_and_add_not_move():
    doc = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
    session = QuerySession(doc, "$.items[1].id")
    delta = session.apply_patches(
        [{"op": "remove", "path": "/items/0"}]
    )["$.items[1].id"]
    assert delta.removed == [("/items/1/id", 2)]
    assert delta.added == [("/items/1/id", 3)]
    assert delta.moved == []


def test_move_operation_is_recognised_as_positional_change():
    doc = {"items": [{"id": 10}, {"id": 20}, {"id": 30}]}
    session = QuerySession(doc, "$.items[*].id")
    delta = session.apply_patches(
        [{"op": "move", "from": "/items/0", "path": "/items/2"}]
    )["$.items[*].id"]
    assert delta.value_changed == []
    assert delta.added == []
    assert delta.removed == []
    moved = {src: (dst, value) for src, dst, value in delta.moved}
    assert moved["/items/0/id"] == ("/items/2/id", 10)


# ---------------------------------------------------------------------------
# Descendant selectors
# ---------------------------------------------------------------------------


def test_descendant_change_and_add():
    doc = {"a": {"x": {"name": 1}, "y": {"name": 2}}, "b": {"z": {"name": 3}}}
    session = QuerySession(doc, "$.a..name")
    delta = session.apply_patches(
        [{"op": "replace", "path": "/a/y/name", "value": 22}]
    )["$.a..name"]
    assert delta.value_changed == [("/a/y/name", 2, 22)]
    delta = session.apply_patches(
        [{"op": "add", "path": "/a/w", "value": {"name": 4}}]
    )["$.a..name"]
    assert delta.added == [("/a/w/name", 4)]


def test_descendant_ignores_other_branch():
    doc = {"a": {"x": {"name": 1}}, "b": {"z": {"name": 3}}}
    session = QuerySession(doc, "$.a..name")
    delta = session.apply_patches(
        [{"op": "replace", "path": "/b/z/name", "value": 30}]
    )["$.a..name"]
    assert delta.empty


# ---------------------------------------------------------------------------
# Filter field not present in the output path must trigger recomputation
# ---------------------------------------------------------------------------


def test_filter_predicate_field_change_triggers_recalculation():
    doc = {"items": [{"id": 1, "v": 10}, {"id": 2, "v": 50}, {"id": 3, "v": 5}]}
    session = QuerySession(doc, "$.items[?@.v < 20].id")
    # The matched output is `id`; patching `v` (the predicate field) must
    # still update membership.
    delta = session.apply_patches(
        [{"op": "replace", "path": "/items/1/v", "value": 10}]
    )["$.items[?@.v < 20].id"]
    assert delta.added == [("/items/1/id", 2)]
    delta = session.apply_patches(
        [{"op": "replace", "path": "/items/0/v", "value": 99}]
    )["$.items[?@.v < 20].id"]
    assert delta.removed == [("/items/0/id", 1)]


# ---------------------------------------------------------------------------
# Batch atomicity: validate & apply in order, no half batch on failure
# ---------------------------------------------------------------------------


def test_failed_batch_rolls_back_document_and_results():
    doc = {"a": [1, 2, 3], "b": 1}
    original = copy.deepcopy(doc)
    session = QuerySession(doc, "$.a[*]")
    with pytest.raises(PatchBatchError) as exc:
        session.apply_patches(
            [
                {"op": "replace", "path": "/b", "value": 99},
                {"op": "add", "path": "/a/-", "value": 4},
                {"op": "test", "path": "/b", "value": 1},
                {"op": "replace", "path": "/a/0", "value": 0},
            ]
        )
    assert exc.value.index == 2
    assert doc == original
    assert session.match_pointers("$.a[*]") == ["/a/0", "/a/1", "/a/2"]
    assert session.stats.batches_applied == 0


def test_failed_batch_does_not_publish_partial_deltas():
    doc = {"a": [1]}
    session = QuerySession(doc, "$.a[*]")
    with pytest.raises(PatchBatchError):
        session.apply_patches(
            [
                {"op": "replace", "path": "/a/0", "value": 9},
                {"op": "remove", "path": "/a/9"},
            ]
        )
    assert session.matches("$.a[*]") == [1]


# ---------------------------------------------------------------------------
# Access counting: unrelated subtree must not cause a full scan
# ---------------------------------------------------------------------------


def count_nodes(node):
    total = 1
    if isinstance(node, dict):
        total += sum(count_nodes(v) for v in node.values())
    elif isinstance(node, list):
        total += sum(count_nodes(v) for v in node)
    return total


def test_unrelated_subtree_patch_does_not_scan():
    doc = {
        "watched": [{"v": 1}, {"v": 2}],
        "noise": [{"v": i} for i in range(500)],
        "more": {"k%d" % i: {"deep": i} for i in range(500)},
    }
    total_nodes = count_nodes(doc)
    session = QuerySession(doc, "$.watched[*].v")
    assert total_nodes > 2000
    session.stats.nodes_visited = 0
    delta = session.apply_patches(
        [{"op": "replace", "path": "/noise/0/v", "value": 123}]
    )["$.watched[*].v"]
    assert delta.empty
    # Nothing in the query's region (or even under a wildcard) was touched:
    # only root resolution happened, well below scanning even `noise`.
    assert session.stats.nodes_visited <= 2


def test_watched_patch_visits_only_local_subtree():
    doc = {
        "watched": [{"v": 1}, {"v": 2}],
        "noise": [{"v": i} for i in range(500)],
    }
    session = QuerySession(doc, "$.watched[*].v")
    session.stats.nodes_visited = 0
    session.apply_patches(
        [{"op": "replace", "path": "/watched/0/v", "value": 99}]
    )
    # root, watched list, element 0, its v field == 4 nodes; never the noise.
    assert session.stats.nodes_visited <= 5


# ---------------------------------------------------------------------------
# Unsupported / unanalysable nodes fall back to full find with a reason
# ---------------------------------------------------------------------------


def test_unanalysable_query_falls_back_with_reason():
    doc = {"items": [1, 2, 3]}
    session = QuerySession(doc, "$.items[-1]")
    assert "$.items[-1]" in session.stats.fallback_queries
    reason = session.stats.fallback_queries["$.items[-1]"]
    assert "non-negative" in reason
    delta = session.apply_patches(
        [{"op": "replace", "path": "/items/2", "value": 30}]
    )["$.items[-1]"]
    assert delta.value_changed == [("/items/-1", 3, 30)]
    assert session.stats.full_recomputations >= 1
    assert session.stats.conservative_reasons


def test_where_query_falls_back():
    doc = {"items": [{"id": 1, "tag": "x"}, {"id": 2}]}
    expression = "$.items[*] where tag"
    session = QuerySession(doc, expression)
    assert expression in session.stats.fallback_queries
    delta = session.apply_patches(
        [{"op": "add", "path": "/items/1/tag", "value": "y"}]
    )[expression]
    assert len(delta.added) == 1
    assert session.stats.full_recomputations >= 1
