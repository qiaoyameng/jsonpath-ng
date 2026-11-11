"""Tests for incremental JSONPath subscription sessions."""

from __future__ import annotations

import gc
import weakref

import pytest

from jsonpath_ng.ext import parse
from jsonpath_ng.subscription import IllegalPatchPathError
from jsonpath_ng.subscription import Patch
from jsonpath_ng.subscription import PatchOperation
from jsonpath_ng.subscription import PatchVersionError
from jsonpath_ng.subscription import SubscriptionDocument
from jsonpath_ng.subscription import _match_path


def _full_find(expression, data):
    """Return the authoritative hit set produced by a full ``find`` run."""

    return {
        _match_path(match): match.value for match in expression.find(data)
    }


@pytest.fixture
def catalog():
    return {
        "threshold": 10,
        "store": {
            "items": [
                {"name": "a", "price": 5, "tags": ["x"]},
                {"name": "b", "price": 20, "tags": ["y", "z"]},
                {"name": "c", "price": 15, "tags": ["x"]},
            ],
            "meta": {"region": "eu"},
        },
    }


def test_initial_hit_set_matches_full_find(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[?price > $.threshold].name")
    subscription = session.register(expression)

    expected = {
        ("store", "items", index, "name"): item["name"]
        for index, item in enumerate(catalog["store"]["items"])
        if item["price"] > catalog["threshold"]
    }
    assert subscription.hits == expected
    assert subscription.hits == _full_find(expression, catalog)


def test_scalar_filter_threshold_change_produces_add_and_remove(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[?price > $.threshold].name")
    subscription = session.register(expression)
    assert set(subscription.hits) == {
        ("store", "items", 1, "name"),
        ("store", "items", 2, "name"),
    }

    # Lowering the threshold makes item "a" match too.
    changes = document.apply_patch(
        Patch(0, [PatchOperation("replace", ("threshold",), 2)])
    )
    (patch,) = changes
    assert patch.from_version == 0
    assert patch.to_version == 1
    assert [operation.path for operation in patch.added] == [
        ("store", "items", 0, "name")
    ]
    assert patch.removed == []
    assert document.version == 1

    # Raising it back removes the two cheapest qualifying items.
    changes = document.apply_patch(
        Patch(1, [PatchOperation("replace", ("threshold",), 18)])
    )
    (patch,) = changes
    removed = sorted(operation.path for operation in patch.removed)
    assert removed == [
        ("store", "items", 0, "name"),
        ("store", "items", 2, "name"),
    ]
    assert subscription.hits == {("store", "items", 1, "name"): "b"}


def test_root_reference_filter_is_invalidated_by_root_field(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    subscription = session.register(
        parse("$.store.items[?price > $.threshold].name")
    )

    # A change to an unrelated root field must not recompute this filter.
    document.reset_access_count()
    document.apply_patch(
        Patch(0, [PatchOperation("replace", ("store", "meta", "region"), "us")])
    )
    assert document.access_count == 0

    # A change to the referenced root field must invalidate it.
    document.reset_access_count()
    document.apply_patch(
        Patch(1, [PatchOperation("replace", ("threshold",), 0)])
    )
    assert document.access_count > 0
    assert set(subscription.hits) == {
        ("store", "items", 0, "name"),
        ("store", "items", 1, "name"),
        ("store", "items", 2, "name"),
    }


def test_head_insert_shifts_wildcard_hits(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[*].name")
    subscription = session.register(expression)
    assert list(subscription.hits.values()) == ["a", "b", "c"]

    changes = document.apply_patch(
        Patch(
            0,
            [PatchOperation("add", ("store", "items", 0), {"name": "z", "price": 1})],
        )
    )
    (patch,) = changes
    # The new element appears at index 0; every existing hit shifts right.
    assert [operation.path for operation in patch.added] == [
        ("store", "items", 0, "name")
    ]
    moved_paths = {
        (operation.from_path, operation.path) for operation in patch.moved
    }
    assert (
        ("store", "items", 0, "name"),
        ("store", "items", 1, "name"),
    ) in moved_paths
    assert (
        ("store", "items", 1, "name"),
        ("store", "items", 2, "name"),
    ) in moved_paths
    assert (
        ("store", "items", 2, "name"),
        ("store", "items", 3, "name"),
    ) in moved_paths
    assert list(subscription.hits.values()) == ["z", "a", "b", "c"]
    assert subscription.hits == _full_find(expression, document.data)


def test_delete_container_removes_descendant_hits(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[*].name")
    subscription = session.register(expression)
    assert len(subscription.hits) == 3

    changes = document.apply_patch(
        Patch(0, [PatchOperation("remove", ("store", "items"))])
    )
    (patch,) = changes
    assert len(patch.removed) == 3
    assert subscription.hits == {}
    assert "items" not in document.data["store"]
    assert subscription.hits == _full_find(expression, document.data)


def test_replace_leaf_value_reports_replace(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[0].name")
    subscription = session.register(expression)
    assert subscription.hits == {("store", "items", 0, "name"): "a"}

    changes = document.apply_patch(
        Patch(0, [PatchOperation("replace", ("store", "items", 0, "name"), "A")])
    )
    (patch,) = changes
    assert [operation.value for operation in patch.replaced] == ["A"]
    assert subscription.hits == {("store", "items", 0, "name"): "A"}


def test_move_reports_move_operation(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[*].name")
    subscription = session.register(expression)

    # Move the whole first element to the end of the array; its name hit
    # travels with it and must be reported as a move rather than remove/add.
    changes = document.apply_patch(
        Patch(
            0,
            [
                PatchOperation(
                    "move",
                    ("store", "items", "-"),
                    from_path=("store", "items", 0),
                )
            ],
        )
    )
    (patch,) = changes
    # Each name hit shifts position; the moved element's name lands last.
    moved_pairs = {
        (operation.from_path, operation.path): operation.value
        for operation in patch.moved
    }
    assert moved_pairs[
        (
            ("store", "items", 0, "name"),
            ("store", "items", 2, "name"),
        )
    ] == "a"
    assert patch.added == []
    assert patch.removed == []
    assert patch.replaced == []
    assert list(subscription.hits.values()) == ["b", "c", "a"]
    assert subscription.hits == _full_find(expression, document.data)


def test_unaffected_branches_are_not_visited(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    session.register(parse("$.store.items[*].name"))

    # Changing an unrelated root field touches nothing below the expression.
    document.reset_access_count()
    document.apply_patch(
        Patch(0, [PatchOperation("replace", ("threshold",), 99)])
    )
    assert document.access_count == 0

    # A change inside the expression's subtree only visits that branch.
    document.reset_access_count()
    document.apply_patch(
        Patch(1, [PatchOperation("replace", ("store", "items", 0, "name"), "A")])
    )
    assert document.access_count > 0


def test_bad_base_version_leaves_document_and_subscriptions_unchanged(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[*].name")
    subscription = session.register(expression)
    before_hits = subscription.hits
    before_data = document.data

    with pytest.raises(PatchVersionError):
        document.apply_patch(
            Patch(5, [PatchOperation("replace", ("threshold",), 0)])
        )
    assert document.version == 0
    assert document.data is before_data
    assert subscription.hits == before_hits


def test_illegal_path_leaves_document_and_subscriptions_unchanged(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$.store.items[*].name")
    subscription = session.register(expression)
    before_hits = subscription.hits
    before_data = document.data

    with pytest.raises(IllegalPatchPathError):
        document.apply_patch(
            Patch(0, [PatchOperation("replace", ("store", "missing", "x"), 1)])
        )
    with pytest.raises(IllegalPatchPathError):
        document.apply_patch(
            Patch(0, [PatchOperation("remove", ("store", "items", 99))])
        )
    assert document.version == 0
    assert document.data is before_data
    assert subscription.hits == before_hits


def test_unsupported_expression_recomputes_fully(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expression = parse("$..name")
    subscription = session.register(expression)
    assert subscription.maintainable is False

    document.apply_patch(
        Patch(
            0,
            [PatchOperation("add", ("store", "items", 3), {"name": "d", "price": 1})],
        )
    )
    assert subscription.hits == _full_find(expression, document.data)


def test_session_close_releases_document_and_subscriptions(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    subscription = session.register(parse("$.store.items[*].name"))

    document_ref = weakref.ref(document)
    subscription_ref = weakref.ref(subscription)

    session.close()
    assert session.closed
    assert session.document is None
    del subscription
    gc.collect()
    assert subscription_ref() is None

    del document
    gc.collect()
    assert document_ref() is None


def test_document_close_closes_sessions(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    session.register(parse("$.store.items[*].name"))
    document.close()
    assert session.closed
    assert document.data is None


def test_hits_match_full_find_after_every_update(catalog):
    """Apply a mixed sequence of patches and verify against full find."""

    document = SubscriptionDocument(catalog)
    session = document.open_session()
    expressions = [
        parse("$.store.items[*].name"),
        parse("$.store.items[?price > $.threshold].name"),
        parse("$.store.items[0].price"),
        parse("$.store.meta.region"),
    ]
    subscriptions = [session.register(expression) for expression in expressions]

    patches = [
        # Head insert shifts every array element.
        [PatchOperation("add", ("store", "items", 0), {"name": "z", "price": 1})],
        # Threshold change flips filter membership.
        [PatchOperation("replace", ("threshold",), 2)],
        # A leaf value replacement.
        [PatchOperation("replace", ("store", "items", 1, "price"), 99)],
        # A whole container is deleted.
        [PatchOperation("remove", ("store", "meta"))],
        # A new member is added to an object.
        [PatchOperation("add", ("store", "meta"), {"region": "apac"})],
        # An element moves within the array.
        [
            PatchOperation(
                "move", ("store", "items", "-"), from_path=("store", "items", 0)
            )
        ],
    ]

    for index, operations in enumerate(patches):
        document.apply_patch(Patch(document.version, operations))
        for expression, subscription in zip(expressions, subscriptions):
            assert subscription.hits == _full_find(expression, document.data), (
                f"after patch {index}, {expression} diverged from full find"
            )


def test_register_on_closed_session_raises(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    session.close()
    with pytest.raises(Exception):
        session.register(parse("$.store.items[*].name"))


def test_multiple_subscriptions_update_independently(catalog):
    document = SubscriptionDocument(catalog)
    session = document.open_session()
    names = session.register(parse("$.store.items[*].name"))
    prices = session.register(parse("$.store.items[*].price"))

    changes = document.apply_patch(
        Patch(0, [PatchOperation("replace", ("store", "items", 0, "price"), 7)])
    )
    assert len(changes) == 2
    names_patch, prices_patch = changes
    assert names_patch.operations == []
    assert [op.value for op in prices_patch.replaced] == [7]
    assert names.hits == _full_find(names.expression, document.data)
    assert prices.hits == _full_find(prices.expression, document.data)
