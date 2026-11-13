import copy
import gc
import weakref

import pytest

import jsonpath_ng.ext
from jsonpath_ng.ext import MatchChange
from jsonpath_ng.ext import Notification
from jsonpath_ng.ext import PatchError
from jsonpath_ng.ext import SubscriptionClosedError
from jsonpath_ng.ext import SubscriptionLimitError
from jsonpath_ng.ext import SubscriptionSession


DOCUMENT = {
    "threshold": 10,
    "items": [
        {"id": 1, "score": 12, "name": "alpha", "active": True, "tags": ["a"]},
        {"id": 2, "score": 5, "name": "beta", "active": False, "tags": ["b"]},
        {"id": 3, "score": 20, "name": "gamma", "active": True, "tags": ["c"]},
        {"id": 4, "score": 8, "name": "delta", "active": True, "tags": []},
        {"id": 5, "score": 25, "name": "epsilon", "active": False, "tags": []},
    ],
}


def full_find(expression, document):
    parsed = jsonpath_ng.ext.parse(expression)
    return [datum.value for datum in parsed.find(document)]


@pytest.fixture()
def session():
    return SubscriptionSession(copy.deepcopy(DOCUMENT))


def test_initial_matches_match_full_find(session):
    subscription = session.subscribe("$.items[?@.score >= 10].name")
    assert subscription.incremental is True
    assert subscription.values == full_find(
        "$.items[?@.score >= 10].name", session.document
    )


def test_unsupported_expression_forces_full_recompute(session):
    subscription = session.subscribe("$..name")
    assert subscription.incremental is False
    notifications = session.commit(
        [{"op": "replace", "path": "/items/0/name", "value": "changed"}]
    )
    notification = notifications[subscription.id]
    assert notification.recomputed is True
    assert subscription.values == full_find("$..name", session.document)


def test_head_insert_reports_moves_and_discovers_new_match(session):
    subscription = session.subscribe("$.items[*].id")
    notifications = session.commit(
        [
            {
                "op": "add",
                "path": "/items/0",
                "value": {
                    "id": 99,
                    "score": 99,
                    "name": "zeta",
                    "active": True,
                    "tags": [],
                },
            }
        ]
    )
    notification = notifications[subscription.id]
    moves = {(change.old_path, change.path) for change in notification.moved}
    assert (("items", 4, "id"), ("items", 5, "id")) in moves
    assert subscription.values == full_find("$.items[*].id", session.document)


def test_head_insert_into_filter_evaluates_new_candidate(session):
    subscription = session.subscribe("$.items[?@.score >= 10].name")
    session.commit(
        [
            {
                "op": "add",
                "path": "/items/0",
                "value": {
                    "id": 99,
                    "score": 1,
                    "name": "low",
                    "active": True,
                    "tags": [],
                },
            }
        ]
    )
    assert subscription.values == full_find(
        "$.items[?@.score >= 10].name", session.document
    )
    assert "low" not in subscription.values


def test_filter_threshold_change_re_evaluates_candidate(session):
    subscription = session.subscribe("$.items[?@.score >= 10].name")
    notifications = session.commit(
        [{"op": "replace", "path": "/items/1/score", "value": 50}]
    )
    notification = notifications[subscription.id]
    assert any(change.kind == "add" for change in notification.added)
    assert subscription.values == full_find(
        "$.items[?@.score >= 10].name", session.document
    )


def test_root_referencing_filter_invalidates_on_root_change(session):
    expression = "$.items[?$.threshold == 10 && @.score >= 10].name"
    subscription = session.subscribe(expression)
    assert subscription.plan.references_root is True
    session.commit([{"op": "replace", "path": "/threshold", "value": 99}])
    assert subscription.values == full_find(expression, session.document)
    assert subscription.values == []


def test_deleting_container_removes_nested_matches(session):
    subscription = session.subscribe("$.items[*].tags[*]")
    session.commit([{"op": "remove", "path": "/items/0"}])
    assert subscription.values == full_find(
        "$.items[*].tags[*]", session.document
    )


def test_deleting_whole_container_clears_filter_matches(session):
    subscription = session.subscribe("$.items[?@.active].id")
    session.commit([{"op": "remove", "path": "/items"}])
    assert subscription.matches == {}


def test_rejected_patch_keeps_document_and_subscriptions(session):
    subscription = session.subscribe("$.items[*].id")
    before = copy.deepcopy(session.document)
    before_values = subscription.values
    with pytest.raises(PatchError):
        session.commit([{"op": "remove", "path": "/items/99"}])
    assert session.document == before
    assert subscription.values == before_values


def test_base_version_mismatch_is_atomic(session):
    with pytest.raises(PatchError):
        session.commit(
            [{"op": "replace", "path": "/threshold", "value": 1}],
            base_version=12345,
        )
    assert session.version == 0


def test_notification_carries_versions(session):
    subscription = session.subscribe("$.threshold")
    notifications = session.commit(
        [{"op": "replace", "path": "/threshold", "value": 20}]
    )
    notification = notifications[subscription.id]
    assert isinstance(notification, Notification)
    assert notification.base_version == 0
    assert notification.version == 1


def test_value_change_is_reported_as_replace(session):
    subscription = session.subscribe("$.items[0].name")
    notifications = session.commit(
        [{"op": "replace", "path": "/items/0/name", "value": "renamed"}]
    )
    change = notifications[subscription.id].replaced[0]
    assert isinstance(change, MatchChange)
    assert change.old_value == "alpha"
    assert change.new_value == "renamed"


def test_unaffected_branch_is_not_traversed(session):
    wildcard = session.subscribe("$.items[*].id")
    notifications = session.commit(
        [{"op": "replace", "path": "/items/0/id", "value": 100}]
    )
    # Only the changed leaf and its container are visited, not the other
    # 99% of the array.
    assert notifications[wildcard.id].visits <= 2


def test_single_candidate_filter_change_is_local(session):
    subscription = session.subscribe("$.items[?@.score >= 10].name")
    notifications = session.commit(
        [{"op": "replace", "path": "/items/1/score", "value": 50}]
    )
    # Re-evaluating one candidate touches just that candidate's reads,
    # independent of the array length.
    assert notifications[subscription.id].visits <= 3
    assert subscription.values == full_find(
        "$.items[?@.score >= 10].name", session.document
    )


def test_recorded_dependencies_cover_filter_reads(session):
    subscription = session.subscribe("$.items[?@.score >= 10].name")
    dependencies = subscription.dependencies
    assert ("items",) in [path for _depth, path in dependencies["predicate_anchors"]]
    assert ("items", 1, "score") not in dependencies["values"]
    # The filter iterates its input container, recorded as an input path.
    assert ("items",) in dependencies["inputs"]


def test_fixed_index_dependencies_do_not_go_stale(session):
    subscription = session.subscribe("$.items[-1].id")
    session.commit(
        [
            {
                "op": "add",
                "path": "/items/0",
                "value": {
                    "id": 99,
                    "score": 1,
                    "name": "z",
                    "active": True,
                    "tags": [],
                },
            }
        ]
    )
    # A negative index resolves to exactly one current node, even after the
    # array shifted under it.
    assert len(subscription.matches) == 1
    assert subscription.values == full_find("$.items[-1].id", session.document)


def test_session_enforces_subscription_limit():
    session = SubscriptionSession({}, max_subscriptions=1)
    session.subscribe("$.a")
    with pytest.raises(SubscriptionLimitError):
        session.subscribe("$.b")


def test_close_releases_document_and_subscriptions():
    session = SubscriptionSession({"a": 1}, max_subscriptions=4)
    first = session.subscribe("$.a")
    second = session.subscribe("$.a")
    references = [weakref.ref(first), weakref.ref(second)]
    store_reference = weakref.ref(session._store)
    session.close()
    assert session.closed is True
    assert first.closed is True
    with pytest.raises(SubscriptionClosedError):
        session.commit([])
    del first, second
    gc.collect()
    assert all(reference() is None for reference in references)
    assert store_reference() is None


def test_context_manager_closes_session():
    with SubscriptionSession({"x": 1}) as session:
        subscription = session.subscribe("$.x")
        assert subscription.values == [1]
    assert session.closed is True
    assert subscription.closed is True


@pytest.mark.parametrize(
    "expression",
    (
        "$.threshold",
        "$.items[0].name",
        "$.items[*].id",
        "$.items[?@.active].name",
        "$.items[?@.score >= 10].name",
        "$.items[?$.threshold == 10 && @.score >= 10].name",
        "$.items[*].tags[*]",
    ),
)
def test_incremental_results_always_match_full_find(session, expression):
    subscription = session.subscribe(expression)
    patches = [
        [{"op": "add", "path": "/items/0",
         "value": {"id": 99, "score": 1, "name": "z",
                   "active": True, "tags": []}}],
        [{"op": "replace", "path": "/items/1/score", "value": 77}],
        [{"op": "replace", "path": "/threshold", "value": 1}],
        [{"op": "remove", "path": "/items/2"}],
        [{"op": "replace", "path": "/items/0/name", "value": "renamed"}],
    ]
    for patch in patches:
        session.commit(patch)
        assert subscription.values == full_find(expression, session.document)
