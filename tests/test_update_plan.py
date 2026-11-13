import copy
from collections import UserDict

import pytest

from jsonpath_ng import UpdatePlan
from jsonpath_ng import UpdatePlanConflictError
from jsonpath_ng import UpdatePlanStaleError
from jsonpath_ng import UnsupportedUpdateDataError
from jsonpath_ng import build_update_plan


CLEANING_DOC = {
    "threshold": 100,
    "items": [
        {"name": "cheap", "price": 50},
        {"name": "borderline", "price": 150},
        {"name": "expensive", "price": 200},
    ],
}


def test_rules_are_resolved_against_the_original_document():
    data = copy.deepcopy(CLEANING_DOC)
    plan = UpdatePlan(data)
    plan.replace(
        "$.items[?@.price < $.threshold].price",
        lambda value, parent, key: value + 10,
    )
    plan.delete("$.items[?@.price >= $.threshold]")

    assert data == CLEANING_DOC
    assert [(operation.action, operation.path) for operation in plan.operations] == [
        ("replace", "$.items[0].price"),
        ("delete", "$.items[1]"),
        ("delete", "$.items[2]"),
    ]

    result = plan.apply()

    assert result is data
    assert result == {
        "threshold": 100,
        "items": [{"name": "cheap", "price": 60}],
    }


def test_array_positions_and_head_deletes_use_the_original_array():
    data = {
        "items": [
            {"name": "first", "price": 1},
            {"name": "second", "price": 2},
            {"name": "third", "price": 3},
        ]
    }
    plan = UpdatePlan(data)
    plan.delete("$.items[0]")
    plan.delete("$.items[-3]")
    plan.replace("$.items[2].price", 30)

    assert len(plan.operations) == 2
    assert plan.operations[0].sources == ("$.items[0]", "$.items[-3]")

    plan.apply()

    assert data == {
        "items": [
            {"name": "second", "price": 2},
            {"name": "third", "price": 30},
        ]
    }


def test_shared_subcontainer_and_same_slot_only_run_once():
    shared = {"price": 10}
    data = {"left": shared, "right": shared}
    calls = []

    def replace(value, parent, key):
        calls.append((value, parent, key))
        return value + 1

    plan = UpdatePlan(data)
    plan.replace("$.left.price", replace)
    plan.replace("$.right.price", replace)

    assert plan.conflicts == ()
    assert len(plan.operations) == 1
    assert plan.operations[0].sources == ("$.left.price", "$.right.price")

    plan.apply()

    assert data == {"left": {"price": 11}, "right": {"price": 11}}
    assert len(calls) == 1
    assert calls[0][0] == 10
    assert calls[0][2] == "price"


def test_equal_values_at_distinct_positions_are_not_merged():
    data = {
        "left": {"price": 10},
        "right": {"price": 10},
    }
    calls = []

    def replace(value, parent, key):
        calls.append((id(parent), key))
        return value + len(calls)

    plan = UpdatePlan(data)
    plan.replace("$.left.price", replace)
    plan.replace("$.right.price", replace)

    assert len(plan.operations) == 2

    plan.apply()

    assert len(calls) == 2
    assert data == {"left": {"price": 11}, "right": {"price": 12}}


def test_parent_child_conflict_reports_paths_and_expressions():
    data = copy.deepcopy(CLEANING_DOC)
    plan = build_update_plan(data)
    plan.replace("$.items", [])
    plan.delete("$.items[0].price")

    assert len(plan.conflicts) == 1
    conflict = plan.conflicts[0]
    assert conflict.kind == "parent-child"
    assert str(conflict.path) == "$.items"
    assert conflict.expression == "$.items"
    assert str(conflict.related_path) == "$.items[0].price"
    assert conflict.related_expression == "$.items[0].price"

    with pytest.raises(UpdatePlanConflictError) as error:
        plan.apply()

    assert str(error.value.path) == "$.items"
    assert str(error.value.related_path) == "$.items[0].price"
    assert data == CLEANING_DOC


def test_shared_parent_child_conflict_is_detected():
    shared = {"child": {"price": 1}}
    data = {"left": shared, "right": shared}
    plan = UpdatePlan(data)
    plan.replace("$.left.child", {})
    plan.delete("$.right.child.price")

    assert [conflict.kind for conflict in plan.conflicts] == ["parent-child"]


def test_replace_delete_conflict_reports_the_canonical_path():
    data = {"price": 10}
    plan = UpdatePlan(data)
    plan.replace("$.price", 20)
    plan.delete("$.price")

    assert len(plan.conflicts) == 1
    conflict = plan.conflicts[0]
    assert conflict.kind == "replace-delete"
    assert str(conflict.path) == "$.price"
    assert conflict.expressions == ("$.price",)

    with pytest.raises(UpdatePlanConflictError):
        plan.apply()

    assert data == {"price": 10}


def test_last_callback_failure_leaves_document_and_plan_inspectable():
    data = {"items": [{"price": 1}, {"price": 2}]}
    original = copy.deepcopy(data)
    calls = []

    def first(value, parent, key):
        calls.append("first")
        parent[key] = value + 10
        return value + 10

    def last(value, parent, key):
        calls.append("last")
        parent[key] = value + 20
        raise RuntimeError("last callback failed")

    plan = UpdatePlan(data)
    plan.replace("$.items[0].price", first)
    plan.replace("$.items[1].price", last)
    original_operations = plan.operations
    original_digest = plan.digest

    with pytest.raises(RuntimeError, match="last callback failed"):
        plan.apply()

    assert calls == ["first", "last"]
    assert data == original
    assert plan.operations == original_operations
    assert [(operation.action, operation.path) for operation in plan.operations] == [
        ("replace", "$.items[0].price"),
        ("replace", "$.items[1].price"),
    ]
    assert plan.digest == original_digest


def test_stale_plan_is_refused_without_touching_the_document():
    data = {"threshold": 100, "items": [{"price": 50}]}
    plan = UpdatePlan(data)
    plan.replace("$.items[0].price", 60)

    data["threshold"] = 200

    with pytest.raises(UpdatePlanStaleError):
        plan.apply()

    assert data == {"threshold": 200, "items": [{"price": 50}]}
    assert str(plan.operations[0].path) == "$.items[0].price"


def test_swapping_non_conflicting_rules_produces_the_same_result():
    def run(rules):
        data = copy.deepcopy(CLEANING_DOC)
        plan = UpdatePlan(data)
        for rule in rules:
            plan.add_rule(rule)
        plan.apply()
        return data

    replace = (
        "replace",
        "$.items[?@.price < $.threshold].price",
        lambda value, parent, key: value + 10,
    )
    delete = ("delete", "$.items[?@.price >= $.threshold]", None)

    assert run([replace, delete]) == run([delete, replace])


@pytest.mark.parametrize(
    "data",
    (
        ("not-a-document",),
        [("tuple-item",)],
        UserDict({"a": 1}),
        {"a": {1, 2}},
    ),
)
def test_restricted_data_is_rejected(data):
    with pytest.raises(UnsupportedUpdateDataError):
        UpdatePlan(data)


def test_cycles_are_rejected():
    data = {}
    data["self"] = data

    with pytest.raises(UnsupportedUpdateDataError, match="cycle"):
        UpdatePlan(data)


def test_swapping_head_delete_and_array_replacement_produces_same_result():
    def run(rules):
        data = {
            "items": [
                {"price": 1},
                {"price": 2},
                {"price": 3},
            ]
        }
        plan = UpdatePlan(data, rules)
        plan.apply()
        return data

    delete = ("delete", "$.items[0]")
    replace = {"action": "replace", "path": "$.items[2].price", "value": 30}

    assert run([delete, replace]) == run([replace, delete])


def test_same_digest_document_can_be_applied():
    data = {"price": 10}
    plan = UpdatePlan(data, [("$.price", 20)])
    equivalent = copy.deepcopy(data)

    result = plan.apply(equivalent)

    assert result is equivalent
    assert equivalent == {"price": 20}
    assert data == {"price": 10}


def test_root_threshold_filter_uses_the_original_threshold():
    data = {
        "threshold": 100,
        "items": [{"price": 50}, {"price": 150}],
    }
    plan = UpdatePlan(data)
    plan.replace("$.threshold", 300)
    plan.delete("$.items[?@.price >= $.threshold]")

    assert [(operation.action, operation.path) for operation in plan.operations] == [
        ("replace", "$.threshold"),
        ("delete", "$.items[1]"),
    ]

    plan.apply()

    assert data == {"threshold": 300, "items": [{"price": 50}]}
