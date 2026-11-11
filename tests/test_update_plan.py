"""
Tests for the declarative ``jsonpath_ng.update_plan`` module.

The plans resolve every expression against the same original document and
apply the resolved positions in one pass, unlike chained ``update()`` and
``filter()`` calls whose matches and indices shift after each mutation.
"""

import copy

import pytest

from jsonpath_ng import PlanRule
from jsonpath_ng import PlannedUpdate
from jsonpath_ng import UpdatePlan
from jsonpath_ng.exceptions import UpdatePlanConflictError
from jsonpath_ng.exceptions import UpdatePlanStaleError
from jsonpath_ng.exceptions import UpdatePlanTypeError
from jsonpath_ng.ext.parser import parse as ext_parse
from jsonpath_ng.parser import parse as base_parse

CLEANING_DOCUMENT = {
    "threshold": 15,
    "items": [
        {"price": 10, "ok": False},
        {"price": 20, "ok": True},
        {"price": 30, "ok": False},
        {"price": 40, "ok": True},
    ],
}


def test_plan_resolves_all_rules_against_original_document():
    """A head-of-array delete must not shift later original-document indices."""

    document = copy.deepcopy(CLEANING_DOCUMENT)
    plan = UpdatePlan.build(
        document,
        [
            PlanRule.delete(ext_parse("items[?@.ok == false]")),
            PlanRule.replace(ext_parse("items[1].price"), 99),
            PlanRule.replace(ext_parse("items[3].price"), 88),
        ],
    )

    assert plan.paths() == [
        '$["items"][0]',
        '$["items"][1]["price"]',
        '$["items"][2]',
        '$["items"][3]["price"]',
    ]
    assert {update.operation for update in plan} == {"delete", "replace"}

    result = plan.apply()

    # Indices 0 and 2 are removed; the surviving original indices 1 and 3 keep
    # their replacements, interpreted against the original array.
    assert result == {
        "threshold": 15,
        "items": [
            {"price": 99, "ok": True},
            {"price": 88, "ok": True},
        ],
    }


def test_chained_update_and_filter_suffer_index_shift():
    """Demonstrate the problem the plan avoids: sequential calls shift indices."""

    document = copy.deepcopy(CLEANING_DOCUMENT)
    ext_parse("items[?@.ok == false]").filter(lambda _: True, document)
    # Original index 3 is now at position 1, so the old index now points at
    # a different element than the one intended in the original document.
    ext_parse("items[1].price").update(document, 99)

    assert document["items"] == [
        {"price": 20, "ok": True},
        {"price": 99, "ok": True},
    ]


def test_root_threshold_filter_is_evaluated_on_original_document():
    """Filters referencing ``$`` resolve the root of the unmodified document."""

    document = copy.deepcopy(CLEANING_DOCUMENT)
    plan = UpdatePlan.build(
        document,
        [PlanRule.delete(ext_parse("items[?@.price > $.threshold]"))],
    )

    assert plan.paths() == ['$["items"][1]', '$["items"][2]', '$["items"][3]']

    plan.apply()

    assert document == {
        "threshold": 15,
        "items": [{"price": 10, "ok": False}],
    }


def test_shared_subcontainer_is_updated_at_every_alias():
    """Two aliases of the same object stay distinct plan positions."""

    shared = {"id": 1, "price": 10}
    document = {
        "groups": [shared, {"id": 2, "price": 20}],
        "primary": shared,
    }

    plan = UpdatePlan.build(
        document,
        [
            PlanRule.replace(ext_parse("groups[0].price"), 99),
            PlanRule.replace(ext_parse("primary.price"), 99),
        ],
    )

    assert plan.paths() == [
        '$["groups"][0]["price"]',
        '$["primary"]["price"]',
    ]
    plan.apply()

    assert document["groups"][0] is document["primary"]
    assert document["groups"][0]["price"] == 99
    assert document["groups"][1]["price"] == 20


def test_same_container_position_hit_twice_updates_once():
    """Repeated matches of one location merge, recording every source expression."""

    document = {"a": {"b": 1}}
    callback_calls = []

    def bump(value, parent, key):
        callback_calls.append(key)
        return value + 1

    plan = UpdatePlan.build(
        document,
        [
            PlanRule.replace_with(ext_parse("a.b"), bump),
            PlanRule.replace_with(ext_parse("$.a.b"), bump),
        ],
    )

    assert len(plan) == 1
    update = plan.updates[0]
    assert update.rendered_path == '$["a"]["b"]'
    assert update.sources == ("a.b", "$.a.b")

    plan.apply()
    assert callback_calls == ["b"]
    assert document == {"a": {"b": 2}}


def test_equal_values_at_different_positions_are_not_merged():
    document = {"x": 1, "y": 1}

    plan = UpdatePlan.build(document, [PlanRule.replace(ext_parse("$.*"), 7)])

    assert plan.paths() == ['$["x"]', '$["y"]']
    plan.apply()
    assert document == {"x": 7, "y": 7}


def test_repeated_array_index_resolves_to_one_position():
    document = {"a": [1, 2, 3]}
    seen = []

    plan = UpdatePlan.build(
        document,
        [
            PlanRule.replace_with(
                ext_parse("a[0,0,1]"),
                lambda value, parent, key: (seen.append(key), value + 1)[1],
            )
        ],
    )

    assert plan.paths() == ['$["a"][0]', '$["a"][1]']
    plan.apply()
    assert seen == [0, 1]
    assert document == {"a": [2, 3, 3]}


def test_replace_and_delete_conflict_reports_path_and_sources():
    with pytest.raises(UpdatePlanConflictError) as exc_info:
        UpdatePlan.build(
            {"a": {"b": 1}},
            [
                PlanRule.replace(ext_parse("a.b"), 5),
                PlanRule.delete(ext_parse("a.b")),
            ],
        )

    message = str(exc_info.value)
    assert '$["a"]["b"]' in message
    assert "a.b" in message
    assert "replaced and deleted" in message


def test_incompatible_replacement_values_conflict():
    with pytest.raises(UpdatePlanConflictError) as exc_info:
        UpdatePlan.build(
            {"a": {"b": 1}},
            [
                PlanRule.replace(ext_parse("a.b"), 5),
                PlanRule.replace(ext_parse("$.a.b"), 6),
            ],
        )

    message = str(exc_info.value)
    assert '$["a"]["b"]' in message
    assert "5" in message and "6" in message


def test_equal_replacement_values_at_same_position_compose():
    document = {"a": {"b": 1}}

    plan = UpdatePlan.build(
        document,
        [
            PlanRule.replace(ext_parse("a.b"), 5),
            PlanRule.replace(ext_parse("$.a.b"), 5),
        ],
    )

    assert len(plan) == 1
    plan.apply()
    assert document == {"a": {"b": 5}}


def test_parent_child_conflict_reports_both_paths_and_sources():
    with pytest.raises(UpdatePlanConflictError) as exc_info:
        UpdatePlan.build(
            {"a": {"b": 1}},
            [
                PlanRule.delete(ext_parse("a")),
                PlanRule.replace(ext_parse("a.b"), 5),
            ],
        )

    message = str(exc_info.value)
    assert '$["a"]["b"]' in message
    assert '$["a"]' in message
    assert "a.b" in message and "a" in message


def test_failed_build_does_not_modify_the_document():
    document = {"a": {"b": 1}}
    snapshot = copy.deepcopy(document)

    with pytest.raises(UpdatePlanConflictError):
        UpdatePlan.build(
            document,
            [
                PlanRule.delete(ext_parse("a")),
                PlanRule.replace(ext_parse("a.b"), 5),
            ],
        )

    assert document == snapshot


def test_plan_rejects_unsupported_containers():
    with pytest.raises(UpdatePlanTypeError):
        UpdatePlan.build({"a": (1, 2)}, [PlanRule.delete(ext_parse("a"))])

    class Marker:
        pass

    with pytest.raises(UpdatePlanTypeError):
        UpdatePlan.build({"a": Marker()}, [PlanRule.delete(ext_parse("a"))])

    with pytest.raises(UpdatePlanTypeError):
        UpdatePlan.build({1: "x"}, [PlanRule.delete(ext_parse("$.*"))])


def test_stale_plan_is_refused_and_document_unchanged():
    document = {"a": 1}
    plan = UpdatePlan.build(document, [PlanRule.replace(ext_parse("a"), 2)])
    assert plan.is_current()

    document["a"] = 42
    assert not plan.is_current()

    with pytest.raises(UpdatePlanStaleError):
        plan.apply()
    assert document == {"a": 42}


def test_approval_callback_runs_once_per_approved_position():
    document = {"items": [1, 2, 3], "label": "x"}
    plan = UpdatePlan.build(
        document,
        [
            PlanRule.replace(ext_parse("items[*]"), 0),
            PlanRule.replace(ext_parse("label"), "y"),
        ],
    )

    approved = []
    plan.apply(lambda update: approved.append(update.rendered_path))

    assert approved == plan.paths()
    assert len(approved) == 4


def test_last_approval_callback_failure_leaves_document_and_plan_intact():
    document = {
        "items": [
            {"price": 10, "ok": False},
            {"price": 20, "ok": True},
            {"price": 30, "ok": False},
        ],
    }
    snapshot = copy.deepcopy(document)
    plan = UpdatePlan.build(
        document,
        [
            PlanRule.delete(ext_parse("items[?@.ok == false]")),
            PlanRule.replace(ext_parse("items[1].price"), 99),
        ],
    )
    planned_paths = plan.paths()
    planned_digest = plan.digest

    seen = []

    def approval(update):
        seen.append(update.rendered_path)
        if len(seen) == len(plan):
            raise RuntimeError("last approval rejected")

    with pytest.raises(RuntimeError, match="last approval rejected"):
        plan.apply(approval)

    # The input document must be byte-for-byte unchanged after the abort.
    assert document == snapshot
    # The computed plan remains inspectable and reusable after the failure.
    assert plan.paths() == planned_paths
    assert plan.digest == planned_digest
    assert plan.is_current()

    plan.apply()
    assert document == {"items": [{"price": 99, "ok": True}]}


def test_failing_replacement_callback_aborts_without_partial_mutation():
    document = {"a": 1, "b": 2, "c": 3}
    snapshot = copy.deepcopy(document)

    def maybe_fail(value, parent, key):
        if key == "c":
            raise ValueError("callback boom")
        return value + 10

    plan = UpdatePlan.build(
        document, [PlanRule.replace_with(ext_parse("$.*"), maybe_fail)]
    )

    with pytest.raises(ValueError, match="callback boom"):
        plan.apply()
    assert document == snapshot


def test_reordering_non_conflicting_rules_yields_identical_plans_and_results():
    rules = [
        PlanRule.delete(ext_parse("items[?@.ok == false]")),
        PlanRule.replace_with(
            ext_parse("items[1].price"), lambda value, parent, key: value * 10
        ),
        PlanRule.replace(ext_parse("threshold"), 0),
    ]

    document_a = copy.deepcopy(CLEANING_DOCUMENT)
    document_b = copy.deepcopy(CLEANING_DOCUMENT)
    plan_a = UpdatePlan.build(document_a, rules)
    plan_b = UpdatePlan.build(document_b, list(reversed(rules)))

    assert plan_a.paths() == plan_b.paths()
    assert plan_a.apply() == plan_b.apply()


def test_apply_uses_in_place_mutation_not_serialized_text_substitution():
    document = {"items": [{"price": 10}, {"price": 20}]}
    nested = document["items"]

    plan = UpdatePlan.build(
        document, [PlanRule.replace(ext_parse("items[0].price"), 99)]
    )
    result = plan.apply()

    assert result is document
    assert result["items"] is nested
    assert nested[0] == {"price": 99}


def test_delete_within_dict_and_array_keep_original_positions():
    document = {
        "keep": {"v": 1},
        "drop": {"v": 2},
        "array": ["a", "b", "c", "d"],
    }

    UpdatePlan.build(
        document,
        [
            PlanRule.delete(ext_parse("drop")),
            PlanRule.delete(ext_parse("array[0,2]")),
            PlanRule.replace(ext_parse("array[1]"), "B"),
        ],
    ).apply()

    assert document == {"keep": {"v": 1}, "array": ["B", "d"]}


def test_negative_indices_are_resolved_against_the_original_array():
    document = {"a": [1, 2, 3, 4]}

    plan = UpdatePlan.build(
        document,
        [
            PlanRule.delete(ext_parse("a[-1]")),
            PlanRule.replace(ext_parse("a[-2]"), 30),
        ],
    )

    # Negative indices are normalized to concrete positions in the origin.
    assert plan.paths() == ['$["a"][2]', '$["a"][3]']
    plan.apply()
    assert document == {"a": [1, 2, 30]}


def test_callback_returning_none_keeps_original_value():
    document = {"a": 1, "b": 2}

    UpdatePlan.build(
        document,
        [PlanRule.replace_with(ext_parse("a"), lambda value, parent, key: None)],
    ).apply()

    assert document == {"a": 1, "b": 2}


def test_root_replace_and_delete():
    document = {"a": 1}
    plan = UpdatePlan.build(document, [PlanRule.replace(base_parse("$"), {"b": 2})])
    assert plan.paths() == ["$"]
    assert plan.apply() == {"b": 2}

    root_document = {"a": 1}
    root_plan = UpdatePlan.build(root_document, [PlanRule.delete(base_parse("$"))])
    assert root_plan.apply() is None


def test_plan_length_iteration_and_repr():
    document = {"a": 1, "b": 2}
    plan = UpdatePlan.build(document, [PlanRule.replace(ext_parse("$.*"), 0)])

    assert len(plan) == 2
    assert all(isinstance(update, PlannedUpdate) for update in plan)
    assert repr(plan).startswith("UpdatePlan(updates=2, digest=")


def test_empty_plan_is_a_noop():
    document = {"a": 1}
    plan = UpdatePlan.build(document, [])

    assert plan.paths() == []
    assert plan.apply() is document
    assert document == {"a": 1}


def test_digest_marks_structural_equality_and_identity_independent_changes():
    document = {"a": 1, "b": [2, 3]}
    plan = UpdatePlan.build(document, [PlanRule.replace(ext_parse("a"), 2)])

    assert plan.is_current(copy.deepcopy(document))
    changed = copy.deepcopy(document)
    changed["b"].append(4)
    assert not plan.is_current(changed)


def test_legacy_update_and_filter_semantics_unchanged():
    document = {"a": [1, 2, 3]}
    ext_parse("a[0]").update(document, 9)
    assert document == {"a": [9, 2, 3]}

    ext_parse("a[*]").filter(lambda value: value == 2, document)
    assert document == {"a": [9, 3]}

    other = {"foo": [{"baz": 1}, {"baz": 2}]}
    ext_parse("foo[*].baz").filter(lambda value: value == 2, other)
    assert other == {"foo": [{"baz": 1}, {}]}
