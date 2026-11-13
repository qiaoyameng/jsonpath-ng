"""Tests for the optional query diagnostics in `jsonpath_ng.diagnostics`."""

import pytest

from jsonpath_ng import JSONPath
from jsonpath_ng import QueryDiagnostics
from jsonpath_ng.diagnostics import INDEX_OUT_OF_RANGE
from jsonpath_ng.diagnostics import MISSING_FIELD
from jsonpath_ng.diagnostics import PREDICATE_FALSE
from jsonpath_ng.diagnostics import TYPE_MISMATCH
from jsonpath_ng.ext.parser import parse as ext_parse
from jsonpath_ng.jsonpath import DatumInContext
from jsonpath_ng.parser import parse as base_parse


# The same expression evaluated against two versions of one small response.
# Version A renamed a field; version B changed a value's type.
EXPRESSION = "$.store.book[?@.price < 10].title"
RESPONSE_RENAMED_FIELD = {"store": {"books": [{"title": "cheap", "price": 5}]}}
RESPONSE_RETYPED_VALUE = {"store": {"book": [{"title": "cheap", "price": "5"}]}}


def evaluate(expression, data, **kwargs):
    with QueryDiagnostics(**kwargs) as diagnostics:
        matches = ext_parse(expression).find(data)
    return matches, diagnostics


def test_first_candidate_loss_pinpoints_the_renamed_field():
    matches, diagnostics = evaluate(EXPRESSION, RESPONSE_RENAMED_FIELD)

    assert matches == []
    failure = diagnostics.first_failure
    assert failure is not None
    assert failure.failure == MISSING_FIELD
    assert failure.node_type == "Fields"
    assert "'book'" in failure.detail
    # The path of the datum that lacked the field is preserved.
    assert failure.path == "store"


def test_first_candidate_loss_pinpoints_the_changed_filter_condition():
    matches, diagnostics = evaluate(EXPRESSION, RESPONSE_RETYPED_VALUE)

    assert matches == []
    failure = diagnostics.first_failure
    assert failure is not None
    assert failure.failure == TYPE_MISMATCH
    assert failure.node_type == "Expression"
    assert "str" in failure.detail and "int" in failure.detail
    # Root references and nested filters keep the candidate's path.
    assert failure.path == "store.book.[0]"


def test_successful_query_records_counts_without_failures():
    data = {"store": {"book": [{"title": "cheap", "price": 5}]}}
    matches, diagnostics = evaluate(EXPRESSION, data)

    assert [match.value for match in matches] == ["cheap"]
    assert diagnostics.first_failure is None
    assert diagnostics.error is None
    # Every recorded step completed with an output count.
    assert all(event.output_count is not None for event in diagnostics.iter_events())
    # The filter saw one candidate and kept it.
    (filter_event,) = (
        event for event in diagnostics.iter_events() if event.node_type == "Filter"
    )
    assert filter_event.input_count == 1
    assert filter_event.output_count == 1


def test_results_are_identical_with_and_without_diagnostics():
    data = {"a": [{"b": 1}, {"b": 20}, {"c": 3}], "d": "x"}
    expression = ext_parse("$.a[?@.b > 10]")

    expected = [match.value for match in expression.find(data)]
    matches, diagnostics = evaluate("$.a[?@.b > 10]", data)

    assert [match.value for match in matches] == expected
    assert diagnostics.first_failure is not None


def test_missing_field_index_out_of_range_and_predicate_false_are_distinct():
    _, missing = evaluate("$.a.b", {"a": {}})
    assert missing.first_failure.failure == MISSING_FIELD

    _, out_of_range = evaluate("$.a[5]", {"a": [1]})
    assert out_of_range.first_failure.failure == INDEX_OUT_OF_RANGE
    assert "length 1" in out_of_range.first_failure.detail

    _, wrong_type = evaluate("$.a[0]", {"a": {"not": "a list"}})
    assert wrong_type.first_failure.failure == TYPE_MISMATCH

    _, predicate = evaluate("$.a[?@.x > 5]", {"a": [{"x": 1}]})
    assert predicate.first_failure.failure == PREDICATE_FALSE


def test_sample_values_are_hidden_by_default():
    data = {"secret": {"token": "hunter2"}}
    _, diagnostics = evaluate("$.secret.missing", data)

    assert diagnostics.first_failure is not None
    for event in diagnostics.iter_events():
        assert event.sample_input is None
        assert event.sample_output is None
    assert "hunter2" not in diagnostics.render()


def test_sample_values_are_available_on_request():
    data = {"secret": {"token": "hunter2"}}
    _, diagnostics = evaluate("$.secret.missing", data, include_values=True)

    samples = [
        event.sample_input
        for event in diagnostics.iter_events()
        if event.sample_input is not None
    ]
    assert any("hunter2" in sample for sample in samples)


def test_event_budget_truncates_without_affecting_results():
    data = {"a": [{"b": index} for index in range(10)]}
    matches, diagnostics = evaluate("$..b", data, max_events=4)

    assert len(matches) == 10
    assert diagnostics.truncated is True
    assert diagnostics.event_count <= 4
    assert "truncated" in diagnostics.truncation_note
    assert "truncated" in diagnostics.render()


class _CountingNode(JSONPath):
    """An extension node that counts how often it is evaluated."""

    def __init__(self):
        self.calls = 0

    def find(self, datum):
        self.calls += 1
        return [DatumInContext.wrap(datum)]


def test_diagnostics_never_reevaluate_nodes():
    counter = _CountingNode()
    expression = base_parse("$.a").child(counter)

    with QueryDiagnostics():
        matches = expression.find({"a": 1})

    assert len(matches) == 1
    # The counting node ran exactly once per datum:
    # diagnostics only observe the real evaluation chain.
    assert counter.calls == 1


class _ExplodingNode(JSONPath):
    def find(self, datum):
        raise ValueError("boom")


def test_exceptions_keep_completed_steps_and_propagate():
    expression = base_parse("$.a").child(_ExplodingNode())
    diagnostics = QueryDiagnostics()

    with pytest.raises(ValueError, match="boom"), diagnostics:
        expression.find({"a": 1})

    # The original exception is preserved, not converted to an empty match.
    assert isinstance(diagnostics.error, ValueError)
    # The steps completed before the failure are still recorded.
    completed = [
        event for event in diagnostics.iter_events() if event.output_count is not None
    ]
    assert any(event.node_type == "Fields" for event in completed)
    assert "ValueError" in diagnostics.render()


def test_render_shows_where_candidates_are_lost():
    _, diagnostics = evaluate(EXPRESSION, RESPONSE_RENAMED_FIELD)
    rendered = diagnostics.render()

    assert "Fields" in rendered
    assert "out=0" in rendered
    assert "FAILED missing_field" in rendered
