import re

import pytest

from jsonpath_ng import Fields
from jsonpath_ng import find_with_diagnostics
from jsonpath_ng import parse
from jsonpath_ng import diagnostics
from jsonpath_ng.diagnostics import (
    INDEX_OUT_OF_RANGE,
    MISSING_FIELD,
    PREDICATE_FALSE,
    TYPE_MISMATCH,
)
from jsonpath_ng.ext import parse as ext_parse


# A single small response in two versions: the first version still has the
# `book` field and numeric prices, the second version renamed `book` to
# `items` and changed `price` to a string `cost`.
VERSION_ONE = {
    "store": {
        "book": [
            {"category": "reference", "price": 8, "title": "alpha"},
            {"category": "fiction", "price": 20, "title": "beta"},
        ]
    }
}

VERSION_TWO = {
    "store": {
        "items": [
            {"kind": "reference", "cost": "8", "name": "alpha"},
        ]
    }
}

QUERY = "$.store.book[?@.price < 10].title"


def test_first_failure_differs_between_response_versions():
    """The same expression loses its candidates at different nodes."""

    path = ext_parse(QUERY)

    matches_one, report_one = find_with_diagnostics(path, VERSION_ONE)
    assert [match.value for match in matches_one] == ["alpha"]
    first_one = report_one.first_failure()
    assert first_one is not None
    # In version one only the second book fails the predicate.
    assert first_one.node == "@.price < 10"
    assert first_one.path == "store.book.[1]"
    assert first_one.reason == PREDICATE_FALSE

    matches_two, report_two = find_with_diagnostics(path, VERSION_TWO)
    assert matches_two == []
    first_two = report_two.first_failure()
    assert first_two is not None
    # In version two the field was renamed, so `book` is missing first.
    assert first_two.node == "book"
    assert first_two.path == "store"
    assert first_two.reason == MISSING_FIELD


def test_results_are_identical_with_and_without_diagnostics():
    path = ext_parse(QUERY)
    for data in (VERSION_ONE, VERSION_TWO):
        plain = [match.value for match in path.find(data)]
        matches, report = find_with_diagnostics(path, data)
        assert [match.value for match in matches] == plain
        assert report.error is None


class CountingFields(Fields):
    """A real AST node that counts how often the evaluation chain runs it."""

    def __init__(self, *fields):
        super().__init__(*fields)
        self.calls = 0

    def find(self, datum):
        self.calls += 1
        return super().find(datum)


def test_diagnostics_call_each_node_exactly_once():
    """Diagnostics observe the real chain; they never re-run it."""

    data = {"store": {"book": [{"price": 5, "title": "alpha"}]}}

    plain_node = CountingFields("price")
    ext_parse("$.store.book[0]").child(plain_node).find(data)
    assert plain_node.calls == 1

    diagnostic_node = CountingFields("price")
    path = ext_parse("$.store.book[0]").child(diagnostic_node)
    matches, report = find_with_diagnostics(path, data)
    assert diagnostic_node.calls == 1
    assert [match.value for match in matches] == [5]
    assert len(report.events) > 0


@pytest.mark.parametrize(
    "expression, data, node, reason",
    (
        ("$.store.missing", VERSION_ONE, "missing", MISSING_FIELD),
        ("$.store.book[9]", VERSION_ONE, "[9]", INDEX_OUT_OF_RANGE),
        (
            "$.store.book.price",
            VERSION_ONE,
            "price",
            TYPE_MISMATCH,
        ),
        (
            "$.store.items[?@.cost < 10]",
            VERSION_TWO,
            "@.cost < 10",
            TYPE_MISMATCH,
        ),
        (
            "$.store.book[?@.price > 100]",
            VERSION_ONE,
            "@.price > 100",
            PREDICATE_FALSE,
        ),
    ),
)
def test_reasons_are_reported_separately(expression, data, node, reason):
    matches, report = find_with_diagnostics(ext_parse(expression), data)
    assert matches == []
    first = report.first_failure()
    assert first is not None
    assert first.node == node
    assert first.reason == reason
    assert first.input_count > 0
    assert first.output_count == 0


def test_named_operators_report_type_mismatch():
    _, len_report = find_with_diagnostics(
        ext_parse("$.a.`len`"), {"a": 5}
    )
    assert len_report.first_failure().reason == TYPE_MISMATCH

    _, keys_report = find_with_diagnostics(
        ext_parse("$.a.`keys`"), {"a": [1, 2]}
    )
    assert keys_report.first_failure().reason == TYPE_MISMATCH


def test_report_supports_iteration_and_length():
    _, report = find_with_diagnostics(
        ext_parse("$.store.book[0].price"), VERSION_ONE
    )
    assert len(report) == len(report.events)
    assert list(report) == report.events


def test_reason_codes_are_distinct():
    assert (
        len(
            {
                MISSING_FIELD,
                INDEX_OUT_OF_RANGE,
                TYPE_MISMATCH,
                PREDICATE_FALSE,
            }
        )
        == 4
    )


def test_samples_are_hidden_by_default():
    _, report = find_with_diagnostics(
        ext_parse("$.store.missing"),
        VERSION_ONE,
    )
    first = report.first_failure()
    assert first is not None
    assert first.sample is diagnostics._NO_SAMPLE
    assert "sample=" not in str(first)


def test_samples_can_be_included():
    _, report = find_with_diagnostics(
        ext_parse("$.store.missing"),
        VERSION_ONE,
        include_samples=True,
    )
    first = report.first_failure()
    assert first is not None
    assert first.sample == {"book": VERSION_ONE["store"]["book"]}
    assert "sample=" in str(first)


def test_event_budget_truncates_without_changing_results():
    path = ext_parse(QUERY)
    plain = [match.value for match in path.find(VERSION_ONE)]

    matches, report = find_with_diagnostics(path, VERSION_ONE, event_budget=2)
    assert [match.value for match in matches] == plain
    assert len(report.events) == 2
    assert report.truncated is True
    assert report.event_budget == 2


def test_zero_event_budget_records_nothing():
    path = ext_parse(QUERY)
    matches, report = find_with_diagnostics(path, VERSION_ONE, event_budget=0)
    assert [match.value for match in matches] == ["alpha"]
    assert report.events == []
    assert report.truncated is True


def test_successful_query_has_no_first_failure():
    path = ext_parse("$.store.book[0].price")
    matches, report = find_with_diagnostics(path, VERSION_ONE)
    assert [match.value for match in matches] == [8]
    assert report.first_failure() is None
    assert report.truncated is False


def test_exception_keeps_completed_steps_and_original_type():
    """A raised exception propagates unchanged, not as an empty match."""

    path = ext_parse("$.items[?@.name =~ '(']")
    data = {"items": [{"name": "alpha"}]}

    with pytest.raises(re.PatternError) as excinfo:
        find_with_diagnostics(path, data)

    assert type(excinfo.value) is re.PatternError
    report = excinfo.value.jsonpath_diagnostics
    assert isinstance(report, diagnostics.DiagnosticReport)
    assert report.error is excinfo.value
    # Steps that completed before the exception are retained.
    assert len(report.events) > 0
    assert any(event.node == "items" for event in report.events)


def test_collector_is_reset_after_success_and_failure():
    path = ext_parse("$.store.book[0].price")
    find_with_diagnostics(path, VERSION_ONE)
    assert diagnostics.get_collector() is None

    with pytest.raises(re.PatternError):
        find_with_diagnostics(ext_parse("$[?@.x =~ '(']"), [{"x": "a"}])
    assert diagnostics.get_collector() is None


def test_nested_filter_keeps_its_path():
    data = {
        "teams": [
            {"name": "alpha", "members": [{"age": 30}, {"age": 17}]}
        ]
    }
    path = ext_parse("$.teams[?@.members[?@.age >= 18]]")
    matches, report = find_with_diagnostics(path, data)
    assert len(matches) == 1

    first = report.first_failure()
    assert first is not None
    assert first.reason == PREDICATE_FALSE
    assert first.path == "teams.[0].members.[1]"


def test_root_reference_keeps_its_path():
    data = {
        "store": {
            "book": [{"price": 8}, {"price": 2}],
            "threshold": 5,
        }
    }
    path = ext_parse("$.store.book[?@.price < $.store.threshold]")
    matches, report = find_with_diagnostics(path, data)
    assert [match.value for match in matches] == [{"price": 2}]

    first = report.first_failure()
    assert first is not None
    assert first.node == "@.price < $.store.threshold"
    assert first.path == "store.book.[0]"
    assert first.reason == PREDICATE_FALSE


def test_base_parser_queries_are_diagnosed():
    path = parse("$.store.book[0].price")
    matches, report = find_with_diagnostics(path, VERSION_ONE)
    assert [match.value for match in matches] == [8]
    assert [event.node for event in report.events] == [
        "$",
        "store",
        "book",
        "[0]",
        "price",
    ]


def test_where_predicate_is_reported():
    path = parse("store.book where price")
    data = {"store": {"book": {"title": "alpha"}}}
    matches, report = find_with_diagnostics(path, data)
    assert matches == []
    where_events = [
        event
        for event in report.events
        if event.node == "book where price" and event.reason == PREDICATE_FALSE
    ]
    assert len(where_events) == 1
    event = where_events[0]
    assert event.input_count == 1
    assert event.output_count == 0
    assert report.first_failure() is not None
