"""Tests for parameterized query templates."""

import concurrent.futures

import pytest

import jsonpath_ng
import jsonpath_ng.ext
from jsonpath_ng.exceptions import JSONPathParameterError
from jsonpath_ng.exceptions import JsonPathParserError
from jsonpath_ng.jsonpath import BoundParameter
from jsonpath_ng.jsonpath import Placeholder
from jsonpath_ng.template import ParameterizedJSONPath
from jsonpath_ng.template import parse_template

DOCUMENT = {
    "threshold": 72,
    "users": [
        {"name": "alice", "role": "admin", "score": 90},
        {"name": "bob", "role": "user", "score": 55},
        {"name": "carol", "role": "admin", "score": 72},
    ],
}


def values(template: ParameterizedJSONPath, parameters: dict) -> list:
    return [match.value for match in template.find(DOCUMENT, parameters)]


def test_template_is_parsed_once_and_reused():
    template = parse_template("users[?@.role == $role].name")

    assert template.parameter_names == ("role",)
    assert isinstance(template.expression, jsonpath_ng.Child)
    assert values(template, {"role": "admin"}) == ["alice", "carol"]
    assert values(template, {"role": "user"}) == ["bob"]
    # The parsed AST keeps placeholders and is never mutated by execution.
    assert str(template) == "users[?@.role == $role].name"
    assert template == parse_template("users[?@.role == $role].name")


def test_placeholder_nodes_are_stored_in_the_ast():
    template = parse_template("users[?@.score > $cut]")

    expression = template.expression
    filter_node = expression.right
    assert isinstance(filter_node.expressions[0].value, Placeholder)
    assert filter_node.expressions[0].value.name == "cut"


def test_each_binding_produces_an_independent_bound_ast():
    template = parse_template("users[?@.role == $role]")

    first = template.bind({"role": "admin"})
    second = template.bind({"role": "user"})

    bound_values = [
        node.right.expressions[0].value.value
        for node in (first, second)
        if isinstance(node, jsonpath_ng.Child)
        and isinstance(node.right.expressions[0].value, BoundParameter)
    ]
    assert bound_values == ["admin", "user"]
    # Rebinding does not change earlier bound trees.
    assert [match.value for match in first.find(DOCUMENT)] == [
        DOCUMENT["users"][0],
        DOCUMENT["users"][2],
    ]
    assert [match.value for match in second.find(DOCUMENT)] == [DOCUMENT["users"][1]]
    # The template itself still contains placeholders, not bound values.
    assert isinstance(template.expression.right.expressions[0].value, Placeholder)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, [{"v": None}]),
        (True, [{"v": True}]),
        (False, [{"v": False}]),
        (1, [{"v": 1}]),
        (0, [{"v": 0}]),
        (1.5, [{"v": 1.5}]),
        ("text", [{"v": "text"}]),
    ],
)
def test_parameter_types_are_preserved(value, expected):
    data = [
        {"v": None},
        {"v": True},
        {"v": False},
        {"v": 1},
        {"v": 0},
        {"v": 1.5},
        {"v": "text"},
    ]
    template = parse_template("$[?@.v == $value]")

    assert [match.value for match in template.find(data, {"value": value})] == expected


def test_boolean_is_not_coerced_to_number():
    data = [{"v": True}, {"v": 1}, {"v": False}, {"v": 0}]
    template = parse_template("$[?@.v == $value]")

    assert [m.value for m in template.find(data, {"value": True})] == [{"v": True}]
    assert [m.value for m in template.find(data, {"value": 1})] == [{"v": 1}]
    assert [m.value for m in template.find(data, {"value": False})] == [{"v": False}]
    assert [m.value for m in template.find(data, {"value": 0})] == [{"v": 0}]


def test_all_comparison_operators_accept_parameters():
    cases = [
        ("==", {"cut": 72}, ["carol"]),
        ("!=", {"cut": 72}, ["alice", "bob"]),
        (">", {"cut": 60}, ["alice", "carol"]),
        (">=", {"cut": 72}, ["alice", "carol"]),
        ("<", {"cut": 72}, ["bob"]),
        ("<=", {"cut": 55}, ["bob"]),
    ]
    for operator, parameters, expected in cases:
        template = parse_template(f"users[?@.score {operator} $cut].name")
        assert values(template, parameters) == expected


def test_regex_operator_takes_the_parameter_as_a_pattern():
    data = {"users": [{"n": "alice"}, {"n": "bob"}, {"n": "carol"}]}
    template = parse_template("users[?@.n =~ $pattern]")

    assert [m.value for m in template.find(data, {"pattern": "^a"})] == [{"n": "alice"}]
    assert [m.value for m in template.find(data, {"pattern": "o"})] == [
        {"n": "bob"},
        {"n": "carol"},
    ]


def test_missing_parameter_fails_before_document_access():
    template = parse_template("users[?@.role == $role]")

    with pytest.raises(JSONPathParameterError, match=r"Missing parameter"):
        template.find(DOCUMENT, {})


def test_unknown_parameter_fails_before_document_access():
    template = parse_template("users[?@.role == $role]")

    with pytest.raises(JSONPathParameterError, match=r"Unknown parameter"):
        template.find(DOCUMENT, {"role": "admin", "other": 1})


@pytest.mark.parametrize(
    "value",
    [
        [1, 2],
        {"a": 1},
        (1,),
        object(),
    ],
)
def test_unsupported_parameter_types_fail_before_document_access(value):
    template = parse_template("users[?@.role == $role]")

    with pytest.raises(JSONPathParameterError, match=r"unsupported value type"):
        template.find(DOCUMENT, {"role": value})


def test_duplicate_declaration_uses_one_binding():
    template = parse_template("users[?@.role == $role && @.name != $role]")

    assert template.parameter_names == ("role",)
    assert values(template, {"role": "admin"}) == [
        {"name": "alice", "role": "admin", "score": 90},
        {"name": "carol", "role": "admin", "score": 72},
    ]
    # A duplicate declaration does not create a second parameter.
    with pytest.raises(JSONPathParameterError, match=r"Unknown parameter"):
        template.find(DOCUMENT, {"role": "admin", "role2": "x"})


def test_template_without_parameters_is_rejected():
    with pytest.raises(JSONPathParameterError, match=r"no parameters"):
        parse_template("users[*].name")


@pytest.mark.parametrize(
    "query",
    [
        "$.$field",
        "users[$field]",
        "users[?@[$field]]",
        "users[?@.a $op @.b]",
        "users[?@.a == $value + 1]",
        "users[?$flag]",
        "$field",
    ],
)
def test_placeholders_are_rejected_outside_comparison_values(query):
    with pytest.raises(JsonPathParserError):
        jsonpath_ng.ext.parse(query)


def test_placeholders_are_rejected_by_the_core_parser():
    with pytest.raises(JsonPathParserError):
        jsonpath_ng.parse("$field")


@pytest.mark.parametrize(
    "name",
    [
        "admin'] || @.role == '",
        'a"b',
        "a\\b",
        "a(b)c",
        "a.b*c",
        "@.role",
        "$.users",
        "==",
        "true",
        "null",
        "1",
        "1 == 1 || true",
        "[?@.x]",
    ],
)
def test_expression_like_strings_are_plain_values(name):
    data = {"users": [{"n": name}, {"n": "admin"}]}
    template = parse_template("users[?@.n == $name]")

    assert [m.value for m in template.find(data, {"name": name})] == [{"n": name}]
    # The string is never interpreted as a query, so other members stay hidden.
    assert [m.value for m in template.find(data, {"name": "admin"})] == [
        {"n": "admin"}
    ]


def test_nested_filters_share_one_binding():
    data = {
        "groups": [
            {"members": [{"role": "admin", "n": 1}, {"role": "user", "n": 2}]},
            {"members": [{"role": "user", "n": 3}]},
        ]
    }
    template = parse_template("groups[?@.members[?@.role == $role]]")

    assert [m.value for m in template.find(data, {"role": "admin"})] == [
        data["groups"][0]
    ]
    assert [m.value for m in template.find(data, {"role": "user"})] == [
        data["groups"][0],
        data["groups"][1],
    ]


def test_root_and_current_node_keep_their_meaning():
    template = parse_template("users[?@.score > $cut && $.threshold == $cut]")

    assert values(template, {"cut": 72}) == [
        {"name": "alice", "role": "admin", "score": 90}
    ]
    assert values(template, {"cut": 1}) == []


def test_concurrent_executions_are_isolated():
    template = parse_template("users[?@.role == $role && @.score >= $cut].name")
    cases = [
        ({"role": "admin", "cut": 80}, ["alice"]),
        ({"role": "user", "cut": 50}, ["bob"]),
        ({"role": "admin", "cut": 1}, ["alice", "carol"]),
        ({"role": "user", "cut": 100}, []),
    ] * 25

    def run(case):
        parameters, expected = case
        assert values(template, parameters) == expected
        return parameters

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, cases))

    # The template is unchanged after the concurrent executions.
    assert template.parameter_names == ("role", "cut")
    assert str(template) == "users[?@.role == $role && @.score >= $cut].name"


def test_parse_and_find_still_work_without_templates():
    expression = jsonpath_ng.ext.parse("users[?@.role == 'admin'].name")

    assert [m.value for m in expression.find(DOCUMENT)] == ["alice", "carol"]
    assert [m.value for m in jsonpath_ng.parse("users[*].name").find(DOCUMENT)] == [
        "alice",
        "bob",
        "carol",
    ]


class _ExplodingDocument:
    """A document that fails loudly on any access."""

    def __getattr__(self, name):
        raise AssertionError(f"the document was accessed via {name!r}")

    def __getitem__(self, key):
        raise AssertionError(f"the document was accessed via {key!r}")


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"role": "admin", "extra": 1},
        {"role": ["not", "a", "scalar"]},
        {"role": {"nested": "object"}},
    ],
)
def test_invalid_bindings_fail_before_the_document_is_accessed(parameters):
    template = parse_template("users[?@.role == $role]")

    with pytest.raises(JSONPathParameterError):
        template.find(_ExplodingDocument(), parameters)
