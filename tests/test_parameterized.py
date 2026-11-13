"""
Tests for parameterized queries.

A template is parsed once and executed many times. Parameters may only occupy
a filter comparison's value position, must preserve their JSON type, and are
validated before the queried document is ever touched.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

import jsonpath_ng.ext
from jsonpath_ng.exceptions import DuplicateParameterError
from jsonpath_ng.exceptions import JsonPathLexerError
from jsonpath_ng.exceptions import JsonPathParserError
from jsonpath_ng.exceptions import MissingParameterError
from jsonpath_ng.exceptions import UnsupportedParameterTypeError
from jsonpath_ng.ext.filter import Expression
from jsonpath_ng.ext.filter import Filter
from jsonpath_ng.ext.filter import Parameter
from jsonpath_ng.ext.parser import ExtendedJsonPathLexer
from jsonpath_ng.ext.template import ParameterizedQuery
from jsonpath_ng.ext.template import parse_template

DOC = {
    "items": [
        {"v": 1, "tag": "a"},
        {"v": "x", "tag": "a"},
        {"v": 1, "tag": "b"},
        {"v": None, "tag": "b"},
        {"v": True, "tag": "c"},
    ]
}


@pytest.mark.parametrize(
    "text, expected_value",
    [
        (":name", "name"),
        (":a-b_c@1", "a-b_c@1"),
        (":''", ""),
        (r":'a.b[0] == \"x\" \\'", 'a.b[0] == "x" \\'),
        (":'it\\'s'", "it's"),
        (':"a\\"b"', 'a"b'),
    ],
)
def test_placeholder_lexing(text, expected_value):
    tokens = list(ExtendedJsonPathLexer().tokenize(text))
    assert len(tokens) == 1
    assert tokens[0].type == "PLACEHOLDER"
    assert tokens[0].value == expected_value


def test_a_colon_outside_a_placeholder_is_still_a_slice_literal():
    tokens = list(ExtendedJsonPathLexer().tokenize("[:3]"))
    assert [(token.value, token.type) for token in tokens] == [
        ("[", "["),
        (":", ":"),
        (3, "NUMBER"),
        ("]", "]"),
    ]


def test_an_unterminated_quoted_placeholder_is_a_lexer_error():
    with pytest.raises(JsonPathLexerError):
        list(ExtendedJsonPathLexer().tokenize(":'unterminated"))


def test_template_parses_once_and_reuses_the_ast():
    query = parse_template("$.items[?(@.v == :p)]")

    assert isinstance(query, ParameterizedQuery)
    assert query.parameters == ("p",)
    assert isinstance(query.path.right, Filter)
    expression = query.path.right.expressions[0]
    assert isinstance(expression, Expression)
    assert isinstance(expression.value, Parameter)
    assert expression.value.name == "p"

    # Repeated executions reuse the single parsed AST.
    first = query.find(DOC, {"p": 1})
    second = query.find(DOC, {"p": 1})
    assert first == second


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(1, [{"v": 1, "tag": "a"}, {"v": 1, "tag": "b"}], id="int"),
        pytest.param(1.0, [{"v": 1, "tag": "a"}, {"v": 1, "tag": "b"}], id="float"),
        pytest.param("x", [{"v": "x", "tag": "a"}], id="string"),
        pytest.param(None, [{"v": None, "tag": "b"}], id="null"),
        pytest.param(True, [{"v": True, "tag": "c"}], id="true"),
        pytest.param(False, [], id="false"),
    ],
)
def test_parameter_types_are_preserved(value, expected):
    query = parse_template("$.items[?(@.v == :p)]")
    assert [match.value for match in query.find(DOC, {"p": value})] == expected


def test_boolean_is_not_coerced_to_a_number():
    query = parse_template("$.items[?(@.v == :p)]")
    # RFC 9535: `true` must never compare equal to the number 1.
    assert [match.value for match in query.find(DOC, {"p": 1})] == [
        {"v": 1, "tag": "a"},
        {"v": 1, "tag": "b"},
    ]
    assert [match.value for match in query.find(DOC, {"p": True})] == [
        {"v": True, "tag": "c"}
    ]


@pytest.mark.parametrize("value", [[1, 2], {"k": 1}, (1, 2), object()])
def test_unsupported_parameter_types_are_rejected(value):
    query = parse_template("$.items[?(@.v == :p)]")
    with pytest.raises(UnsupportedParameterTypeError):
        query.find(DOC, {"p": value})


def test_missing_parameter_is_reported():
    query = parse_template("$.items[?(@.v == :p)]")
    with pytest.raises(MissingParameterError, match="'p'"):
        query.find(DOC, {})


def test_multiple_missing_parameters_are_all_reported():
    query = parse_template("$.items[?(@.v == :p && @.tag == :q)]")
    assert set(query.parameters) == {"p", "q"}
    with pytest.raises(MissingParameterError) as exc_info:
        query.find(DOC, {})
    message = str(exc_info.value)
    assert "'p'" in message
    assert "'q'" in message


def test_binding_errors_happen_before_the_document_is_touched():
    class ExplodingDocument:
        def __getattribute__(self, name):
            raise AssertionError("the document must not be accessed")

    query = parse_template("$.items[?(@.v == :p)]")

    with pytest.raises(MissingParameterError):
        query.find(ExplodingDocument(), {})
    with pytest.raises(UnsupportedParameterTypeError):
        query.find(ExplodingDocument(), {"p": ["not", "json"]})


def test_bindings_must_be_a_dict():
    query = parse_template("$.items[?(@.v == :p)]")
    with pytest.raises(TypeError):
        query.find(DOC, [("p", 1)])


def test_extra_parameters_are_ignored():
    query = parse_template("$.items[?(@.v == :p)]")
    assert [match.value for match in query.find(DOC, {"p": "x", "unused": 9})] == [
        {"v": "x", "tag": "a"}
    ]


def test_duplicate_parameter_declaration_fails_at_compile_time():
    with pytest.raises(DuplicateParameterError, match="'p'"):
        parse_template("$.items[?(@.v == :p && @.tag == :p)]")


@pytest.mark.parametrize(
    "template",
    [
        "$.:p",
        "$[:p]",
        "$.x[?(:p)]",
        "$.x[?(@.a + :p)]",
        "$.x[?(:p == @.a)]",
        "$[:p].x",
    ],
)
def test_parameters_cannot_stand_for_paths_or_operators(template):
    with pytest.raises(JsonPathParserError):
        parse_template(template)


def test_parameter_value_is_never_interpreted_as_a_path():
    query = parse_template("$.items[?(@.v == :p)]")
    document = {"items": [{"v": "$.evil"}, {"v": "@.x"}, {"v": "a == b"}]}
    for value in ("$.evil", "@.x", "a == b"):
        assert [match.value for match in query.find(document, {"p": value})] == [
            {"v": value}
        ]


def test_quoted_placeholder_accepts_special_characters_in_the_name():
    # The name contains quotes, a backslash, brackets, dots, and operators.
    # It must behave as an opaque dictionary key rather than syntax.
    name = """a.b[0] == "x" \\"""
    query = parse_template(r"$.items[?(@.v == :'a.b[0] == \"x\" \\')]")
    assert query.parameters == (name,)

    document = {"items": [{"v": "match"}, {"v": "other"}]}
    assert [match.value for match in query.find(document, {name: "match"})] == [
        {"v": "match"}
    ]
    with pytest.raises(MissingParameterError):
        query.find(document, {"different key": "match"})


def test_nested_filters_share_the_same_binding():
    query = parse_template(
        "$.groups[?(@.items[?(@.v == :p)] && @.items[?(@.w == :w)])]"
    )
    document = {
        "groups": [
            {"items": [{"v": 1, "w": "z"}]},
            {"items": [{"v": 1, "w": "q"}]},
            {"items": [{"v": 9, "w": "z"}]},
        ]
    }
    assert [
        match.value for match in query.find(document, {"p": 1, "w": "z"})
    ] == [{"items": [{"v": 1, "w": "z"}]}]


def test_root_and_current_node_keep_their_ordinary_meaning():
    query = parse_template(
        "$.groups[?(@.items[?(@.v == :p)] && $.flag == @.flag)]"
    )
    document = {
        "flag": True,
        "groups": [
            {"flag": True, "items": [{"v": 1}]},
            {"flag": False, "items": [{"v": 1}]},
            {"flag": True, "items": [{"v": 9}]},
        ],
    }
    assert [match.value for match in query.find(document, {"p": 1})] == [
        {"flag": True, "items": [{"v": 1}]}
    ]


def test_concurrent_executions_are_isolated():
    query = parse_template("$.items[?(@.v == :p)]")
    document = {
        "items": [{"v": "a"}, {"v": "b"}, {"v": "a"}, {"v": "c"}, {"v": "b"}]
    }

    def execute(value):
        return [match.value for match in query.find(document, {"p": value})]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(execute, ["a", "b", "c"] * 100))

    expected = {
        "a": [{"v": "a"}, {"v": "a"}],
        "b": [{"v": "b"}, {"v": "b"}],
        "c": [{"v": "c"}],
    }
    for index, value in enumerate(["a", "b", "c"] * 100):
        assert results[index] == expected[value]

    # After all threads finish, no binding leaks into a bare parameter lookup.
    with pytest.raises(MissingParameterError):
        Parameter("p").find(document)


def test_template_renders_back_to_the_placeholder_syntax():
    query = parse_template("$.items[?(@.v == :p)]")
    assert str(query) == "$.items[?@.v == :p]"
    reparsed = jsonpath_ng.ext.parse(str(query))
    assert reparsed == query.path


def test_ordinary_parse_and_find_remain_unchanged():
    path = jsonpath_ng.ext.parse("$.items[?(@.v == 1)]")
    assert [match.value for match in path.find(DOC)] == [
        {"v": 1, "tag": "a"},
        {"v": 1, "tag": "b"},
    ]
    core_path = jsonpath_ng.parse("$.items")
    assert [match.value for match in core_path.find(DOC)] == [DOC["items"]]


def test_parameter_works_with_the_regex_operator():
    query = parse_template("$.items[?(@.tag =~ :p)]")
    document = {"items": [{"tag": "foobar"}, {"tag": "baz"}, {"tag": "afoot"}]}
    assert [match.value for match in query.find(document, {"p": "^foo"})] == [
        {"tag": "foobar"}
    ]


def test_descendant_filters_share_bindings():
    query = parse_template("$..items[?(@.v == :p)]")
    document = {
        "items": [{"v": 1}, {"v": 2}],
        "nested": {"items": [{"v": 1}]},
    }
    assert [match.value for match in query.find(document, {"p": 1})] == [
        {"v": 1},
        {"v": 1},
    ]
