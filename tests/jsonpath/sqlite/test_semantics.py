"""Verify boolean/number and missing/null query semantics explicitly."""

from __future__ import annotations

import pytest

from jsonpath_ng.ext.parser import parse
from jsonpath_ng.ext.sqlite import compile_jsonpath

from .helpers import decode_value
from .helpers import make_connection


# Each element exercises a different JSON type or absence of ``v``.
VALUES_DOCUMENT = {
    "arr": [
        {"v": None},
        {},
        {"v": 0},
        {"v": False},
        {"v": ""},
        {"v": "0"},
        {"v": True},
        {"v": 1},
        {"v": 0.0},
    ]
}


def values_for(query: str) -> list[object]:
    connection = make_connection([VALUES_DOCUMENT])
    compiled = compile_jsonpath(parse(query))
    return [
        decode_value(value, type_name)
        for value, _path, type_name in compiled.execute(connection)
    ]


@pytest.mark.parametrize(
    "query, expected",
    [
        # Booleans never compare equal to numbers, and vice versa.
        ("$.arr[?@.v == true].v", [True]),
        ("$.arr[?@.v == false].v", [False]),
        ("$.arr[?@.v == 1].v", [1]),
        ("$.arr[?@.v == 0].v", [0, 0.0]),
        # Strings never compare equal to numbers.
        ("$.arr[?@.v == '0'].v", ["0"]),
        ("$.arr[?@.v == ''].v", [""]),
        # Only explicit null equals null; a missing member does not.
        ("$.arr[?@.v == null].v", [None]),
        # != null matches every present member, including null only excluded;
        # the missing member does match because nothing equals null.
        (
            "$.arr[?@.v != null].v",
            [0, False, "", "0", True, 1, 0.0],
        ),
        # RFC 9535: != is the negation of ==, so missing/null/other-type
        # values all match an inequality against a number.
        ("$.arr[?@.v != 1].v", [None, 0, False, "", "0", True, 0.0]),
        # Ordering comparisons only match numbers; booleans, strings, null,
        # and missing members do not.
        ("$.arr[?@.v < 1].v", [0, 0.0]),
        ("$.arr[?@.v >= 0].v", [0, 1, 0.0]),
        ("$.arr[?@.v > 0].v", [1]),
    ],
)
def test_scalar_filter_semantics(query, expected):
    assert values_for(query) == expected


def test_existence_distinguishes_null_from_missing():
    present = values_for("$.arr[?@.v].v")
    absent = values_for("$.arr[?!@.v].v")

    # Existence matches every present value, including null and false.
    assert present == [None, 0, False, "", "0", True, 1, 0.0]
    # Negated existence only matches the element with no ``v`` member.
    assert absent == []
    assert [
        element for element in VALUES_DOCUMENT["arr"] if "v" not in element
    ] == [{}]


def test_returned_types_distinguish_booleans_and_numbers():
    connection = make_connection([{"a": True, "b": 1, "c": False, "d": 0}])
    compiled = compile_jsonpath(parse("$[*]"))

    rows = compiled.execute(connection)
    by_path = {path: type_name for _value, path, type_name in rows}

    assert by_path['$["a"]'] == "true"
    assert by_path['$["b"]'] == "integer"
    assert by_path['$["c"]'] == "false"
    assert by_path['$["d"]'] == "integer"


def test_special_field_names_are_data_only():
    document = {"a.b": 1, 'q"r': 2, "p%": 3, "a*b": 4}
    connection = make_connection([document])

    for field, expected in document.items():
        compiled = compile_jsonpath(parse(f"$[{field!r}]"))
        rows = compiled.execute(connection)
        assert rows == (
            [(expected, f'$[{_json_path_quote(field)}]', "integer")]
        )


def _json_path_quote(field: str) -> str:
    escaped = (
        field.replace("\\", "\\\\").replace('"', '\\"')
        .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    )
    return f'"{escaped}"'


def test_special_field_names_in_filter_targets():
    document = {
        "arr": [
            {"a.b": 1},
            {"a.b": 2},
            {'q"r': 1},
            {"it's": 1},
            {"back\\slash": 1},
        ]
    }
    connection = make_connection([document])

    cases = [
        ("$.arr[?@['a.b'] == 1]", [document["arr"][0]]),
        ('$.arr[?@[\'q"r\'] == 1]', [document["arr"][2]]),
        ("$.arr[?@['it\\'s'] == 1]", [document["arr"][3]]),
        ("$.arr[?@['back\\\\slash'] == 1]", [document["arr"][4]]),
    ]
    for query, expected in cases:
        compiled = compile_jsonpath(parse(query))
        rows = compiled.execute(connection)
        values = [
            decode_value(value, type_name) for value, _path, type_name in rows
        ]
        assert values == expected, query


def test_string_literals_with_quotes_and_backslashes():
    document = {
        "arr": [
            {"s": "it's"},
            {"s": 'say "hi"'},
            {"s": "back\\slash str"},
        ]
    }
    connection = make_connection([document])

    cases = [
        ("$.arr[?@.s == \"it's\"]", [document["arr"][0]]),
        ("$.arr[?@.s == 'say \"hi\"']", [document["arr"][1]]),
        ("$.arr[?@.s == 'back\\\\slash str']", [document["arr"][2]]),
    ]
    for query, expected in cases:
        compiled = compile_jsonpath(parse(query))
        rows = compiled.execute(connection)
        values = [
            decode_value(value, type_name) for value, _path, type_name in rows
        ]
        assert values == expected, query
