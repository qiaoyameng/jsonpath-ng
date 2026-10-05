import json
import sqlite3

import pytest

from jsonpath_ng import SQLiteJSONPathError, compile_sqlite
from jsonpath_ng.ext import parse


DOCUMENTS = [
    {
        "id": 1,
        "items": [
            {"n": 2, "ok": True, "m": None, "tags": ["beta", "alpha"]},
            {"n": 8, "ok": False, "m": 3, "tags": ["gamma"]},
            {"n": 5, "ok": False, "m": None, "tags": []},
        ],
        "nested": [[10, 11], [12], []],
        "a'b\".[": {"key.with.dots": "special"},
        "people": {"adult": {"age": 30, "name": "Bo"}, "elder": {"age": 80, "name": "Cy"}},
    },
    {
        "id": 2,
        "items": [
            {"n": 1, "ok": True},
            {"n": 9, "ok": False},
        ],
        "nested": [[20], [21, 22]],
        "a'b\".[": {"key.with.dots": "other"},
        "people": {"child": {"age": 10, "name": "Ada"}, "adult": {"age": 30, "name": "Bo"}},
    },
]


@pytest.fixture
def connection():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE documents (body TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO documents (body) VALUES (?)",
        [(json.dumps(document),) for document in DOCUMENTS],
    )
    return connection


@pytest.mark.parametrize(
    "expression",
    [
        "$",
        "id",
        "items[0].n",
        "items[2].tags[0]",
        "items[*].n",
        "nested[*][*]",
        "items[*].tags[*]",
        '$["a\'b\\".["]',
        '$["a\'b\\".["]["key.with.dots"]',
        "items[?n > 1].n",
        "items[?ok == true].n",
        "items[?m == null].n",
        "items[?m != null].m",
        "items[?@.n == 5]",
        "items[?@.n >= 2].n",
        "nested[*][?@ >= 11]",
        "people[?age >= 18].name",
    ],
)
def test_sqlite_matches_find(connection, expression):
    compiled = compile_sqlite(expression, "documents", "body")
    rows = connection.execute(compiled.sql, compiled.parameters).fetchall()

    expected = [
        (document_index, match.value, _normalized_path(match.full_path))
        for document_index, document in enumerate(DOCUMENTS)
        for match in parse(expression).find(document)
    ]
    actual = [
        (
            source_rowid - 1,
            json.loads(match_value),
            match_path,
        )
        for source_rowid, match_value, match_path in rows
    ]

    assert actual == expected


@pytest.mark.parametrize(
    ("expression", "values"),
    [
        ("missing.path", []),
        ("items[9].n", []),
        ("items[*].missing", []),
        ("items[?n > 100].n", []),
        ("items[?n == null]", []),
        ("items[?n == true]", []),
        ("items[?ok == 1]", []),
        ("nested[*][?@ > 100][0]", []),
    ],
)
def test_empty_matches(connection, expression, values):
    compiled = compile_sqlite(expression, "documents", "body")
    rows = connection.execute(compiled.sql, compiled.parameters).fetchall()
    assert rows == values


def test_array_order_is_explicit(connection):
    document = {"values": [{"sort": index, "n": index} for index in range(12)]}
    connection.execute("DELETE FROM documents")
    connection.execute(
        "INSERT INTO documents (body) VALUES (?)",
        (json.dumps(document),),
    )

    compiled = compile_sqlite("values[?n >= 0].n", "documents", "body")
    rows = connection.execute(compiled.sql, compiled.parameters).fetchall()

    assert [json.loads(row[1]) for row in rows] == list(range(12))
    assert [row[2] for row in rows] == [
        f'$["values"][{index}]["n"]' for index in range(12)
    ]


def test_scalar_types_are_distinct(connection):
    cases = [
        ("items[?n == 2]", 2),
        ("items[?ok == true]", True),
        ("items[?m == null]", None),
    ]
    for expression, expected in cases:
        compiled = compile_sqlite(expression, "documents", "body")
        row = connection.execute(compiled.sql, compiled.parameters).fetchone()
        if expected is None:
            value = json.loads(row[1])["m"]
        elif isinstance(expected, bool):
            value = json.loads(row[1])["ok"]
        else:
            value = json.loads(row[1])["n"]
        assert value == expected
        assert type(value) is type(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "items..n",
        "store.*",
        "items[0,1].n",
        "items[0:2].n",
        "items[?n]",
        "items[?n =~ '2']",
        "items[?n + 1 == 3]",
        "items | id",
        "items where n",
        "items.`len`",
    ],
)
def test_unsupported_expressions_are_rejected(expression):
    with pytest.raises(SQLiteJSONPathError):
        compile_sqlite(expression, "documents", "body")


def test_special_field_names_are_parameterized():
    field = '["key\' --"]'
    compiled = compile_sqlite(field, "documents", "body")
    assert field not in compiled.sql
    assert "key' --" in compiled.parameters


@pytest.mark.parametrize("name", ['name"', "name; DROP TABLE x", ""])
def test_identifiers_are_validated(name):
    with pytest.raises(SQLiteJSONPathError):
        compile_sqlite("id", name, "body")


def test_qualified_table_name_is_supported(connection):
    compiled = compile_sqlite("id", "main.documents", "body")
    rows = connection.execute(compiled.sql, compiled.parameters).fetchall()
    assert [json.loads(row[1]) for row in rows] == [1, 2]


def _normalized_path(path) -> str:
    segments = _path_segments(path)
    return "$" + "".join(
        f'[{json.dumps(segment, ensure_ascii=False)}]' if isinstance(segment, str)
        else f"[{segment}]"
        for segment in segments
    )


def _path_segments(path):
    from jsonpath_ng.jsonpath import Child, Fields, Index, Root, Slice

    if isinstance(path, Root):
        return ()
    if isinstance(path, Child):
        return _path_segments(path.left) + _path_segments(path.right)
    if isinstance(path, Fields):
        return path.fields
    if isinstance(path, Index):
        return path.indices
    if isinstance(path, Slice):
        return ()
    raise AssertionError(f"Unexpected path node: {path!r}")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("items[?m == null]", "null"),
        ('items[?m == "null"]', "string"),
        ("items[?m == true]", "boolean"),
        ("items[?m == 1]", "number"),
    ],
)
def test_parser_preserves_null_literal_kind(expression, expected):
    filter_node = parse(expression).right
    value = filter_node.expressions[0].value
    kinds = {
        None: "null",
        "null": "string",
    }
    mapped = (
        "boolean"
        if value is True
        else "number"
        if value == 1
        else kinds[value]
    )
    assert mapped == expected
