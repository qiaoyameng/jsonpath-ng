"""
Cross-check the SQLite JSON1 compiler against ``JSONPath.find()``.

These tests use only the standard-library ``sqlite3`` in-memory database:
there is no database server and no ORM.  Every supported query is compiled to
SQL plus bound parameters, executed against the same documents that ``find()``
runs against, and compared node-by-node (value and normalized path).
"""

from __future__ import annotations

import json
import sqlite3
import typing

import pytest

from jsonpath_ng import Child
from jsonpath_ng import Fields
from jsonpath_ng import Index
from jsonpath_ng.exceptions import JSON1CompileError
from jsonpath_ng.ext.json1 import CompiledJSON1Query
from jsonpath_ng.ext.json1 import _substitute_relation
from jsonpath_ng.ext.json1 import compile_json1
from jsonpath_ng.ext.json1 import render_canonical_path
from jsonpath_ng.ext.parser import parse


# Documents deliberately combine nested arrays, repeated values, booleans,
# numbers, explicit nulls, missing members and punctuation-heavy field names.
DOCUMENTS: list[typing.Any] = [
    {
        "store": {
            "book": [
                {
                    "cat": "fiction",
                    "price": 8,
                    "active": True,
                    "tags": ["a", "b", "a"],
                    "meta": {"n": 1},
                },
                {
                    "cat": "fiction",
                    "price": 12,
                    "active": False,
                    "tags": [],
                    "meta": {"n": 2},
                },
                {
                    "cat": "ref",
                    "price": 5,
                    "active": True,
                    "tags": ["c"],
                    "meta": None,
                },
                {
                    "cat": "ref",
                    "price": None,
                    "active": True,
                    "tags": ["d"],
                    "meta": {"n": 4},
                },
                {"price": 3, "active": False, "tags": []},
            ]
        }
    },
    {"a.b": {'quo"te': {"100%": [10, 20, 10]}}},
    {"store": 3, "empty_arr": [], "empty_obj": {}},
    [1, 1, True, False, None, "x", [5, 5], {"k": 1}],
    {"items": [{"v": 1}, {"v": 2}, {"v": 3}], "name": "keep"},
    {"items": [], "name": "empty"},
    {"items": [{"v": 9}], "name": "one"},
]


@pytest.fixture
def connection() -> typing.Iterator[sqlite3.Connection]:
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, document TEXT NOT NULL)"
    )
    con.executemany(
        "INSERT INTO documents (document) VALUES (?)",
        [(json.dumps(document),) for document in DOCUMENTS],
    )
    yield con
    con.close()


def _path_tokens(full_path: typing.Any) -> tuple[typing.Any, ...]:
    """Extract normalized path components from a ``find()`` datum's path."""

    chain: list[typing.Any] = []
    node = full_path
    while isinstance(node, Child):
        chain.append(node.right)
        node = node.left
    chain.append(node)
    chain.reverse()

    tokens: list[typing.Any] = []
    for segment in chain:
        if isinstance(segment, Fields):
            tokens.extend(segment.fields)
        elif isinstance(segment, Index):
            tokens.extend(segment.indices)
    return tuple(tokens)


def _resolve_negative(tokens: tuple[typing.Any, ...], document: typing.Any):
    """Resolve negative indices to concrete positions for comparison."""

    current = document
    resolved: list[typing.Any] = []
    for token in tokens:
        if isinstance(token, int) and token < 0 and isinstance(current, list):
            token = len(current) + token
        resolved.append(token)
        current = (
            current[token] if isinstance(current, (list, dict)) else None
        )
    return tuple(resolved)


def _find_reference(expression: str) -> dict[int, list[tuple[typing.Any, tuple]]]:
    grouped: dict[int, list[tuple[typing.Any, tuple]]] = {}
    for doc_index, document in enumerate(DOCUMENTS):
        matches = []
        for datum in parse(expression).find(document):
            tokens = _resolve_negative(_path_tokens(datum.full_path), document)
            matches.append((datum.value, tokens))
        if matches:
            grouped[doc_index] = matches
    return grouped


def _sql_reference(
    compiled: CompiledJSON1Query, connection: sqlite3.Connection
) -> dict[int, list[tuple[typing.Any, tuple]]]:
    grouped: dict[int, list[tuple[typing.Any, tuple]]] = {}
    # Run the raw SQL so each row carries its 1-based doc_id explicitly.
    from jsonpath_ng.ext.json1 import _substitute_relation

    sql = _substitute_relation(compiled.sql, "documents", "document", "id")
    for doc_id, value_text, path_text, _order in connection.execute(
        sql, compiled.bindings
    ):
        grouped.setdefault(doc_id - 1, []).append(
            (json.loads(value_text), tuple(json.loads(path_text)))
        )
    # The high-level execute() helper must agree, including canonical paths.
    for value, components, canonical in compiled.execute(connection):
        assert canonical == render_canonical_path(components)
        assert (value, tuple(components)) in [
            item for matches in grouped.values() for item in matches
        ]
    return grouped


SUPPORTED_EXPRESSIONS = [
    "$.store.book[*].tags[*]",
    "$.store.book[*].price",
    "$.store.book[*].active",
    "$.store.book[?(@.price < 10)].cat",
    "$.store.book[?(@.active == true)].cat",
    "$.store.book[?(@.active != true)].cat",
    "$.store.book[?(@.price == null)].cat",
    "$.store.book[?(@.price != null)].cat",
    "$.store.book[?(@.price < 10 && @.active == true)].cat",
    "$.store.book[?(@.cat == 'fiction' || @.price > 10)].cat",
    "$.store.book[?(@.tags)].cat",
    "$.store.book[?!@.tags].cat",
    "$.store.book[*].meta.n",
    "$['a.b']['quo\"te']['100%'][*]",
    "$[*]",
    "$[2]",
    "$[-1]",
    "$[*][0]",
    "$.store.book[0].tags[0]",
    "$.store.book[-1].price",
    "$.empty_arr[*]",
    "$.empty_obj[*]",
    "$.store.book[?(@.meta.n == 2)].cat",
    "$.store.book[?(@.price >= 5 && @.price <= 8)].cat",
    "$.store.book[?(!@.missing || @.cat == 'ref')].cat",
    "$.items[-1].v",
    "$.items[0].v",
    "$[?@.items[-1].v == 3].name",
    "$[?@.items[-1].v == 9].name",
    "$[?@.items[-1].v >= 2].name",
    "$[?@.items[-1].v != 2].name",
    "$",
]


@pytest.mark.parametrize("expression", SUPPORTED_EXPRESSIONS)
def test_sql_matches_find(expression, connection):
    compiled = compile_json1(expression)
    assert _sql_reference(compiled, connection) == _find_reference(expression)


# Boolean vs number vs null vs missing semantics, exhaustively cross-checked.
SCALAR_DOCUMENTS = [
    {"b": True},
    {"b": False},
    {"b": 1},
    {"b": 0},
    {"b": None},
    {},
    {"b": "x"},
    {"b": 2.5},
    {"b": [1]},
    {"b": {"k": 1}},
]

LITERALS = ["true", "false", "1", "0", "null", "'x'", "2", "2.5"]
OPERATORS = ["==", "!=", "<", "<=", ">", ">="]


@pytest.fixture
def scalar_connection() -> typing.Iterator[sqlite3.Connection]:
    # The filter selector applies to an array, so wrap the scalars once.
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, document TEXT NOT NULL)"
    )
    con.execute(
        "INSERT INTO documents (document) VALUES (?)",
        (json.dumps(SCALAR_DOCUMENTS),),
    )
    yield con
    con.close()


@pytest.fixture
def current_node_connection() -> typing.Iterator[sqlite3.Connection]:
    # A bare current-node filter applies to every kind of array element.
    elements = [1, None, False, 0, "", [], {}, "x", True]
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, document TEXT NOT NULL)"
    )
    con.execute(
        "INSERT INTO documents (document) VALUES (?)",
        (json.dumps(elements),),
    )
    yield con
    con.close()


def _scalar_hits_sql(expression, connection):
    compiled = compile_json1(expression)
    return [
        json.loads(value_text)
        for _id, value_text, _path, _order in connection.execute(
            _substitute_relation(compiled.sql, "documents", "document", "id"),
            compiled.bindings,
        )
    ]


def _scalar_hits_find(expression):
    return [datum.value for datum in parse(expression).find(SCALAR_DOCUMENTS)]


@pytest.mark.parametrize("literal", LITERALS)
@pytest.mark.parametrize("operator", OPERATORS)
def test_scalar_comparison_semantics(literal, operator, scalar_connection):
    expression = f"$[?@.b {operator} {literal}]"
    assert _scalar_hits_sql(expression, scalar_connection) == _scalar_hits_find(
        expression
    )


@pytest.mark.parametrize(
    "expression",
    [
        "$[?@.b]",
        "$[?!@.b]",
        "$[?@.b < 'y']",
        "$[?@.b > 'a']",
        "$[?@.b < 2]",
        "$[?@.b >= 2.5]",
        "$[?@.b == 'x']",
        "$[?@.b != 'x']",
    ],
)
def test_existence_and_ordering_semantics(expression, scalar_connection):
    assert _scalar_hits_sql(
        expression, scalar_connection
    ) == _scalar_hits_find(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "$[?@]",
        "$[?!@]",
        "$[?@ == 1]",
        "$[?@ != null]",
        "$[?@ == false]",
        "$[?@ == true]",
        "$[?@ == 0]",
        "$[?@ < 2]",
        "$[?@ != 1]",
        "$[?@ == 'x']",
        "$[?@ == null]",
        "$[?@ >= 1]",
    ],
)
def test_bare_current_node_semantics(expression, current_node_connection):
    elements = json.loads(
        current_node_connection.execute(
            "SELECT document FROM documents"
        ).fetchone()[0]
    )
    expected = [datum.value for datum in parse(expression).find(elements)]
    assert _scalar_hits_sql(
        expression, current_node_connection
    ) == expected


def test_boolean_and_number_stay_distinct(connection):
    # true must never equal 1, and false must never equal 0.
    compiled = compile_json1("$.store.book[?(@.active == true)].cat")
    result = {value for value, _p, _c in compiled.execute(connection)}
    assert result == {"fiction", "ref"}

    compiled = compile_json1("$.store.book[?(@.active == false)].cat")
    assert [value for value, _p, _c in compiled.execute(connection)] == [
        "fiction"
    ]

    # Booleans are returned as Python bools, null as None, numbers as numbers.
    compiled = compile_json1("$.store.book[*].active")
    values = [value for value, _p, _c in compiled.execute(connection)]
    assert values[:2] == [True, False]
    assert all(isinstance(value, bool) for value in values)

    compiled = compile_json1("$.store.book[*].price")
    prices = [value for value, _p, _c in compiled.execute(connection)]
    assert prices == [8, 12, 5, None, 3]
    assert prices[3] is None
    assert all(
        isinstance(value, (int, float)) for value in prices if value is not None
    )


def test_array_order_follows_traversal_order(connection):
    compiled = compile_json1("$.store.book[*].tags[*]")
    values = [value for value, _p, _c in compiled.execute(connection)]
    # Repeated value "a" is retained and the order is the document order.
    assert values == ["a", "b", "a", "c", "d"]

    # Paths within one array are ascending and nested arrays stay in order.
    compiled = compile_json1("$[*]")
    rows = compiled.execute(connection)
    # Document index 3 is the only root-level array document; execute() keeps
    # insertion order, so collect its consecutive single-component paths.
    root_array_hits = [
        comps
        for _value, comps, _canonical in rows
        if len(comps) == 1 and isinstance(comps[0], int)
    ]
    assert root_array_hits == [[0], [1], [2], [3], [4], [5], [6], [7]]


def test_empty_matches_return_no_rows(connection):
    for expression in [
        "$.empty_arr[*]",
        "$.empty_obj[*]",
        "$.store.book[99].tags",
        "$.missing.chain[*]",
        "$.store.book[?(@.price > 1000)].cat",
    ]:
        assert compile_json1(expression).execute(connection) == []


def test_duplicate_values_are_retained(connection):
    compiled = compile_json1("$['a.b']['quo\"te']['100%'][*]")
    values = [value for value, _p, _c in compiled.execute(connection)]
    assert values == [10, 20, 10]


UNSUPPORTED_EXPRESSIONS = [
    ("$.a[0:2]", "slice"),
    ("$..book", "descent"),
    ("$.a['x','y']", "ulti-field"),
    ("$.a[1,3]", "ulti-index"),
    ("$[?@.a =~ 'x.*']", "egular-expression"),
    ("$[?@.a == @.b]", "comparing one query"),
    ("$[?$.x == 1]", "root document"),
    ("$.a where $.b", "Where"),
]


@pytest.mark.parametrize(
    "expression, fragment", UNSUPPORTED_EXPRESSIONS
)
def test_unsupported_ast_is_rejected_before_execution(expression, fragment):
    with pytest.raises(JSON1CompileError) as exc_info:
        compile_json1(expression)
    assert fragment in str(exc_info.value)


def test_field_names_are_bound_not_inlined(connection):
    expression = "$['a.b']['quo\"te']['100%']"
    compiled = compile_json1(expression)
    # User data travels through bindings, never as SQL syntax.
    for name in ["a.b", 'quo"te', "100%"]:
        assert name not in compiled.sql
        assert name in compiled.bindings
    assert compiled.execute(connection)


def test_compiled_query_exposes_sql_bindings_and_scope():
    compiled = compile_json1("$.a[?(@.b == 1)]")
    assert isinstance(compiled, CompiledJSON1Query)
    assert compiled.sql.lstrip().startswith("WITH")
    assert "SELECT" in compiled.sql
    assert 1 in compiled.bindings
    assert "wildcard" in compiled.supported_scope
    # The fragment holds the CTE chain; the final standalone query lives in sql.
    assert compiled.fragment.lstrip().startswith("WITH")
    assert compiled.sql.endswith("ORDER BY doc_id, ord")
    assert compiled.sql.startswith(compiled.fragment)


def test_ast_input_is_accepted():
    ast = parse("$.store.book[0].cat")
    assert compile_json1(ast).sql == compile_json1(
        "$.store.book[0].cat"
    ).sql


def test_custom_relation_names_are_supported(connection):
    connection.execute(
        'CREATE TABLE "my docs" ("pk" INTEGER PRIMARY KEY, "raw json" TEXT)'
    )
    connection.execute(
        'INSERT INTO "my docs" ("raw json") VALUES (?)',
        (json.dumps(DOCUMENTS[0]),),
    )
    compiled = compile_json1("$.store.book[0].cat")
    rows = compiled.execute(
        connection,
        table="my docs",
        document_column="raw json",
        id_column="pk",
    )
    assert rows == [
        (
            "fiction",
            ["store", "book", 0, "cat"],
            '$["store"]["book"][0]["cat"]',
        )
    ]


def test_render_canonical_path():
    assert render_canonical_path([]) == "$"
    assert render_canonical_path(["a", 0]) == '$["a"][0]'
    assert render_canonical_path(['quo"te', "100%"]) == (
        '$["quo\\"te"]["100%"]'
    )
