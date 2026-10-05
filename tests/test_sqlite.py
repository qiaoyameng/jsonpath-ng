"""Tests for compiling the JSONPath subset into parameterized SQLite SQL.

Every behavioral test cross-checks the generated SQL (executed against an
in-memory :mod:`sqlite3` database) against the pure-Python ``find`` reference
evaluation on the same documents, so no database server is required.
"""

import json
import sqlite3

import pytest

from jsonpath_ng.sqlite import (
    CompiledJSONPath,
    JSONPathError,
    JSONPathNotSupported,
    JSONPathSyntaxError,
    compile_jsonpath,
    decode_row,
    find,
)


DOCS = [
    {
        "store": {
            "book": [
                {"category": "reference", "author": "Nigel Rees",
                 "price": 8.95, "active": True, "note": None},
                {"category": "fiction", "author": "Evelyn Waugh",
                 "price": 12.99, "active": False, "note": None},
                {"category": "fiction", "author": "Herman Melville",
                 "price": 8.99, "active": True, "note": "x"},
                {"category": "fiction", "author": "J. R. R. Tolkien",
                 "price": 22.99, "active": True},
            ],
            "matrix": [[1, 2], [3, [4, 5]], []],
            "special": {'a"b': -1, "k.k": 2, "s'q": 3},
        },
    },
    [None, True, 42, 3.5, "s", [], {"nested": [True, None]}],
    {"empty": [], "obj_field_on_array_case": {"x": 1}},
    5,
    "plain text",
    None,
    True,
]


@pytest.fixture
def connection():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE docs(data TEXT)")
    con.executemany(
        "INSERT INTO docs(data) VALUES (?)",
        [(json.dumps(doc),) for doc in DOCS],
    )
    yield con
    con.close()


def _reference(expression):
    rows = []
    for doc_id, doc in enumerate(DOCS, start=1):
        for match in find(expression, doc):
            rows.append((doc_id, match.path, match.value))
    return rows


def _sql_rows(connection, expression, table="docs", column="data"):
    compiled = compile_jsonpath(expression, table, column)
    rows = []
    for doc_id, path, value_text, json_type, _ord in connection.execute(
        compiled.sql, compiled.params
    ):
        rows.append((doc_id, path, decode_row(value_text, json_type)))
    return rows, compiled


@pytest.mark.parametrize(
    "expression",
    [
        "$",
        "$.store",
        "$.store.book",
        "$.store.book[0].price",
        "$.store.book[*].price",
        "$.store.book[*].author",
        "$.store.book[-1].author",
        "$.store.book[10].author",
        "$.store.matrix[*][*]",
        "$.store.matrix[1][1][*]",
        "$.store.matrix[2][*]",
        "$.store.special['a\"b']",
        "$.store.special[\"k.k\"]",
        "$.store.special[\"s'q\"]",
        "$[*]",
        "$[*].nested[*]",
        "$.empty[*]",
        "$.missing",
        "$.store.missing.deeper",
        "$.store.book[?(@.price < 10)].author",
        "$.store.book[?(@.price == 8.95)].category",
        "$.store.book[?(@.price >= 8.99)].price",
        "$.store.book[?(@.price != 12.99)].author",
        "$.store.book[?(@.category == 'fiction')].author",
        "$.store.book[?('fiction' == @.category)].author",
        "$.store.book[?(@.active == true)].author",
        "$.store.book[?(@.active != true)].author",
        "$.store.book[?(@.note == null)].author",
        "$.store.book[?(@.note != null)].author",
        "$.store.book[?(@.missing == null)].author",
        "$.store.book[?(@.missing != null)].author",
        "$.store.matrix[*][?(@[0] == 3)][0]",
    ],
)
def test_sql_matches_reference(connection, expression):
    expected = _reference(expression)
    actual, _compiled = _sql_rows(connection, expression)
    assert actual == expected


def test_multiple_hits_preserve_array_order(connection):
    expression = "$.store.book[*].author"
    actual, _ = _sql_rows(connection, expression)
    assert [value for _id, _path, value in actual] == [
        "Nigel Rees",
        "Evelyn Waugh",
        "Herman Melville",
        "J. R. R. Tolkien",
    ]


def test_nested_arrays_are_flattened_in_depth_first_order(connection):
    actual, _ = _sql_rows(connection, "$.store.matrix[*][*]")
    assert [(path, value) for _id, path, value in actual] == [
        ("$['store']['matrix'][0][0]", 1),
        ("$['store']['matrix'][0][1]", 2),
        ("$['store']['matrix'][1][0]", 3),
        ("$['store']['matrix'][1][1]", [4, 5]),
    ]

    actual3, _ = _sql_rows(connection, "$.store.matrix[*][*][*]")
    # only the innermost array [4, 5] survives a third expansion; the
    # scalar leaves 1/2/3 correctly yield no further rows.
    assert [(path, value) for _id, path, value in actual3] == [
        ("$['store']['matrix'][1][1][0]", 4),
        ("$['store']['matrix'][1][1][1]", 5),
    ]


def test_empty_match_returns_no_rows(connection):
    for expression in ["$.empty[*]", "$.store.book[99].author",
                       "$.store.missing", "$.store.book[?(@.price > 1000)]"]:
        assert _sql_rows(connection, expression)[0] == []
        assert find(expression, DOCS[0]) == []


def test_missing_null_number_and_boolean_are_distinct(connection):
    docs = [{"items": [
        {"v": None},
        {"v": 1},
        {"v": 0},
        {"v": True},
        {"v": False},
        {},
        {"v": "1"},
    ]}]
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE d(data TEXT)")
    con.execute("INSERT INTO d(data) VALUES (?)", (json.dumps(docs[0]),))

    def paths(expr):
        compiled = compile_jsonpath(expr, "d", "data")
        return [
            path
            for _id, path, _value, _jt, _ord in con.execute(compiled.sql,
                                                            compiled.params)
        ]

    assert paths("$.items[?(@.v == null)].v") == ["$['items'][0]['v']"]
    assert paths("$.items[?(@.v != null)].v") == [
        "$['items'][1]['v']", "$['items'][2]['v']",
        "$['items'][3]['v']", "$['items'][4]['v']",
        "$['items'][6]['v']",
    ]
    assert paths("$.items[?(@.v == 1)].v") == ["$['items'][1]['v']"]
    assert paths("$.items[?(@.v == true)].v") == ["$['items'][3]['v']"]
    assert paths("$.items[?(@.v == false)].v") == ["$['items'][4]['v']"]
    assert paths("$.items[?(@.v == 0)].v") == ["$['items'][2]['v']"]
    # a missing value matches neither equality nor inequality
    assert paths("$.items[?(@.v == null)].v") != ["$['items'][5]['v']"]
    assert paths("$.items[?(@.missing == null)].v") == []
    assert paths("$.items[?(@.missing != null)].v") == []
    con.close()


def test_null_integer_real_true_false_value_types(connection):
    actual, _ = _sql_rows(connection, "$[*]")
    # second document is the mixed array
    sub = [row for row in actual if row[0] == 2]
    assert [value for _id, _path, value in sub] == [
        None, True, 42, 3.5, "s", [], {"nested": [True, None]},
    ]


def test_ordering_does_not_depend_on_physical_row_order(connection):
    # Reverse insertion order via a second table; the ord key must keep
    # array order stable independent of rowid order.
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE rev(data TEXT)")
    con.executemany(
        "INSERT INTO rev(data) VALUES (?)",
        [(json.dumps(doc),) for doc in reversed(DOCS[:3])],
    )
    expression = "$.store.book[*].author"
    compiled = compile_jsonpath(expression, "rev", "data")
    rows = list(con.execute(compiled.sql, compiled.params))
    values = [decode_row(v, t) for _id, _p, v, t, _o in rows]
    assert values == [
        "Nigel Rees", "Evelyn Waugh", "Herman Melville", "J. R. R. Tolkien",
    ]
    con.close()


def test_fragment_filters_documents_with_exists(connection):
    compiled = compile_jsonpath("$.store.book[?(@.price < 10)].author",
                                "docs", "data")
    query = (f"SELECT rowid FROM docs _t WHERE {compiled.exists_sql()} "
             "ORDER BY rowid")
    matched = connection.execute(query, compiled.params)
    assert [row[0] for row in matched] == [1]


def test_fragment_is_a_usable_subquery(connection):
    compiled = compile_jsonpath("$.store.book[*].author", "docs", "data")
    query = (f"SELECT m.doc_id, m.path, m.value, m.json_type "
             f"FROM ({compiled.fragment}) m ORDER BY m.doc_id, m.ord, m.path")
    rows = list(connection.execute(query, compiled.params))
    assert len(rows) == 4
    assert all(r[1].startswith("$['store']['book'][") for r in rows)


def test_sql_is_fully_parameterized_for_data(connection):
    expression = "$.store.special['a\"b']"
    compiled = compile_jsonpath(expression, "docs", "data")
    # the key value only travels through a bind parameter, never inlined
    assert 'a"b' not in compiled.sql
    assert any(value == 'a"b' for value in compiled.params.values())
    rows = list(connection.execute(compiled.sql, compiled.params))
    assert decode_row(rows[0][2], rows[0][3]) == -1


def test_key_quote_and_special_names_are_data(connection):
    malicious = "x') OR 1=1 --"
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE d(data TEXT)")
    con.execute("INSERT INTO d(data) VALUES (?)",
                (json.dumps({malicious: 7, "safe": 8}),))
    compiled = compile_jsonpath(f"$[{json.dumps(malicious)}]", "d", "data")
    rows = list(con.execute(compiled.sql, compiled.params))
    assert decode_row(rows[0][2], rows[0][3]) == 7
    con.close()


def test_identifiers_are_validated_and_quoted():
    compiled = compile_jsonpath("$.a", "docs", "data")
    assert compiled.table == '"docs"'
    assert compiled.column == '"data"'
    for bad in ['docs; DROP TABLE docs--', '"x"', "a b", "1a", ""]:
        with pytest.raises(JSONPathError):
            compile_jsonpath("$", bad, "data")
        with pytest.raises(JSONPathError):
            compile_jsonpath("$", "docs", bad)


@pytest.mark.parametrize(
    "expression",
    [
        "$..a",            # recursive descent
        "$.*",             # object wildcard
        "$.a[0,1]",        # union of indices
        "$.a[1:3]",        # slice
        "$.a['x','y']",    # union of fields
        "$.a[?(@.b)]",     # existence / non scalar predicate
        "$.a[?(@.b == @.c)]",  # two relative paths
        "$.a[?(@.b =~ /x/)]",  # regex
        "$.a[?(@.b == 1 || @.c == 2)]",  # boolean combination
        "$.a[?(@.b == [1])]",  # array literal
        "$.a[?(@.b == {'x': 1})]",  # object literal
        "$.a[?(@.b < null)]",
        "$.a[?(@.b < true)]",
        "$.a[?(true)]",
        "a.b",             # no root
        "$$",
        "$.",
        "$.a[?len(@) > 0]",  # function
    ],
)
def test_unsupported_expressions_are_rejected(expression):
    with pytest.raises(JSONPathError):
        compile_jsonpath(expression, "docs", "data")


def test_unsupported_is_not_silently_dropped(connection):
    # If a compiler ignored an unsupported condition, the query below would
    # return rows; it must instead raise.
    expression = "$.store.book[?(@.price < 10 || @.active == true)].author"
    with pytest.raises(JSONPathError):
        compile_jsonpath(expression, "docs", "data")


def test_array_wildcard_is_supported_but_object_wildcard_is_not():
    assert isinstance(compile_jsonpath("$.a[*]", "t", "c"), CompiledJSONPath)
    with pytest.raises(JSONPathNotSupported):
        compile_jsonpath("$.*", "t", "c")


def test_syntax_error_type():
    with pytest.raises(JSONPathSyntaxError):
        compile_jsonpath("$.a[", "t", "c")


def test_find_returns_match_objects_with_normalized_paths():
    doc = {"a b": [{"c": 1}]}
    matches = find("$['a b'][*].c", doc)
    assert len(matches) == 1
    assert matches[0].value == 1
    assert matches[0].path == "$['a b'][0]['c']"


def test_compiled_result_is_a_dataclass_with_params():
    compiled = compile_jsonpath("$.a[?(@.b == 1)]", "t", "c")
    assert isinstance(compiled, CompiledJSONPath)
    assert compiled.expression == "$.a[?(@.b == 1)]"
    assert compiled.fragment.startswith("WITH ")
    assert compiled.sql.startswith("WITH ")
    assert "ORDER BY doc_id, ord, path" in compiled.sql
    # literals and keys are bound, identifiers quoted
    assert all(isinstance(key, str) for key in compiled.params)
