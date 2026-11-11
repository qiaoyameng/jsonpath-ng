"""Compare SQLite JSON1 results with JSONPath ``find()`` results.

Every supported query is compiled, executed against an in-memory SQLite
database using only the standard library, and checked to match
``expression.find(document)`` for the same batch of documents.
"""

from __future__ import annotations

import pytest

from jsonpath_ng.ext.parser import parse
from jsonpath_ng.ext.sqlite import compile_jsonpath

from .helpers import assert_parity
from .helpers import make_connection


# Documents deliberately cover nested arrays, repeated values, empty
# containers, missing members, explicit null, booleans, numbers, strings,
# and field names containing dots, quotes, and percent signs.
DOCUMENTS = [
    {
        "id": 1,
        "name": "alpha",
        "active": True,
        "score": 10,
        "ratio": 0.5,
        "note": None,
        "tags": ["x", "y", "x"],
        "matrix": [[1, 2], [3, 4]],
        "records": [
            {"k": "a", "n": 1, "flag": True, "meta": {"q": 1}},
            {"k": "b", "n": 2, "flag": False, "meta": {"q": 2}},
            {"k": "a", "n": 3, "flag": True, "meta": {}},
        ],
        "dot.name": 1,
        'quo"te': 2,
        "pct%": 3,
    },
    {
        "id": 2,
        "name": "beta",
        "active": False,
        "score": 10,
        "ratio": 1.5,
        "note": "hello",
        "tags": [],
        "matrix": [],
        "records": [],
        "dot.name": 9,
    },
    {"id": 3, "name": 10, "active": True, "score": None, "tags": ["z"]},
    {},
    {"id": 5, "records": [{"k": "c", "n": 1}, {"k": "c", "n": 1}]},
    {"id": 6, "matrix": [[9]]},
]

ARRAY_DOCUMENT = [
    {"id": 1, "active": True, "kind": "red"},
    {"id": 2, "active": False, "kind": "red"},
    {"id": 3, "active": True, "kind": "blue"},
]


@pytest.mark.parametrize(
    "query",
    [
        # Fields, including missing members and special field names.
        "$.id",
        "$.name",
        "$.active",
        "$.note",
        "$.missing",
        "$['dot.name']",
        "$['quo\"te']",
        "$['pct%']",
        "$['missing','id']",
        # Array indices: positive, negative, multiple, out of range.
        "$.tags[0]",
        "$.tags[-1]",
        "$.tags[0,2]",
        "$.tags[2,0]",
        "$.tags[99]",
        # Wildcards over objects and arrays, at multiple array levels.
        "$.tags[*]",
        "$.matrix[*]",
        "$.matrix[*][*]",
        "$.matrix[-1][-1]",
        "$.records[*].k",
        "$.records[*].n",
        "$.records[*].meta.q",
        # Empty matches on empty containers and missing paths.
        "$.records[*].absent",
        "$.missing.deep[*].x",
        # Filters: existence, negation, scalar comparison, and/or.
        "$.records[?@.flag == true].k",
        "$.records[?@.flag == false].k",
        "$.records[?@.n > 1].k",
        "$.records[?@.n >= 2 && @.flag == false].k",
        "$.records[?@.n == 1 || @.n == 3].k",
        "$.records[?@.meta.q]",
        "$.records[?!@.flag].k",
        "$.records[?@.n != null].k",
        "$.records[?@.n == null].k",
        "$.records[?@.n != 2].k",
        "$.records[?@.n < 3].k",
        "$.records[?@.n <= 2].k",
        "$.records[?@.k == 'a'].n",
        "$.records[?@.n == 1 || @.n == 2 || @.n == 3].n",
        "$.records[?(@.n == 1 && @.flag == true) || @.n == 2].k",
        "$.records[?@.n == 1 && @.n != 2].k",
        # Root wildcard across the whole document batch.
        "$[*].id",
        "$[*].missing",
    ],
)
def test_sql_matches_find(query):
    connection = make_connection(DOCUMENTS)
    assert_parity(parse(query), DOCUMENTS, connection, compile_jsonpath)


@pytest.mark.parametrize(
    "query",
    [
        "$[?@.active == true].id",
        "$[?@.active != true].id",
        "$[?@.active == false].id",
        "$[?@.id >= 2].kind",
        "$[?@.id == 1 || @.kind == 'blue'].id",
        "$[?!@.active].id",
    ],
)
def test_array_root_filters_match_find(query):
    connection = make_connection([ARRAY_DOCUMENT])
    assert_parity(parse(query), [ARRAY_DOCUMENT], connection, compile_jsonpath)


def test_duplicate_values_are_all_returned_in_order():
    # ``tags`` contains "x" twice and records repeat "a"/"c".
    connection = make_connection(DOCUMENTS)
    compiled = compile_jsonpath(parse("$.tags[*]"))
    values = [value for value, _path, _type in compiled.execute(connection)]

    assert values == ["x", "y", "x", "z"]


def test_empty_match_yields_no_rows():
    connection = make_connection([{"a": []}, {}, {"a": [1]}])
    compiled = compile_jsonpath(parse("$.a[*]"))

    assert compiled.execute(connection) == [
        (1, '$["a"][0]', "integer"),
    ]


def test_results_keep_original_path_order_for_objects():
    connection = make_connection([{"o": {"z": 1, "a": 2, "m": 3}}])
    compiled = compile_jsonpath(parse("$.o[*]"))

    assert [value for value, _path, _type in compiled.execute(connection)] == [
        1,
        2,
        3,
    ]


def test_multi_index_keeps_selector_order():
    connection = make_connection([{"a": [10, 20, 30, 40]}])
    compiled = compile_jsonpath(parse("$.a[3,1]"))

    assert [value for value, _path, _type in compiled.execute(connection)] == [
        40,
        20,
    ]
