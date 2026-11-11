"""Verify the SQLite JSON1 compilation entry point itself."""

import json
import sqlite3

import pytest

from jsonpath_ng.exceptions import SQLiteCompilationError
from jsonpath_ng.ext.parser import parse
from jsonpath_ng.ext.sqlite import JSON1Compilation
from jsonpath_ng.ext.sqlite import SUPPORTED_SCOPE
from jsonpath_ng.ext.sqlite import compile_jsonpath

from .helpers import make_connection


def test_compilation_result_shape():
    compiled = compile_jsonpath(parse("$.a[*]"))

    assert isinstance(compiled, JSON1Compilation)
    assert isinstance(compiled.sql, str)
    assert compiled.sql.startswith("with ")
    assert isinstance(compiled.parameters, tuple)
    assert compiled.parameters == ("a",)
    assert compiled.scope == SUPPORTED_SCOPE


def test_scope_documents_supported_and_rejected_syntax():
    assert "Field selectors" in SUPPORTED_SCOPE
    assert "Array index selectors" in SUPPORTED_SCOPE
    assert "Wildcards" in SUPPORTED_SCOPE
    assert "&& (and) and || (or)" in SUPPORTED_SCOPE
    assert "recursive descent" in SUPPORTED_SCOPE
    assert "SQLiteCompilationError" in SUPPORTED_SCOPE


def test_sql_is_a_complete_statement_with_parameters():
    compiled = compile_jsonpath(parse("$.a[?@.b > 1]"))

    connection = make_connection([{"a": [{"b": 2}]}])
    rows = connection.execute(compiled.sql, compiled.parameters).fetchall()
    compact_object = json.dumps({"b": 2}, separators=(",", ":"))

    assert rows == [(compact_object, '$["a"][0]', "object")]


def test_execute_helper_returns_value_path_and_type():
    compiled = compile_jsonpath(parse("$.flag"))

    connection = make_connection([{"flag": True}, {"flag": 1}])

    assert compiled.execute(connection) == [
        ("true", '$["flag"]', "true"),
        (1, '$["flag"]', "integer"),
    ]


def test_custom_table_and_column_names():
    compiled = compile_jsonpath(parse("$.x"), table="docs", column="payload")

    connection = sqlite3.connect(":memory:")
    connection.execute("create table docs(payload text)")
    connection.execute("insert into docs values (?)", ('{"x": 5}',))
    try:
        assert compiled.execute(connection) == [(5, '$["x"]', "integer")]
    finally:
        connection.close()


@pytest.mark.parametrize("identifier", ["", "a-b", "a.b", "a b", 'a"b', "0;drop"])
def test_invalid_identifiers_are_rejected(identifier):
    with pytest.raises(SQLiteCompilationError):
        compile_jsonpath(parse("$.x"), table=identifier)


def test_compilation_is_deterministic():
    expression = parse("$.a[*][?@.b == 1].c")

    first = compile_jsonpath(expression)
    second = compile_jsonpath(expression)

    assert first == second
