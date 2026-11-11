"""Verify that unsupported AST nodes are rejected before execution."""

import pytest

from jsonpath_ng.exceptions import SQLiteCompilationError
from jsonpath_ng.ext.parser import parse
from jsonpath_ng.ext.sqlite import compile_jsonpath


@pytest.mark.parametrize(
    "query",
    [
        "$..x",
        "$.a[0:2]",
        "$.a where @.b",
        "$.a wherenot @.b",
        "`parent`",
        "$.a[?@.x == @.y]",
        "$[?$.x == 1]",
        "$.a[?@['x','y']]",
        "$.a[?@[*].x == 1]",
    ],
)
def test_unsupported_ast_is_rejected(query):
    with pytest.raises(SQLiteCompilationError):
        compile_jsonpath(parse(query))


def test_arithmetic_filter_operand_is_rejected():
    with pytest.raises(SQLiteCompilationError):
        compile_jsonpath(parse("$.a[?@.x + 1 == 2]"))


def test_rejection_message_names_the_node():
    with pytest.raises(SQLiteCompilationError, match="Descendants"):
        compile_jsonpath(parse("$..x"))


def test_filter_after_wildcard_is_still_supported():
    # Filters are child selectors and must not be silently dropped.
    compiled = compile_jsonpath(parse("$.a[?@.x == 1].b"))

    assert "json_each" in compiled.sql
    assert compiled.parameters == ("a", 1, "b")
