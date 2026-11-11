"""Helpers shared by the SQLite JSON1 compiler tests."""

from __future__ import annotations

import json
import sqlite3
import typing

from jsonpath_ng.jsonpath import Child
from jsonpath_ng.jsonpath import Fields
from jsonpath_ng.jsonpath import Index
from jsonpath_ng.jsonpath import JSONPath
from jsonpath_ng.jsonpath import Root


OPEN_CONNECTIONS: list[sqlite3.Connection] = []


def make_connection(documents: list[typing.Any]) -> sqlite3.Connection:
    """Create an in-memory SQLite table holding the given JSON documents."""

    connection = sqlite3.connect(":memory:")
    OPEN_CONNECTIONS.append(connection)
    connection.execute("create table documents(doc text)")
    connection.executemany(
        "insert into documents(doc) values (?)",
        [(json.dumps(document, ensure_ascii=False),) for document in documents],
    )
    return connection


def canonical_path(datum: typing.Any) -> str:
    """Build the normalized ``$["field"][0]`` path for a find() datum."""

    chain: list[typing.Any] = []
    current = datum
    while current is not None and not isinstance(current.path, Root):
        chain.append(current)
        current = current.context
    chain.reverse()

    path = "$"
    for item in chain:
        segment = item.path
        if isinstance(segment, Fields):
            (field,) = segment.fields
            path += "[" + json.dumps(field, ensure_ascii=False) + "]"
        elif isinstance(segment, Index):
            (index,) = segment.indices
            if index < 0:
                index = len(item.context.value) + index
            path += f"[{index}]"
        else:
            raise AssertionError(f"Unexpected path segment: {segment!r}")
    return path


def decode_value(value: typing.Any, type_name: str | None) -> typing.Any:
    """Turn a SQL result cell back into the original Python JSON value."""

    if type_name == "true":
        return True
    if type_name == "false":
        return False
    if type_name == "null":
        return None
    if type_name in ("object", "array"):
        return json.loads(value)
    return value


def find_expectation(
    expression: JSONPath, documents: list[typing.Any]
) -> list[tuple[typing.Any, str]]:
    """Return find() results as normalized ``(value, path)`` pairs."""

    expected: list[tuple[typing.Any, str]] = []
    for document in documents:
        for datum in expression.find(document):
            expected.append((datum.value, canonical_path(datum)))
    return expected


def sql_expectation(
    compiled: typing.Any, connection: sqlite3.Connection
) -> list[tuple[typing.Any, str]]:
    """Run a compilation and return normalized ``(value, path)`` pairs."""

    return [
        (decode_value(value, type_name), path)
        for value, path, type_name in connection.execute(
            compiled.sql, compiled.parameters
        )
    ]


def assert_parity(
    expression: JSONPath,
    documents: list[typing.Any],
    connection: sqlite3.Connection,
    compile_jsonpath: typing.Any,
) -> typing.Any:
    """Assert SQL results equal find() results; return the compilation."""

    compiled = compile_jsonpath(expression)
    actual = sql_expectation(compiled, connection)
    expected = find_expectation(expression, documents)
    assert actual == expected
    return compiled
