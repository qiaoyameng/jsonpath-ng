from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

from .exceptions import JSONPathError
from .ext.filter import Filter as FilterNode
from .jsonpath import (
    Child,
    Descendants,
    Fields,
    Index,
    Intersect,
    JSONPath,
    Parent,
    Root,
    Slice,
    This,
    Union,
    Where,
    WhereNot,
)


class SQLiteJSONPathError(JSONPathError):
    """The JSONPath expression cannot be compiled for SQLite."""


class SQLiteCompiledJSONPath(NamedTuple):
    sql: str
    parameters: tuple[Any, ...]

    @property
    def query(self) -> str:
        return self.sql

    @property
    def params(self) -> tuple[Any, ...]:
        return self.parameters

    def __iter__(self):
        yield self.sql
        yield self.parameters


@dataclass(frozen=True)
class _FieldStep:
    name: str


@dataclass(frozen=True)
class _IndexStep:
    index: int


@dataclass(frozen=True)
class _ArrayWildcardStep:
    pass


@dataclass(frozen=True)
class _FilterStep:
    expressions: tuple[Any, ...]


_COMPARISON_OPERATORS = frozenset({"==", "=", "!=", "<", "<=", ">", ">="})
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _json_value(alias: str) -> str:
    return (
        f"CASE {alias}.type "
        f"WHEN 'object' THEN {alias}.value "
        f"WHEN 'array' THEN {alias}.value "
        "WHEN 'null' THEN 'null' "
        "WHEN 'true' THEN 'true' "
        "WHEN 'false' THEN 'false' "
        f"ELSE json_quote({alias}.value) END"
    )


def _quoted_json_string(reference: str) -> str:
    encoded = f"json_array({reference})"
    return f"substr({encoded}, 2, max(length({encoded}) - 2, 0))"


def compile_sqlite(
    expression: str | JSONPath,
    table: str,
    column: str,
    *,
    extended: bool = True,
) -> SQLiteCompiledJSONPath:
    """Compile supported JSONPath into a parameterized SQLite SELECT.

    Result columns are ``source_rowid``, ``match_value`` (JSON text), and
    ``match_path`` (a normalized root-relative JSONPath). Array matches are
    ordered by explicit array indices, not SQLite's physical row order.
    """
    parsed = _parse_expression(expression, extended=extended)
    steps = _flatten(parsed)
    parameters: list[Any] = []
    table_name = _quote_identifier(table, "table")
    column_name = _quote_identifier(column, "column")
    ctes: list[str] = [
        "source AS (\n"
        f"        SELECT rowid AS source_rowid, column_value.{column_name} AS value,\n"
        f"            CASE WHEN json_valid(column_value.{column_name}) THEN\n"
        f"                json_type(column_value.{column_name})\n"
        "            END AS type,\n"
        "            '$' AS path, '' AS sort_key\n"
        f"        FROM {table_name} column_value\n"
        "    )"
    ]

    for position, step in enumerate(steps):
        previous = "source" if position == 0 else f"step_{position - 1}"
        current = f"step_{position}"
        ctes.append(_compile_step(step, previous, current, parameters))

    final = f"step_{len(steps) - 1}" if steps else "source"
    final_path = "path" if steps else "'$'"
    final_sort_key = "sort_key" if steps else "''"
    sql = (
        f"WITH {',\n    '.join(ctes)}\n"
        "SELECT source_rowid, match_value, match_path FROM (\n"
        f"    SELECT source_rowid, value AS match_value, {final_path} AS match_path,\n"
        f"        {final_sort_key} AS sort_key\n"
        f"    FROM {final}\n"
        ")\n"
        "ORDER BY source_rowid, sort_key"
    )
    return SQLiteCompiledJSONPath(sql, tuple(parameters))


def compile_to_sqlite(
    expression: str | JSONPath,
    table: str,
    column: str,
    *,
    extended: bool = True,
) -> SQLiteCompiledJSONPath:
    return compile_sqlite(expression, table, column, extended=extended)


compile_sql = compile_sqlite


def _parse_expression(expression: str | JSONPath, *, extended: bool) -> JSONPath:
    if isinstance(expression, JSONPath):
        return expression
    if extended:
        from .ext import parse as parse_extended

        return parse_extended(expression)
    from .parser import parse

    return parse(expression)


def _flatten(node: JSONPath) -> tuple[Any, ...]:
    if isinstance(node, Root):
        return ()
    if isinstance(node, Child):
        return _flatten(node.left) + _flatten(node.right)
    if isinstance(node, Fields):
        if len(node.fields) != 1:
            raise SQLiteJSONPathError("Multiple field names are not supported")
        if node.fields[0] == "*":
            raise SQLiteJSONPathError("Object wildcards are not supported")
        return (_FieldStep(node.fields[0]),)
    if isinstance(node, Index):
        if len(node.indices) != 1:
            raise SQLiteJSONPathError("Multiple array indices are not supported")
        index = node.indices[0]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise SQLiteJSONPathError("Only non-negative array indices are supported")
        return (_IndexStep(index),)
    if isinstance(node, Slice):
        if (node.start, node.end, node.step) == (None, None, None):
            return (_ArrayWildcardStep(),)
        raise SQLiteJSONPathError(
            "Array slices are not supported; use [*] or one index"
        )
    if isinstance(node, This):
        raise SQLiteJSONPathError("'this' is only supported inside a filter")
    if isinstance(node, FilterNode):
        expressions = tuple(node.expressions)
        if not expressions:
            raise SQLiteJSONPathError("Empty filters are not supported")
        for expression in expressions:
            if expression.op is None:
                raise SQLiteJSONPathError("Existence-only filters are not supported")
            if expression.op not in _COMPARISON_OPERATORS:
                raise SQLiteJSONPathError(
                    f"Unsupported comparison operator: {expression.op}"
                )
            _flatten_relative(expression.target)
            _validate_scalar(expression.value)
        return (_FilterStep(expressions),)
    _reject_unsupported_node(node)
    raise SQLiteJSONPathError(f"Unsupported JSONPath node: {node.__class__.__name__}")


def _reject_unsupported_node(node: JSONPath) -> None:
    if isinstance(
        node,
        (
            Descendants,
            Intersect,
            Parent,
            Union,
            Where,
            WhereNot,
        ),
    ):
        raise SQLiteJSONPathError(
            f"Unsupported JSONPath node: {node.__class__.__name__}"
        )


def _compile_step(
    step: Any,
    previous: str,
    current: str,
    parameters: list[Any],
) -> str:
    if isinstance(step, _FieldStep):
        parameters.extend((json.dumps(step.name, ensure_ascii=False), step.name))
        return (
            f"{current} AS (\n"
            f"        SELECT p.source_rowid, {_json_value('e')} AS value, e.type AS type,\n"
            "            p.path || '[' || ? || ']' AS path,\n"
            "            p.sort_key\n"
            f"        FROM {previous} p,\n"
            "            json_each(CASE WHEN p.type = 'object' THEN p.value END) e\n"
            "        WHERE e.key = ?\n"
            "    )"
        )

    if isinstance(step, _IndexStep):
        index = step.index
        parameters.extend((index, index))
        return (
            f"{current} AS (\n"
            f"        SELECT p.source_rowid, {_json_value('e')} AS value, e.type AS type,\n"
            "            p.path || '[' || ? || ']' AS path,\n"
            "            p.sort_key || printf('%012d', e.\"key\") AS sort_key\n"
            f"        FROM {previous} p,\n"
            "            json_each(CASE WHEN p.type = 'array' THEN p.value END) e\n"
            "        WHERE e.\"key\" = ?\n"
            "    )"
        )

    if isinstance(step, _ArrayWildcardStep):
        return (
            f"{current} AS (\n"
            f"        SELECT p.source_rowid, {_json_value('e')} AS value, e.type AS type,\n"
            "            p.path || '[' || e.\"key\" || ']' AS path,\n"
            "            p.sort_key || printf('%012d', e.\"key\") AS sort_key\n"
            f"        FROM {previous} p,\n"
            "            json_each(CASE WHEN p.type = 'array' THEN p.value END) e\n"
            "    )"
        )

    predicates = []
    for expression in step.expressions:
        target_steps = _flatten_relative(expression.target)
        predicates.append(_compile_predicate(expression, target_steps, parameters))

    filtered_path = (
        "CASE WHEN p.type = 'array' THEN '[' || e.\"key\" || ']' "
        f"ELSE '[' || {_quoted_json_string('e.key')} || ']' END"
    )
    filtered_sort_key = (
        "CASE WHEN p.type = 'array' THEN printf('%012d', e.\"key\") "
        "ELSE printf('%012d', e.id) END"
    )
    return (
        f"{current} AS (\n"
        f"        SELECT p.source_rowid, {_json_value('e')} AS value, e.type AS type,\n"
        f"            p.path || {filtered_path} AS path,\n"
        f"            p.sort_key || {filtered_sort_key} AS sort_key\n"
        f"        FROM {previous} p,\n"
        "            json_each(CASE WHEN p.type IN ('object', 'array') THEN p.value END) e\n"
        f"        WHERE {' AND '.join(predicates)}\n"
        "    )"
    )


def _compile_predicate(
    expression: Any,
    steps: tuple[Any, ...],
    parameters: list[Any],
) -> str:
    from_clause, final_alias = _compile_relative_join(steps, parameters)
    value = expression.value
    op = expression.op
    sql_op = "=" if op == "==" else op

    if value is None:
        if op not in ("==", "=", "!="):
            raise SQLiteJSONPathError("JSON null only supports equality comparisons")
        condition = f"{final_alias}.type = 'null'"
        if op == "!=":
            condition = f"NOT {condition}"
        return f"EXISTS (SELECT 1 {from_clause} WHERE {condition})"

    if isinstance(value, bool):
        if op not in ("==", "=", "!="):
            raise SQLiteJSONPathError("Booleans only support equality comparisons")
        type_name = "true" if value else "false"
        condition = f"{final_alias}.type = '{type_name}'"
        if op == "!=":
            condition = f"NOT ({condition})"
        return f"EXISTS (SELECT 1 {from_clause} WHERE {condition})"

    if isinstance(value, (int, float)):
        parameters.append(value)
        condition = (
            f"{final_alias}.type IN ('integer', 'real') "
            f"AND {final_alias}.atom {sql_op} ?"
        )
        return f"EXISTS (SELECT 1 {from_clause} WHERE {condition})"

    parameters.append(value)
    condition = (
        f"{final_alias}.type = 'text' AND {final_alias}.atom {sql_op} ?"
    )
    return f"EXISTS (SELECT 1 {from_clause} WHERE {condition})"


def _flatten_relative(node: JSONPath) -> tuple[Any, ...]:
    if isinstance(node, This):
        return ()
    if isinstance(node, Child):
        return _flatten_relative(node.left) + _flatten_relative(node.right)
    if isinstance(node, Fields):
        if len(node.fields) != 1 or node.fields[0] == "*":
            raise SQLiteJSONPathError("Filter targets support one named field only")
        return (_FieldStep(node.fields[0]),)
    if isinstance(node, Index):
        if len(node.indices) != 1:
            raise SQLiteJSONPathError("Filter targets support one array index only")
        index = node.indices[0]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise SQLiteJSONPathError("Filter targets support non-negative indices only")
        return (_IndexStep(index),)
    if isinstance(node, Slice) and (node.start, node.end, node.step) == (
        None,
        None,
        None,
    ):
        return (_ArrayWildcardStep(),)
    if isinstance(node, Root):
        raise SQLiteJSONPathError("Root-relative filter targets are not supported")
    _reject_unsupported_node(node)
    raise SQLiteJSONPathError(
        f"Unsupported filter target: {node.__class__.__name__}"
    )


def _compile_relative_join(
    steps: Sequence[Any],
    parameters: list[Any],
) -> tuple[str, str]:
    if not steps:
        return "", "e"

    from_parts = [
        "FROM (SELECT e.value AS value, e.type AS type, e.atom AS atom)"
    ]
    join_parts = []
    previous_value = "e.value"
    previous_type = "e.type"
    for position, step in enumerate(steps):
        alias = f"t{position}"
        if isinstance(step, _FieldStep):
            join_parts.append(
                "JOIN json_each(CASE WHEN "
                f"{previous_type} = 'object' THEN {previous_value} END) {alias}"
            )
            parameters.append(step.name)
            join_parts.append(f"ON {alias}.key = ?")
        else:
            join_parts.append(
                "JOIN json_each(CASE WHEN "
                f"{previous_type} = 'array' THEN {previous_value} END) {alias}"
            )
            if isinstance(step, _IndexStep):
                parameters.append(step.index)
                join_parts.append(f'ON {alias}."key" = ?')
        previous_value = f"{alias}.value"
        previous_type = f"{alias}.type"

    clause = " ".join(from_parts + join_parts)
    return clause, f"t{len(steps) - 1}"


def _validate_scalar(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    raise SQLiteJSONPathError(f"Unsupported scalar comparison value: {value!r}")


def _quote_identifier(identifier: str, kind: str) -> str:
    if not isinstance(identifier, str) or not identifier:
        raise SQLiteJSONPathError(f"{kind} name must be a non-empty string")
    if "\x00" in identifier:
        raise SQLiteJSONPathError(f"{kind} name must not contain NUL bytes")
    parts = identifier.split(".")
    if kind == "table":
        valid = 1 <= len(parts) <= 2 and all(
            _IDENTIFIER.fullmatch(part) for part in parts
        )
    else:
        valid = len(parts) == 1 and _IDENTIFIER.fullmatch(identifier)
    if not valid:
        raise SQLiteJSONPathError(f"Unsupported {kind} name: {identifier!r}")
    return ".".join(f'"{part}"' for part in parts)
