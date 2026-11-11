"""Compile JSONPath abstract syntax trees into SQLite JSON1 SQL.

The compiler walks a JSONPath AST and emits one ``WITH`` query whose common
table expressions mirror the path segments. Every match becomes a row with
the matched JSON value, its JSON1 type, and its normalized JSONPath
(``$["field"][0]`` form), so dots, quotes, and percent signs in field
names stay ordinary data. Only the standard library ``sqlite3`` module and
the JSON1 extension are needed; no ORM or database service is involved.
"""

from __future__ import annotations

import dataclasses
import typing

from jsonpath_ng.exceptions import SQLiteCompilationError
from jsonpath_ng.ext.arithmetic import Operation
from jsonpath_ng.ext.filter import And
from jsonpath_ng.ext.filter import Expression
from jsonpath_ng.ext.filter import Filter
from jsonpath_ng.ext.filter import Or
from jsonpath_ng.jsonpath import Child
from jsonpath_ng.jsonpath import Fields
from jsonpath_ng.jsonpath import Index
from jsonpath_ng.jsonpath import JSONPath
from jsonpath_ng.jsonpath import Root
from jsonpath_ng.jsonpath import This
from jsonpath_ng.jsonpath import Wildcard

SUPPORTED_SCOPE = """\
Supported JSONPath scope for SQLite JSON1 compilation:

* Root ($) and child (.) composition of the nodes below.
* Field selectors, including multiple field names and names that contain
  dots, quotes, or percent signs; the names are bound as parameters and
  never interpreted as path syntax.
* Array index selectors, including multiple indices and negative indices.
* Wildcards (* / [*]) over objects and arrays; scalars match nothing.
* Filter expressions [? ...] containing:
  - existence and negated existence tests;
  - scalar comparisons ==, =, !=, <, <=, >, >= against numbers, strings,
    booleans, and null literals;
  - logical && (and) and || (or), including nested and mixed groups.

Not supported and rejected before execution:

* recursive descent (..), slices, unions, intersections, parent,
  where/wherenot, arithmetic, sorting, and length/keys extensions;
* regular expression (=~) comparisons;
* comparisons whose right-hand side is another JSONPath query;
* filter targets that are non-singular (wildcards or filters) or that
  reference the root ($) instead of the current node (@).

Unsupported nodes raise SQLiteCompilationError; no condition is dropped.
"""


@dataclasses.dataclass(frozen=True)
class JSON1Compilation:
    """The result of compiling a JSONPath AST to SQLite JSON1 SQL.

    Attributes:
        sql: A complete SQL statement returning one row per match with the
            columns ``value``, ``path``, and ``type``. ``type`` is the JSON1
            type name (``object``, ``array``, ``integer``, ``real``,
            ``text``, ``true``, ``false``, or ``null``).
        parameters: The bind parameters for ``sql``, in order.
        scope: A description of the supported and rejected syntax.
    """

    sql: str
    parameters: tuple[typing.Any, ...]
    scope: str

    def execute(
        self, connection: typing.Any
    ) -> list[tuple[typing.Any, str, str | None]]:
        """Run the compiled query and return ``(value, path, type)`` rows."""

        return [
            (value, path, type_name)
            for value, path, type_name in connection.execute(
                self.sql, self.parameters
            )
        ]


def _quote_identifier(identifier: str) -> str:
    if not identifier or not identifier.replace("_", "").isalnum():
        raise SQLiteCompilationError(
            f"Unsupported SQL identifier {identifier!r}; "
            "use a plain table or column name."
        )
    return '"' + identifier.replace('"', '""') + '"'


def _unsupported(node: JSONPath) -> typing.Never:
    raise SQLiteCompilationError(
        f"JSONPath node {type(node).__name__} ({node!s}) cannot be compiled "
        "to SQLite JSON1 SQL."
    )


def compile_jsonpath(
    expression: JSONPath, table: str = "documents", column: str = "doc"
) -> JSON1Compilation:
    """Compile a JSONPath AST into a SQLite JSON1 query.

    Args:
        expression: A parsed JSONPath expression, for example from
            ``jsonpath_ng.ext.parser.parse()``.
        table: The table holding the raw JSON documents.
        column: The text column in ``table`` that stores each JSON document.

    Returns:
        A :class:`JSON1Compilation` with the SQL, bind parameters, and the
        supported-scope documentation.

    Raises:
        SQLiteCompilationError: If any AST node cannot be compiled. The
            whole expression is rejected; no condition is silently dropped.
    """

    table_identifier = _quote_identifier(table)
    column_identifier = _quote_identifier(column)
    source = f"{table_identifier}.{column_identifier}"

    context = _CompileContext()
    segments = _flatten_segments(expression)
    for segment in segments:
        context.emit(segment)

    ctes = [
        "s0 as (select rowid as rid, "
        f"{source} as node, "
        f"case when {_valid_json(source)} then json_type({source}) end as nt, "
        f"'$' as pp from {table_identifier})"
    ]
    ctes.extend(context.ctes)
    final_index = len(context.ctes)
    ctes.append(
        f"out as (select rid, node, nt, pp, "
        f"row_number() over (order by rid) as ord from s{final_index})"
    )
    # ``json_each`` yields booleans as the integers 1 and 0; reconstruct the
    # JSON text ``true``/``false`` so callers see the original JSON values.
    sql = (
        "with "
        + ",\n".join(ctes)
        + "\nselect case out.nt "
        "when 'true' then 'true' "
        "when 'false' then 'false' "
        "else out.node end as value, "
        "out.pp as path, out.nt as type "
        "from out order by ord"
    )
    return JSON1Compilation(
        sql=sql,
        parameters=tuple(context.parameters),
        scope=SUPPORTED_SCOPE,
    )


def _flatten_segments(expression: JSONPath) -> list[JSONPath]:
    """Turn a left-leaning Child chain into an ordered segment list."""

    if isinstance(expression, Root):
        return []
    if isinstance(expression, Child):
        return _flatten_segments(expression.left) + [expression.right]
    # A bare non-root expression (for example ``Fields('x')``) acts as if it
    # were rooted at the document.
    return [expression]


class _CompileContext:
    """Accumulates CTEs and bind parameters while walking the segments."""

    def __init__(self) -> None:
        self.ctes: list[str] = []
        self.parameters: list[typing.Any] = []

    @property
    def previous(self) -> str:
        return f"s{len(self.ctes)}"

    @property
    def current(self) -> str:
        return f"s{len(self.ctes) + 1}"

    def emit(self, node: JSONPath) -> None:
        previous = self.previous
        current = self.current
        if isinstance(node, Fields):
            body = self._fields_cte(node, previous)
        elif isinstance(node, Wildcard):
            body = self._wildcard_cte(previous)
        elif isinstance(node, Index):
            body = self._index_cte(node, previous)
        elif isinstance(node, Filter):
            body = self._filter_cte(node, previous)
        else:
            _unsupported(node)
        self.ctes.append(f"{current} as ({body})")

    def _fields_cte(self, node: Fields, previous: str) -> str:
        if node._deprecated_wildcard_use:
            _unsupported(node)
        if len(node.fields) == 1:
            self.parameters.append(node.fields[0])
            return (
                f"select {previous}.rid, e.value as node, e.type as nt, "
                f"{previous}.pp || '[' || json_quote(e.key) || ']' as pp "
                f"from {previous}, json_each(case when "
                f"{_is_type(f'{previous}.node', 'object')} "
                f"then {previous}.node end) as e "
                "where e.key = ?"
            )
        # Multiple fields preserve the selector order, not object order.
        branches = []
        for ordinal, field in enumerate(node.fields):
            self.parameters.append(field)
            branches.append(f"select ? as key, {ordinal} as ordinal")
        selector = " union all ".join(branches)
        return (
            f"select {previous}.rid, e.value as node, e.type as nt, "
            f"{previous}.pp || '[' || json_quote(e.key) || ']' as pp "
            f"from {previous}, json_each(case when "
            f"{_is_type(f'{previous}.node', 'object')} "
            f"then {previous}.node end) as e "
            f"join ({selector}) as sel on e.key = sel.key "
            "order by sel.ordinal"
        )

    def _wildcard_cte(self, previous: str) -> str:
        return (
            f"select {previous}.rid, e.value as node, e.type as nt, "
            f"case when {_is_type(f'{previous}.node', 'array')} "
            f"then {previous}.pp || '[' || e.key || ']' "
            f"else {previous}.pp || '[' || json_quote(e.key) || ']' "
            "end as pp "
            f"from {previous}, json_each(case when "
            f"{_is_type(f'{previous}.node', 'object', 'array')} "
            f"then {previous}.node end) as e"
        )

    def _index_cte(self, node: Index, previous: str) -> str:
        if not node.indices:
            raise SQLiteCompilationError(
                "An Index selector needs at least one index."
            )
        clauses = []
        for index in node.indices:
            if index >= 0:
                clauses.append("e.key = ?")
            else:
                clauses.append(
                    f"e.key = json_array_length({previous}.node) + ?"
                )
        # Each index appears twice in the text (ordinal CASE and WHERE),
        # so append the parameters in the same order as the placeholders.
        for index in node.indices:
            self.parameters.append(index)
        ordinal_cases = " ".join(
            f"when {clause} then {ordinal}"
            for ordinal, clause in enumerate(clauses)
        )
        for index in node.indices:
            self.parameters.append(index)
        return (
            f"select {previous}.rid, e.value as node, e.type as nt, "
            f"{previous}.pp || '[' || e.key || ']' as pp, "
            f"case {ordinal_cases} end as sel_ordinal "
            f"from {previous}, json_each(case when "
            f"{_is_type(f'{previous}.node', 'array')} "
            f"then {previous}.node end) as e "
            f"where {' or '.join(clauses)} "
            "order by sel_ordinal"
        )

    def _filter_cte(self, node: Filter, previous: str) -> str:
        # A filter selects among the current node's children (object members
        # or array elements), mirroring ``Filter.find()``: each child is a
        # candidate and survives only when every expression matches it.
        predicates = [
            self._compile_filter_expression(expression, "e.value")
            for expression in node.expressions
        ]
        where = " and ".join(f"({predicate})" for predicate in predicates)
        return (
            f"select {previous}.rid, e.value as node, e.type as nt, "
            f"case when {_is_type(f'{previous}.node', 'array')} "
            f"then {previous}.pp || '[' || e.key || ']' "
            f"else {previous}.pp || '[' || json_quote(e.key) || ']' "
            "end as pp "
            f"from {previous}, json_each(case when "
            f"{_is_type(f'{previous}.node', 'object', 'array')} "
            f"then {previous}.node end) as e where {where}"
        )

    def _compile_filter_expression(
        self, expression: JSONPath, candidate: str
    ) -> str:
        if isinstance(expression, Filter):
            inner = [
                self._compile_filter_expression(item, candidate)
                for item in expression.expressions
            ]
            return "(" + " and ".join(inner) + ")"
        if isinstance(expression, And):
            inner = [
                self._compile_filter_expression(item, candidate)
                for item in expression.operands
            ]
            return "(" + " and ".join(inner) + ")"
        if isinstance(expression, Or):
            inner = [
                self._compile_filter_expression(item, candidate)
                for item in expression.operands
            ]
            return "(" + " or ".join(inner) + ")"
        if isinstance(expression, Expression):
            return self._compile_comparison(expression, candidate)
        _unsupported(expression)

    def _compile_comparison(
        self, expression: Expression, candidate: str
    ) -> str:
        target_path = self._relative_json_path(expression.target)
        # The path is embedded as a single-quoted SQL string literal, so any
        # single quote in a field name must be doubled to avoid breaking it.
        sql_path = target_path.replace("'", "''")

        # ``json_type()`` and ``json_extract()`` raise "malformed JSON" for
        # non-JSON text (for example a bare string candidate), so every
        # call is guarded by ``json_valid()`` first.
        type_call = (
            f"case when {_valid_json(candidate)} "
            f"then json_type({candidate}, '{sql_path}') end"
        )
        value_call = (
            f"case when {_valid_json(candidate)} "
            f"then json_extract({candidate}, '{sql_path}') end"
        )

        if expression.op == "!":
            return f"{type_call} is null"
        if expression.op is None:
            return f"{type_call} is not null"

        if expression.op == "=~":
            raise SQLiteCompilationError(
                "Regular expression (=~) filters cannot be compiled to "
                "SQLite JSON1 SQL."
            )
        if isinstance(expression.value, Operation):
            raise SQLiteCompilationError(
                "Arithmetic inside filters cannot be compiled to "
                "SQLite JSON1 SQL."
            )
        if isinstance(expression.value, JSONPath):
            raise SQLiteCompilationError(
                "Filters comparing one JSONPath query against another "
                "cannot be compiled to SQLite JSON1 SQL."
            )

        literal = expression.value

        if literal is None:
            if expression.op in ("==", "="):
                return f"{type_call} = 'null'"
            if expression.op == "!=":
                # A missing member is not equal to null, so it matches.
                return f"coalesce({type_call} != 'null', 1)"
            raise SQLiteCompilationError(
                f"Operator {expression.op!r} cannot be applied to null "
                "in a SQLite JSON1 filter."
            )

        if isinstance(literal, bool):
            expected_type = "true" if literal else "false"
            if expression.op in ("==", "="):
                return f"{type_call} = '{expected_type}'"
            if expression.op == "!=":
                # A missing member is not equal to a boolean, so it matches.
                return f"coalesce({type_call} != '{expected_type}', 1)"
            raise SQLiteCompilationError(
                f"Operator {expression.op!r} cannot be applied to booleans "
                "in a SQLite JSON1 filter."
            )

        if isinstance(literal, (int, float)):
            type_guard = f"{type_call} in ('integer', 'real')"
            self.parameters.append(literal)
            return self._scalar_clause(expression.op, type_guard, value_call)

        if isinstance(literal, str):
            type_guard = f"{type_call} = 'text'"
            self.parameters.append(literal)
            return self._scalar_clause(expression.op, type_guard, value_call)

        raise SQLiteCompilationError(
            f"Filter literal {literal!r} cannot be compiled to "
            "SQLite JSON1 SQL."
        )

    def _scalar_clause(
        self,
        op: str,
        type_guard: str,
        value_call: str,
    ) -> str:
        sql_op = "=" if op == "==" else op
        if op == "!=":
            # RFC 9535: ``a != b`` is true whenever ``a == b`` is false.
            # A missing member, a null, or a value of a different JSON type
            # can never equal the literal, so they all match. Only values of
            # the literal's own type can be equal, and then the value must
            # compare different.
            return f"coalesce(not ({type_guard} and {value_call} = ?), 1)"
        if op in ("<", "<=", ">", ">=", "==", "="):
            return f"{type_guard} and {value_call} {sql_op} ?"
        raise SQLiteCompilationError(
            f"Filter operator {op!r} cannot be compiled to SQLite JSON1 SQL."
        )

    def _relative_json_path(self, target: JSONPath) -> str:
        """Compile a singular @-relative filter target to a JSON1 path."""

        segments = self._relative_segments(target)
        path = "$"
        for segment in segments:
            if isinstance(segment, Fields):
                if len(segment.fields) != 1:
                    raise SQLiteCompilationError(
                        "Filter targets must be singular; multiple field "
                        f"names in {segment!s} are not supported."
                    )
                # SQLite JSON1 paths quote fields with dot notation
                # (``$."name"``); bracket notation like ``$["name"]`` is
                # not part of the JSON1 path grammar.
                path += "." + _json_string(segment.fields[0])
            elif isinstance(segment, Index):
                if len(segment.indices) != 1:
                    raise SQLiteCompilationError(
                        "Filter targets must be singular; multiple indices "
                        f"in {segment!s} are not supported."
                    )
                index = segment.indices[0]
                if index < 0:
                    raise SQLiteCompilationError(
                        "Negative indices inside filter targets cannot be "
                        "compiled to SQLite JSON1 SQL."
                    )
                path += f"[{index}]"
            else:
                _unsupported(segment)
        return path

    def _relative_segments(self, target: JSONPath) -> list[JSONPath]:
        if isinstance(target, This):
            return []
        if isinstance(target, Child):
            if isinstance(target.left, Root):
                raise SQLiteCompilationError(
                    "Filters referencing the root ($) cannot be compiled "
                    "to SQLite JSON1 SQL."
                )
            return self._relative_segments(target.left) + [target.right]
        _unsupported(target)


def _json_string(value: str) -> str:
    """Encode a field name as a double-quoted JSON string for JSON1 paths."""

    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _valid_json(expression: str) -> str:
    """Guard a JSON1 call against non-JSON text values.

    ``json_type()`` raises "malformed JSON" for a bare string such as
    ``alpha``; filtering with ``json_valid()`` first makes such values
    behave like non-matching scalars instead of aborting the query.
    """

    return f"json_valid({expression})"


def _is_type(expression: str, *types: str) -> str:
    type_list = ", ".join(f"'{type_}'" for type_ in types)
    return (
        f"({_valid_json(expression)} "
        f"and json_type({expression}) in ({type_list}))"
    )
