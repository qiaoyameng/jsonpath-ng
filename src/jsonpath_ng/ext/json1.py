"""
SQLite JSON1 compilation for jsonpath-ng.

This module exposes an explicit entry point, :func:`compile_json1`, that
compiles a subset of the jsonpath-ng AST into an executable SQLite JSON1
query.  It is aimed at documents stored verbatim in a SQLite ``TEXT`` column:
the database performs the structural filtering, and only the matched nodes
leave the database.

The compiler never approximates an AST it does not understand.  Nodes that
cannot be translated raise :class:`~jsonpath_ng.exceptions.JSON1CompileError`
before any SQL runs, so hard conditions are never silently discarded.

Supported in the first version:

* ``Root``/``This`` boundaries and ``Child`` chains.
* Object fields, including names containing dots, quotes, ``%`` and other
  punctuation.  Field names are always bound as data, never inlined as SQL.
* Array indices (including negative indices) and the ``*`` wildcard.
* Filter selectors whose predicates are scalar comparisons combined with
  ``&&``/``||``, plus existence and negated-existence tests.

Not supported in the first version (they raise ``JSON1CompileError``):
slices, recursive descent, unions/intersections, ``where``/``parent``,
multi-field or multi-index selectors, filters that compare two queries,
reference ``$`` from inside a filter, regular expressions, arithmetic and
sorting extensions.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import typing

from jsonpath_ng.exceptions import JSON1CompileError
from jsonpath_ng.ext.filter import And
from jsonpath_ng.ext.filter import Expression
from jsonpath_ng.ext.filter import Filter
from jsonpath_ng.ext.filter import Or
from jsonpath_ng.jsonpath import Child
from jsonpath_ng.jsonpath import Descendants
from jsonpath_ng.jsonpath import Fields
from jsonpath_ng.jsonpath import Index
from jsonpath_ng.jsonpath import JSONPath
from jsonpath_ng.jsonpath import Parent
from jsonpath_ng.jsonpath import Root
from jsonpath_ng.jsonpath import Slice
from jsonpath_ng.jsonpath import This
from jsonpath_ng.jsonpath import Where
from jsonpath_ng.jsonpath import WhereNot
from jsonpath_ng.jsonpath import Wildcard
from jsonpath_ng.jsonpath import _Intersect
from jsonpath_ng.jsonpath import _Union

# Width of one path component inside the lexicographic ordering key.
# json_each.id is a byte offset, so zero-padding to this fixed width makes
# string concatenation reproduce document traversal order.
_ORDER_WIDTH = 12
_ORDER_PAD = "0" * _ORDER_WIDTH

_CONTAINER_TYPES = "('array','object')"
_NUMBER_TYPES = "('integer','real')"


@dataclasses.dataclass(frozen=True)
class CompiledJSON1Query:
    """The result of compiling a JSONPath AST to SQLite JSON1.

    Attributes:
        sql: The complete, ready-to-execute SQL statement.  It selects one
            row per matched node with columns ``doc_id``, ``value``,
            ``path`` (a JSON array of normalized path components) and the
            internal ``ord`` ordering key.  Canonical bracket-notation
            paths are rendered client-side by :meth:`execute`.
        fragment: The recursive CTE body, without the final ``SELECT``,
            for callers that want to embed the traversal in a larger query.
        bindings: The positional parameters that must be bound to ``sql``.
        path: The original JSONPath expression, when one was supplied.
        supported_scope: A human-readable description of the supported
            subset, useful for documentation and error messages.
    """

    sql: str
    fragment: str
    bindings: tuple[typing.Any, ...]
    path: str | None
    supported_scope: str

    def execute(
        self,
        connection: sqlite3.Connection,
        table: str = "documents",
        document_column: str = "document",
        id_column: str = "id",
    ) -> list[tuple[typing.Any, list[typing.Any], str]]:
        """Run the compiled query and decode matched nodes.

        The returned tuples are ``(value, path_components, canonical_path)``.
        Booleans and numbers are distinguished, and JSON ``null`` becomes
        Python ``None``.  Rows are ordered by document id and then by the
        original traversal order of the path inside each document.
        """

        sql = _substitute_relation(self.sql, table, document_column, id_column)
        rows = connection.execute(sql, self.bindings).fetchall()
        results: list[tuple[typing.Any, list[typing.Any], str]] = []
        for _, value_text, path_text, _order_key in rows:
            value = json.loads(value_text)
            components = json.loads(path_text)
            canonical = render_canonical_path(components)
            results.append((value, components, canonical))
        return results


SUPPORTED_SCOPE = (
    "fields (single), array indices including negative indices, the * "
    "wildcard, and filters with existence checks, negation, scalar "
    "comparisons and &&/|| combinations"
)


def _quote_identifier(identifier: str) -> str:
    """Quote a SQL identifier (table or column name)."""

    if not isinstance(identifier, str) or not identifier:
        msg = f"SQL identifiers must be non-empty strings, got {identifier!r}"
        raise JSON1CompileError(msg)
    return '"' + identifier.replace('"', '""') + '"'


def _substitute_relation(
    sql: str, table: str, document_column: str, id_column: str
) -> str:
    table_ref = _quote_identifier(table)
    relation = (
        f"{table_ref} AS d "
        f"JOIN json_each(d.{_quote_identifier(document_column)}) AS e "
        f"ON 1=1"
    )
    return (
        sql.replace("/*__relation__*/", relation)
        .replace("/*__table__*/", table_ref)
        .replace("/*__id__*/", _quote_identifier(id_column))
        .replace("/*__doc__*/", _quote_identifier(document_column))
    )


def _escape_json1_field(field: str) -> str:
    """Escape a member name for use inside a double-quoted JSON1 path."""

    return field.replace("\\", "\\\\").replace('"', '\\"')


def _json1_field_path(field: str) -> str:
    return '$."' + _escape_json1_field(field) + '"'


def _flatten_main_path(ast: JSONPath) -> list[JSONPath]:
    """Flatten a ``Child`` chain into its ordered segments."""

    segments: list[JSONPath] = []
    node: JSONPath | None = ast
    while isinstance(node, Child):
        segments.append(node.right)
        node = node.left
    if isinstance(node, Descendants):
        msg = (
            "Recursive descent (..) is not supported in the first JSON1 version"
        )
        raise JSON1CompileError(msg)
    if not isinstance(node, Root):
        msg = (
            "JSON1 compilation requires the main path to start at the root "
            f"($); found {type(node).__name__}"
        )
        raise JSON1CompileError(msg)
    segments.reverse()
    return segments


def _ensure_singular_fields(node: Fields) -> str:
    if node._deprecated_wildcard_use:
        msg = "The deprecated Fields('*') wildcard is not supported; use Wildcard()"
        raise JSON1CompileError(msg)
    if len(node.fields) != 1:
        msg = (
            "Multi-field selectors are not supported in the first JSON1 "
            f"version; got {len(node.fields)} fields"
        )
        raise JSON1CompileError(msg)
    return node.fields[0]


def _ensure_singular_index(node: Index) -> int:
    if len(node.indices) != 1:
        msg = (
            "Multi-index selectors are not supported in the first JSON1 "
            f"version; got {len(node.indices)} indices"
        )
        raise JSON1CompileError(msg)
    return node.indices[0]


def _reject_unsupported(node: JSONPath) -> typing.Never:
    name = type(node).__name__
    msg = (
        f"JSONPath node {name} is outside the supported JSON1 subset "
        f"({SUPPORTED_SCOPE})"
    )
    raise JSON1CompileError(msg)

def _relative_field_path(field: str) -> str:
    return '"' + _escape_json1_field(field) + '"'


def _compile_relative_target(
    target: JSONPath, candidate: str
) -> tuple[str, list[typing.Any]]:
    """Compile a singular, ``@``-rooted filter target to a JSON1 path.

    Returns the SQL expression that yields the JSON1 path (relative to the
    candidate value) and any bindings it needs.  Positive paths are bound as
    constants; negative array indices are computed from the live array
    length because SQLite JSON1 has no negative-index syntax.
    """

    segments = _flatten_relative_chain(target)
    bindings: list[typing.Any] = []
    container_guard = (
        f"typeof({candidate}) = 'text' AND json_valid({candidate}) = 1"
    )
    parts: list[str] = []
    negative_index: int | None = None
    negative_part_position: int | None = None

    for segment in segments:
        if isinstance(segment, Fields):
            parts.append("." + _relative_field_path(_ensure_singular_fields(segment)))
        elif isinstance(segment, Index):
            index = _ensure_singular_index(segment)
            if index >= 0:
                parts.append(f"[{index}]")
            else:
                if negative_index is not None:
                    msg = (
                        "Multiple negative indices in one filter target are "
                        "not supported in the first JSON1 version"
                    )
                    raise JSON1CompileError(msg)
                negative_index = index
                negative_part_position = len(parts)
                parts.append(f"[{index}]")
        else:
            _reject_unsupported(segment)

    if negative_index is None:
        return _sql_string("$" + "".join(parts)), bindings

    # Split the path at the negative index.  The array prefix is resolved at
    # runtime so a short or empty array yields a missing path instead of an
    # invalid JSON1 path error.  Any segments after the index (for example
    # the ".v" in "@.items[-1].v") are appended to the computed path.
    assert negative_part_position is not None
    neg_position = negative_part_position
    prefix_parts = parts[:neg_position]
    suffix_parts = parts[neg_position + 1 :]
    prefix = "$" + "".join(prefix_parts)
    prefix_literal = _sql_string(prefix)
    length = f"json_array_length({candidate}, {prefix_literal})"
    is_array = (
        f"({container_guard} AND "
        f"json_type({candidate}, {prefix_literal}) = 'array')"
    )
    suffix = "".join(suffix_parts)
    computed = (
        "(CASE WHEN "
        f"{is_array} AND {length} >= {-negative_index} "
        f"THEN {_sql_string(prefix + '[')} || ({length} {negative_index:+d}) "
        f"|| ']' || {_sql_string(suffix)} "
        "ELSE '$.__json1_compiler_missing__' END)"
    )
    return computed, bindings


def _flatten_relative_chain(target: JSONPath) -> list[JSONPath]:
    if isinstance(target, This):
        return []
    segments: list[JSONPath] = []
    node: JSONPath | None = target
    while isinstance(node, Child):
        segments.append(node.right)
        node = node.left
    if not isinstance(node, This):
        if isinstance(node, Root):
            msg = (
                "Filters referencing the root document ($) are not supported "
                "in the first JSON1 version"
            )
            raise JSON1CompileError(msg)
        _reject_unsupported(node)
    segments.reverse()
    return segments


def _sql_string(value: str) -> str:
    """Render a literal string for inlining inside generated SQL.

    Only compiler-produced strings (JSON1 paths) ever reach this function;
    user data continues to travel through bindings.
    """

    return "'" + value.replace("'", "''") + "'"


def _target_expressions(
    target: JSONPath, candidate: str
) -> tuple[str, str, list[typing.Any]]:
    """Return ``(type_sql, value_sql, bindings)`` for a filter target.

    ``type_sql`` yields the SQLite JSON1 type name (or NULL when missing),
    and ``value_sql`` yields the scalar value.  Booleans collapse to 0/1 via
    ``json_extract``, so the type expression is what keeps them distinct.
    """

    # A bare current node (`@`) refers to the candidate itself.  Its own
    # type marker is reliable for every candidate, including scalars.
    if isinstance(target, This):
        candidate_type = candidate.replace(".value", ".type")
        return candidate_type, candidate, []

    path_sql, bindings = _compile_relative_target(target, candidate)
    # Only objects/arrays carry parseable JSON text in the value column; a
    # scalar candidate (bare number/string) would make json_type() raise.
    candidate_type = candidate.replace(".value", ".type")
    container_guard = f"{candidate_type} IN {_CONTAINER_TYPES}"
    type_sql = (
        "CASE WHEN "
        f"{container_guard} THEN json_type({candidate}, {path_sql}) END"
    )
    value_sql = (
        "CASE WHEN "
        f"{container_guard} THEN json_extract({candidate}, {path_sql}) END"
    )
    return type_sql, value_sql, bindings


def _compile_predicate(
    node: JSONPath, candidate: str
) -> tuple[str, list[typing.Any]]:
    if isinstance(node, And):
        pieces: list[str] = []
        bindings: list[typing.Any] = []
        for operand in node.operands:
            piece, piece_bindings = _compile_predicate(operand, candidate)
            pieces.append(piece)
            bindings.extend(piece_bindings)
        return "(" + " AND ".join(pieces) + ")", bindings
    if isinstance(node, Or):
        pieces = []
        bindings = []
        for operand in node.operands:
            piece, piece_bindings = _compile_predicate(operand, candidate)
            pieces.append(piece)
            bindings.extend(piece_bindings)
        return "(" + " OR ".join(pieces) + ")", bindings
    if isinstance(node, Expression):
        return _compile_expression(node, candidate)
    msg = (
        "Only comparison expressions combined with &&/|| are supported in "
        f"filters; found {type(node).__name__}"
    )
    raise JSON1CompileError(msg)


def _compile_expression(
    expression: Expression, candidate: str
) -> tuple[str, list[typing.Any]]:
    if isinstance(expression.value, JSONPath):
        msg = (
            "Filters comparing one query against another are not supported "
            "in the first JSON1 version"
        )
        raise JSON1CompileError(msg)

    type_sql, value_sql, bindings = _target_expressions(
        expression.target, candidate
    )
    op = expression.op

    # A bare current-node test (`@`) always matches, even for scalars, null,
    # false or empty containers; its negation therefore never matches.
    if isinstance(expression.target, This):
        if op is None:
            return "(1)", bindings
        if op == "!":
            return "(0)", bindings

    # Existence test: ``@.x`` (op is None) and negated existence ``!@.x``.
    if op is None:
        return f"{type_sql} IS NOT NULL", bindings
    if op == "!":
        return f"{type_sql} IS NULL", bindings
    if op == "=~":
        msg = "Regular-expression filters (=~) are not supported in JSON1 compilation"
        raise JSON1CompileError(msg)

    literal = expression.value
    exists = f"{type_sql} IS NOT NULL"

    if literal is None:
        return _null_predicate(op, type_sql, exists), bindings
    if isinstance(literal, bool):
        return _bool_predicate(op, type_sql, literal), bindings
    if isinstance(literal, (int, float)):
        return _number_predicate(op, type_sql, value_sql, literal, bindings), bindings
    if isinstance(literal, str):
        return _string_predicate(op, type_sql, value_sql, literal, bindings), bindings
    msg = f"Unsupported filter literal type: {type(literal).__name__}"
    raise JSON1CompileError(msg)


def _null_predicate(op: str, type_sql: str, exists: str) -> str:
    is_null = f"{type_sql} = 'null'"
    if op in ("==", "=", "<=", ">="):
        # Equality-based operators on JSON null reduce to equality; missing
        # never equals an explicit null literal.
        return f"({exists} AND {is_null})"
    if op == "!=":
        # RFC 9535: != is the negation of ==, so a missing member (NULL
        # type) is "not equal to" the literal and matches.
        return f"COALESCE(NOT ({exists} AND {is_null}), 1)"
    # Ordering against null is never true.
    return "(0)"


def _bool_predicate(op: str, type_sql: str, literal: bool) -> str:
    type_name = "true" if literal else "false"
    same_type = f"{type_sql} = '{type_name}'"
    if op in ("==", "=", "<=", ">="):
        return f"({same_type})"
    if op == "!=":
        # Missing members and every non-matching boolean are "not equal".
        return f"COALESCE(NOT ({same_type}), 1)"
    # Booleans have no ordering, so < and > are always false.
    return "(0)"


def _number_predicate(
    op: str,
    type_sql: str,
    value_sql: str,
    literal: int | float,
    bindings: list[typing.Any],
) -> str:
    is_number = f"{type_sql} IN {_NUMBER_TYPES}"
    if op in ("==", "="):
        bindings.append(literal)
        return f"({is_number} AND {value_sql} = ?)"
    if op == "!=":
        bindings.append(literal)
        # Missing or non-number values and unequal numbers are "not equal".
        return f"COALESCE(NOT (({is_number}) AND {value_sql} = ?), 1)"
    if op in ("<", "<=", ">", ">="):
        sql_op = op
        bindings.append(literal)
        return f"({is_number} AND {value_sql} {sql_op} ?)"
    msg = f"Unsupported operator {op!r}"
    raise JSON1CompileError(msg)


def _string_predicate(
    op: str,
    type_sql: str,
    value_sql: str,
    literal: str,
    bindings: list[typing.Any],
) -> str:
    is_text = f"{type_sql} = 'text'"
    if op in ("==", "="):
        bindings.append(literal)
        return f"({is_text} AND {value_sql} = ?)"
    if op == "!=":
        bindings.append(literal)
        # Missing or non-text values and unequal strings are "not equal".
        return f"COALESCE(NOT (({is_text}) AND {value_sql} = ?), 1)"
    if op in ("<", "<=", ">", ">="):
        bindings.append(literal)
        return f"({is_text} AND {value_sql} {op} ?)"
    msg = f"Unsupported operator {op!r}"
    raise JSON1CompileError(msg)


def _safe_json(column: str) -> str:
    """Expand ``column`` only for array/object rows.

    The decision uses the row's carried JSON1 type marker rather than calling
    ``json_type()`` on the value: a value column can hold a bare SQL string
    that is not itself JSON, which would make ``json_type()`` raise
    "malformed JSON".
    """

    type_column = column.replace(".val", ".typ")
    return (
        f"CASE WHEN {type_column} IN {_CONTAINER_TYPES} "
        f"THEN {column} ELSE 'null' END"
    )


def _order_term() -> str:
    return f"printf('%0{_ORDER_WIDTH}d', e.id)"


def _base_cte(bindings: list[typing.Any], first_field: str) -> str:
    # One binding seeds the path array, the other constrains the member key.
    bindings.append(first_field)
    bindings.append(first_field)
    return (
        "WITH s1 AS (\n"
        "  SELECT d./*__id__*/ AS doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_array(?) AS path, {_order_term()} AS ord\n"
        "  FROM /*__relation__*/\n"
        "  WHERE json_type(d./*__doc__*/) = 'object' AND e.key = ?\n"
        ")"
    )


def _field_step(
    step_index: int,
    previous: str,
    field: str,
    bindings: list[typing.Any],
) -> str:
    bindings.append(field)
    return (
        f", s{step_index} AS (\n"
        f"  SELECT {previous}.doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_insert({previous}.path, '$[#]', e.key) AS path,\n"
        f"    {previous}.ord || {_order_term()} AS ord\n"
        f"  FROM {previous}, json_each({_safe_json(previous + '.val')}) e\n"
        f"  WHERE {previous}.typ = 'object' AND e.key = ?\n"
        ")"
    )


def _wildcard_step(step_index: int, previous: str) -> str:
    return (
        f", s{step_index} AS (\n"
        f"  SELECT {previous}.doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_insert({previous}.path, '$[#]', e.key) AS path,\n"
        f"    {previous}.ord || {_order_term()} AS ord\n"
        f"  FROM {previous}, json_each({_safe_json(previous + '.val')}) e\n"
        f"  WHERE {previous}.typ IN {_CONTAINER_TYPES}\n"
        ")"
    )


def _index_step(
    step_index: int,
    previous: str,
    index: int,
    bindings: list[typing.Any],
) -> str:
    array_length = (
        f"CASE WHEN {previous}.typ = 'array' "
        f"THEN json_array_length({previous}.val) ELSE 0 END"
    )
    if index >= 0:
        bindings.append(index)
        key_filter = "e.key = ?"
    else:
        # Resolve the negative position against the live array length.
        key_filter = (
            f"e.key = {array_length} {index:+d} AND {array_length} >= {-index}"
        )
    return (
        f", s{step_index} AS (\n"
        f"  SELECT {previous}.doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_insert({previous}.path, '$[#]', e.key) AS path,\n"
        f"    {previous}.ord || {_order_term()} AS ord\n"
        f"  FROM {previous}, json_each({_safe_json(previous + '.val')}) e\n"
        f"  WHERE {previous}.typ = 'array' AND {key_filter}\n"
        ")"
    )


def _filter_step(
    step_index: int,
    previous: str,
    node: Filter,
    bindings: list[typing.Any],
) -> str:
    predicates: list[str] = []
    for expression in node.expressions:
        predicate, predicate_bindings = _compile_predicate(expression, "e.value")
        predicates.append(predicate)
        bindings.extend(predicate_bindings)
    where = " AND ".join(predicates)
    return (
        f", s{step_index} AS (\n"
        f"  SELECT {previous}.doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_insert({previous}.path, '$[#]', e.key) AS path,\n"
        f"    {previous}.ord || {_order_term()} AS ord\n"
        f"  FROM {previous}, json_each({_safe_json(previous + '.val')}) e\n"
        f"  WHERE {previous}.typ IN {_CONTAINER_TYPES}"
        + (f" AND {where}" if where else "")
        + "\n)"
    )


def _leaf_value_sql(cte_name: str) -> str:
    """Render the matched value as lossless JSON text.

    ``json_extract`` turns JSON booleans into integers, so boolean leaves are
    re-serialized from their type marker before the result leaves SQLite.
    """

    typ = f"{cte_name}.typ"
    val = f"{cte_name}.val"
    return (
        "CASE "
        f"WHEN {typ} = 'true' THEN 'true' "
        f"WHEN {typ} = 'false' THEN 'false' "
        f"WHEN {typ} = 'null' THEN 'null' "
        f"WHEN {typ} IN ('array','object') THEN {val} "
        f"ELSE json_quote({val}) "
        "END"
    )


def render_canonical_path(components: list[typing.Any]) -> str:
    """Render normalized path components as bracket-notation JSONPath."""

    rendered = ["$"]
    for component in components:
        if isinstance(component, bool) or not isinstance(component, int):
            member = str(component).replace("\\", "\\\\").replace('"', '\\"')
            rendered.append('["' + member + '"]')
        else:
            rendered.append(f"[{component}]")
    return "".join(rendered)


def compile_json1(
    query: JSONPath | str,
    *,
    parser: typing.Callable[[str], JSONPath] | None = None,
) -> CompiledJSON1Query:
    """Compile a JSONPath expression or AST to a SQLite JSON1 query.

    Args:
        query: A parsed JSONPath AST or an expression string.  Strings are
            parsed with the extended jsonpath-ng parser by default.
        parser: Optional callable used to parse string queries.

    Returns:
        A :class:`CompiledJSON1Query` holding the SQL, positional bindings
        and the supported-scope description.

    Raises:
        JSON1CompileError: If any part of the AST is outside the supported
            subset.  The error is raised before execution, so unsupported
            conditions are never dropped.
    """

    original_text: str | None = None
    if isinstance(query, str):
        original_text = query
        if parser is None:
            from jsonpath_ng.ext.parser import parse as default_parser

            parser = default_parser
        ast = parser(query)
    else:
        ast = query

    segments = _flatten_main_path(ast)
    bindings: list[typing.Any] = []
    cte_parts: list[str] = []
    step_index = 0
    previous = "d"

    if not segments:
        # Bare "$": match each whole document.
        fragment = (
            "WITH s1 AS (\n"
            "  SELECT d./*__id__*/ AS doc_id, d./*__doc__*/ AS val,\n"
            "    json_type(d./*__doc__*/) AS typ,\n"
            "    json_array() AS path, " + _ORDER_PAD + " AS ord\n"
            "  FROM /*__table__*/ AS d\n"
            ")"
        )
        cte_parts.append(fragment)
        step_index = 1
        previous = "s1"

    for position, segment in enumerate(segments):
        if isinstance(segment, Fields):
            field = _ensure_singular_fields(segment)
            if step_index == 0:
                cte_parts.append(_base_cte(bindings, field))
            else:
                cte_parts.append(
                    _field_step(step_index + 1, previous, field, bindings)
                )
        elif isinstance(segment, Wildcard):
            if step_index == 0:
                cte_parts.append(_root_wildcard_cte())
            else:
                cte_parts.append(_wildcard_step(step_index + 1, previous))
        elif isinstance(segment, Index):
            index = _ensure_singular_index(segment)
            if step_index == 0:
                cte_parts.append(_root_index_cte(index, bindings))
            else:
                cte_parts.append(
                    _index_step(step_index + 1, previous, index, bindings)
                )
        elif isinstance(segment, Filter):
            if step_index == 0:
                cte_parts.append(_root_filter_cte(segment, bindings))
            else:
                cte_parts.append(
                    _filter_step(step_index + 1, previous, segment, bindings)
                )
        elif isinstance(segment, Slice):
            msg = "Array slices are not supported in the first JSON1 version"
            raise JSON1CompileError(msg)
        elif isinstance(segment, Descendants):
            msg = "Recursive descent (..) is not supported in the first JSON1 version"
            raise JSON1CompileError(msg)
        elif isinstance(segment, (Where, WhereNot, Parent, _Union, _Intersect)):
            _reject_unsupported(segment)
        else:
            _reject_unsupported(segment)
        step_index += 1
        previous = f"s{step_index}"

    fragment = "\n".join(cte_parts)
    final_cte = previous
    value_sql = _leaf_value_sql(final_cte)
    sql = (
        fragment
        + f"\nSELECT {final_cte}.doc_id, {value_sql} AS value,\n"
        + f"  {final_cte}.path AS path, {final_cte}.ord AS ord\n"
        + f"FROM {final_cte}\n"
        + "ORDER BY doc_id, ord"
    )
    return CompiledJSON1Query(
        sql=sql,
        fragment=fragment,
        bindings=tuple(bindings),
        path=original_text,
        supported_scope=SUPPORTED_SCOPE,
    )


def _root_wildcard_cte() -> str:
    return (
        "WITH s1 AS (\n"
        "  SELECT d./*__id__*/ AS doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_array(e.key) AS path, {_order_term()} AS ord\n"
        "  FROM /*__relation__*/\n"
        f"  WHERE json_type(d./*__doc__*/) IN {_CONTAINER_TYPES}\n"
        ")"
    )


def _root_index_cte(index: int, bindings: list[typing.Any]) -> str:
    array_length = (
        "CASE WHEN json_type(d./*__doc__*/) = 'array' "
        "THEN json_array_length(d./*__doc__*/) ELSE 0 END"
    )
    if index >= 0:
        bindings.append(index)
        key_filter = "e.key = ?"
    else:
        key_filter = (
            f"e.key = {array_length} {index:+d} AND {array_length} >= {-index}"
        )
    return (
        "WITH s1 AS (\n"
        "  SELECT d./*__id__*/ AS doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_array(e.key) AS path, {_order_term()} AS ord\n"
        "  FROM /*__relation__*/\n"
        f"  WHERE json_type(d./*__doc__*/) = 'array' AND {key_filter}\n"
        ")"
    )


def _root_filter_cte(node: Filter, bindings: list[typing.Any]) -> str:
    predicates: list[str] = []
    for expression in node.expressions:
        predicate, predicate_bindings = _compile_predicate(expression, "e.value")
        predicates.append(predicate)
        bindings.extend(predicate_bindings)
    where = " AND ".join(predicates)
    return (
        "WITH s1 AS (\n"
        "  SELECT d./*__id__*/ AS doc_id, e.value AS val, e.type AS typ,\n"
        f"    json_array(e.key) AS path, {_order_term()} AS ord\n"
        "  FROM /*__relation__*/\n"
        f"  WHERE json_type(d./*__doc__*/) IN {_CONTAINER_TYPES}"
        + (f" AND {where}" if where else "")
        + "\n)"
    )
