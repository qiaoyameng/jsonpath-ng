"""Compile a small, well defined subset of JSONPath into parameterized SQL.

The JSON document is expected to be stored in a SQLite column that contains
JSON text.  Compilation targets the SQLite JSON1 extension (``json_each``),
so it works against an in-memory :mod:`sqlite3` database without any server.

Supported JSONPath features
---------------------------

* root: ``$``
* object fields: ``$.foo`` and ``$['weird key']`` / ``$["weird key"]``
* array indices, including negative ones: ``$.items[0]``, ``$[-1]``
* array wildcard: ``$.items[*]``
* a single trailing scalar comparison filter on array elements,
  e.g. ``$.items[?(@.price < 10)]`` or ``$.items[?(@.active == true)]``

Everything else (object wildcards, recursive descent, slices, unions,
``||``/``&&`` combinations, regular expressions, arithmetic, functions,
filters on non arrays, ...) is rejected with :class:`JSONPathError` instead
of being silently ignored.

Comparison semantics are strict about types:

* a missing relative value never matches any comparison
* ``null``, ``true`` and ``false`` can only be tested with ``==``/``!=``
* numbers compare numerically (integers and reals together)
* strings compare with ``==``/``!=``/``<``/``<=``/``>``/``>=``
"""

from __future__ import annotations

import json as _json
import re as _re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .exceptions import JSONPathError

__all__ = [
    "JSONPathError",
    "JSONPathSyntaxError",
    "JSONPathNotSupported",
    "CompiledJSONPath",
    "Match",
    "compile_jsonpath",
    "decode_row",
    "find",
]


class JSONPathSyntaxError(JSONPathError):
    """Raised when the JSONPath expression cannot be parsed."""


class JSONPathNotSupported(JSONPathError):
    """Raised when the expression is valid JSONPath but outside the subset."""


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


class _Segment:
    pass


class _Field(_Segment):
    def __init__(self, name: str):
        self.name = name


class _Index(_Segment):
    def __init__(self, index: int):
        self.index = index


class _Wildcard(_Segment):
    pass


class _Filter(_Segment):
    def __init__(self, rel: List[Tuple[str, Any]], op: str, literal: Any):
        # rel is a sequence of ('field', name) / ('index', int)
        self.rel = rel
        self.op = op
        self.literal = literal


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_FIELD_RE_FIRST = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_FIELD_RE_CHAR = _FIELD_RE_FIRST | set("0123456789-")
_COMPARE_OPS = ("==", "=", "!=", "<=", ">=", "<", ">")
_SWAP_OP = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==",
            "=": "=", "!=": "!="}


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def error(self, msg: str) -> None:
        raise JSONPathSyntaxError(
            f"{msg} at position {self.pos} in {self.text!r}")

    def eof(self) -> bool:
        return self.pos >= len(self.text)

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def expect(self, ch: str) -> None:
        self.skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == ch:
            self.pos += 1
        else:
            self.error(f"expected {ch!r}")

    def accept(self, ch: str) -> bool:
        self.skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == ch:
            self.pos += 1
            return True
        return False

    def parse(self) -> List[_Segment]:
        self.skip_ws()
        if not self.text.startswith("$", self.pos):
            self.error("JSONPath must start with '$'")
        self.pos += 1
        segments: List[_Segment] = []
        while True:
            self.skip_ws()
            if self.eof():
                break
            ch = self.text[self.pos]
            if ch == ".":
                self.pos += 1
                if self.pos < len(self.text) and self.text[self.pos] == ".":
                    raise JSONPathNotSupported(
                        "recursive descent ('..') is not supported")
                segments.append(self._parse_dot_name())
            elif ch == "[":
                segments.append(self._parse_bracket())
            else:
                self.error("expected '.' or '['")
        self.skip_ws()
        if not self.eof():
            self.error("unexpected trailing input")
        return segments

    def _parse_dot_name(self) -> _Segment:
        if self.pos < len(self.text) and self.text[self.pos] == "*":
            raise JSONPathNotSupported(
                "object wildcard '.*' is not supported; only '[*]' on arrays")
        name = self._read_bare_name()
        if name is None:
            self.error("expected a field name after '.'")
        return _Field(name)

    def _read_bare_name(self) -> Optional[str]:
        start = self.pos
        if (self.pos < len(self.text)
                and self.text[self.pos] in _FIELD_RE_FIRST):
            self.pos += 1
            while (self.pos < len(self.text)
                   and self.text[self.pos] in _FIELD_RE_CHAR):
                self.pos += 1
            return self.text[start:self.pos]
        return None

    def _parse_bracket(self) -> _Segment:
        self.expect("[")
        self.skip_ws()
        if self.eof():
            self.error("unterminated '['")
        ch = self.text[self.pos]
        if ch == "?":
            segment = self._parse_filter()
        elif ch == "*":
            self.pos += 1
            segment = _Wildcard()
        elif ch in ("'", '"'):
            segment = _Field(self._parse_string())
        elif ch == "-" or ch.isdigit():
            segment = _Index(self._parse_integer())
        else:
            self.error("expected an index, quoted name, '*' or '?' in '['")
        self.skip_ws()
        self.expect("]")
        return segment

    def _parse_integer(self) -> int:
        start = self.pos
        if self.pos < len(self.text) and self.text[self.pos] == "-":
            self.pos += 1
        digits_start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == digits_start:
            self.error("expected an integer index")
        return int(self.text[start:self.pos])

    def _parse_string(self) -> str:
        quote = self.text[self.pos]
        self.pos += 1
        chars: List[str] = []
        while True:
            if self.eof():
                self.error("unterminated string")
            ch = self.text[self.pos]
            if ch == quote:
                self.pos += 1
                return "".join(chars)
            if ch == "\\":
                self.pos += 1
                if self.eof():
                    self.error("unterminated escape")
                esc = self.text[self.pos]
                mapping = {quote: quote, "\\": "\\", "/": "/",
                           "b": "\b", "f": "\f", "n": "\n",
                           "r": "\r", "t": "\t"}
                if esc in mapping:
                    chars.append(mapping[esc])
                    self.pos += 1
                elif esc == "u":
                    hexd = self.text[self.pos + 1:self.pos + 5]
                    if len(hexd) != 4 or any(c not in "0123456789abcdefABCDEF"
                                             for c in hexd):
                        self.error("invalid \\u escape")
                    chars.append(chr(int(hexd, 16)))
                    self.pos += 5
                else:
                    self.error("invalid string escape")
            else:
                chars.append(ch)
                self.pos += 1

    def _parse_filter(self) -> _Filter:
        self.pos += 1  # consume '?'
        self.skip_ws()
        parenthesized = self.accept("(")
        self.skip_ws()
        left_is_rel = self.text.startswith("@", self.pos)
        if left_is_rel:
            rel = self._parse_relative()
            literal = _MISSING
        else:
            rel = []
            literal = self._parse_literal()
        self.skip_ws()
        for op in _COMPARE_OPS:
            if self.text.startswith(op, self.pos):
                operator = "==" if op == "=" else op
                self.pos += len(op)
                break
        else:
            self.error("expected a comparison operator in filter")
        self.skip_ws()
        if left_is_rel:
            literal = self._parse_literal()
        else:
            rel = self._parse_relative()
        if not rel:
            self.error("one side of the comparison must be a '@' path")
        self.skip_ws()
        if parenthesized:
            self.expect(")")
            self.skip_ws()
        if not self.eof() and self.text[self.pos] == "]":
            if not left_is_rel:
                operator = _SWAP_OP[operator]
            return self._finish_filter(rel, operator, literal)
        self.error("only a single scalar comparison is allowed in a filter")

    def _finish_filter(self, rel, operator, literal) -> _Filter:
        if (operator in ("<", "<=", ">", ">=")
                and (literal is None or isinstance(literal, bool))):
            raise JSONPathNotSupported(
                "ordering comparisons are only supported for numbers/strings")
        if literal is None or isinstance(literal, (bool, int, float, str)):
            return _Filter(rel, operator, literal)
        raise JSONPathNotSupported(
            "filters can only compare against scalar literals")

    def _parse_relative_if_present(self) -> List[Tuple[str, Any]]:
        self.skip_ws()
        if self.text.startswith("@", self.pos):
            return self._parse_relative()
        return []

    def _parse_relative(self) -> List[Tuple[str, Any]]:
        if not self.text.startswith("@", self.pos):
            self.error("expected '@' relative path")
        self.pos += 1
        steps: List[Tuple[str, Any]] = []
        while True:
            self.skip_ws()
            if self.eof():
                break
            if self.text[self.pos] == ".":
                self.pos += 1
                self.skip_ws()
                if not self.eof() and self.text[self.pos] == "[":
                    pass
                else:
                    name = self._read_bare_name()
                    if name is None:
                        self.error("expected a field name after '.'")
                    steps.append(("field", name))
                    continue
            if self.text[self.pos] == "[":
                self.pos += 1
                self.skip_ws()
                if self.eof():
                    self.error("unterminated '['")
                ch = self.text[self.pos]
                if ch in ("'", '"'):
                    steps.append(("field", self._parse_string()))
                elif ch == "-" or ch.isdigit():
                    steps.append(("index", self._parse_integer()))
                else:
                    self.error(
                        "only field names and indices allowed after '@'")
                self.skip_ws()
                self.expect("]")
                continue
            break
        return steps

    def _parse_literal(self) -> Any:
        self.skip_ws()
        if self.eof():
            self.error("expected a literal after operator")
        ch = self.text[self.pos]
        if ch in ("'", '"'):
            return self._parse_string()
        if ch == "-" or ch.isdigit():
            return self._parse_number()
        word = self._read_word()
        if word is None:
            self.error("unsupported literal")
        low = word.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low == "null":
            return None
        self.error("unsupported literal (allowed: number, quoted string, "
                   "true, false, null)")

    def _parse_number(self) -> Union[int, float]:
        start = self.pos
        if self.text[self.pos] == "-":
            self.pos += 1
        digit_start = self.pos
        while not self.eof() and self.text[self.pos].isdigit():
            self.pos += 1
        if self.pos == digit_start:
            self.error("invalid number")
        is_float = False
        if not self.eof() and self.text[self.pos] == ".":
            nxt = self.pos + 1
            if nxt < len(self.text) and self.text[nxt].isdigit():
                is_float = True
                self.pos += 1
                while not self.eof() and self.text[self.pos].isdigit():
                    self.pos += 1
        if not self.eof() and self.text[self.pos] in "eE":
            nxt = self.pos + 1
            if nxt < len(self.text) and (self.text[nxt].isdigit()
                                         or self.text[nxt] in "+-"):
                is_float = True
                self.pos += 1
                if self.text[self.pos] in "+-":
                    self.pos += 1
                while not self.eof() and self.text[self.pos].isdigit():
                    self.pos += 1
        token = self.text[start:self.pos]
        try:
            return float(token) if is_float else int(token)
        except ValueError:
            self.error("invalid number")
            return 0  # pragma: no cover

    def _read_word(self) -> Optional[str]:
        start = self.pos
        while not self.eof() and (self.text[self.pos].isalnum()
                                 or self.text[self.pos] == "_"):
            self.pos += 1
        if self.pos == start:
            return None
        return self.text[start:self.pos]


_MISSING = object()


# ---------------------------------------------------------------------------
# Normalized paths
# ---------------------------------------------------------------------------


def _quote_name(name: str) -> str:
    """Render an object member name per RFC 9535 normal form."""
    escaped = (name.replace("\\", "\\\\").replace("'", "\\'")
               .replace("\b", "\\b").replace("\f", "\\f").replace("\n", "\\n")
               .replace("\r", "\\r").replace("\t", "\\t"))
    return "['" + escaped + "']"


# ---------------------------------------------------------------------------
# Reference evaluation in Python
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """One matched value together with its normalized path."""

    value: Any
    path: str


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise JSONPathError(f"value is not JSON serializable: {value!r}")


def _resolve(value: Any, rel: Sequence[Tuple[str, Any]]):
    """Return (kind, value) for a relative path; kind is None when missing."""
    current = value
    for kind_of_step, token in rel:
        if kind_of_step == "field":
            if not isinstance(current, dict) or token not in current:
                return None, None
            current = current[token]
        else:
            if not isinstance(current, list):
                return None, None
            index = token if token >= 0 else len(current) + token
            if index < 0 or index >= len(current):
                return None, None
            current = current[index]
    return _json_type(current), current


def _compare(kind: Optional[str], actual: Any, op: str, literal: Any) -> bool:
    if kind is None:
        return False
    if literal is None:
        return (kind == "null") if op in ("==", "=") else (kind != "null")
    if isinstance(literal, bool):
        wanted = "true" if literal else "false"
        return (kind == wanted) if op == "==" else (kind != wanted)
    if isinstance(literal, (int, float)):
        if kind not in ("integer", "real"):
            return False
        if op == "==":
            return actual == literal
        if op == "!=":
            return actual != literal
        if op == "<":
            return actual < literal
        if op == "<=":
            return actual <= literal
        if op == ">":
            return actual > literal
        return actual >= literal
    if isinstance(literal, str):
        if kind != "text":
            return False
        if op == "==":
            return actual == literal
        if op == "!=":
            return actual != literal
        if op == "<":
            return actual < literal
        if op == "<=":
            return actual <= literal
        if op == ">":
            return actual > literal
        return actual >= literal
    return False  # pragma: no cover - rejected at compile time


def find(expression: str, document: Any) -> List[Match]:
    """Evaluate ``expression`` against ``document`` using Python only.

    This is the reference implementation; the generated SQL produces the
    same (value, path) pairs in the same order.
    """
    segments = _Parser(expression).parse()
    results: List[Tuple[Any, str, str]] = []

    def visit(node: Any, path: str, ordinal: str, segs: List[_Segment],
              node_is_element: bool = False) -> None:
        if not segs:
            results.append((node, path, ordinal))
            return
        segment, rest = segs[0], segs[1:]
        is_element = isinstance(segment, (_Wildcard, _Filter))
        if isinstance(segment, _Field):
            if isinstance(node, dict) and segment.name in node:
                visit(node[segment.name],
                      path + _quote_name(segment.name), ordinal, rest)
        elif isinstance(segment, _Index):
            if isinstance(node, list):
                index = (segment.index if segment.index >= 0
                         else len(node) + segment.index)
                if 0 <= index < len(node):
                    visit(node[index], f"{path}[{index}]",
                          ordinal + f"{index:020d}", rest, False)
        elif isinstance(segment, _Wildcard):
            if isinstance(node, list):
                for index, item in enumerate(node):
                    visit(item, f"{path}[{index}]",
                          ordinal + f"{index:020d}", rest, True)
        elif isinstance(segment, _Filter):
            if node_is_element:
                kind, actual = _resolve(node, segment.rel)
                if _compare(kind, actual, segment.op, segment.literal):
                    visit(node, path, ordinal, rest, True)
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    kind, actual = _resolve(item, segment.rel)
                    if _compare(kind, actual, segment.op, segment.literal):
                        visit(item, f"{path}[{index}]",
                              ordinal + f"{index:020d}", rest, True)

    visit(document, "$", "", segments)
    return [Match(value, path) for value, path, _ in results]



def decode_row(value_json: str, json_type: str) -> Any:
    """Decode a ``(value, json_type)`` pair produced by the compiled SQL."""
    if json_type == "null":
        return None
    if json_type == "true":
        return True
    if json_type == "false":
        return False
    return _json.loads(value_json)

# ---------------------------------------------------------------------------
# SQL compilation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledJSONPath:
    """Result of compiling a JSONPath expression.

    Attributes:
        expression: the original JSONPath text.
        sql: a complete ``WITH ... SELECT`` statement. Rows contain the
            columns ``path``, ``value`` (a JSON text scalar), ``json_type``
            and ``ord`` (a deterministic array ordering key).
        fragment: the same WITH/SELECT body usable as a correlated subquery.
        params: bind parameters, keyed by name.
        table: quoted table identifier used.
        column: quoted JSON column identifier used.
    """

    expression: str
    sql: str
    fragment: str
    params: dict
    table: str
    column: str

    @property
    def cte(self) -> str:
        """Everything up to (but not including) the final SELECT."""
        return self.fragment

    def exists_sql(self) -> str:
        """An EXISTS predicate for documents containing at least one match."""
        return (f"EXISTS (SELECT 1 FROM ({self.fragment}) _m "
                "WHERE _m.doc_id = _t.rowid)")


_IDENTIFIER_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str, kind: str) -> str:
    if not isinstance(identifier, str) or not _IDENTIFIER_RE.match(identifier):
        raise JSONPathError(
            f"invalid {kind} identifier {identifier!r}: only ASCII letters, "
            "digits and underscores are accepted; identifiers are quoted "
            "in SQL")
    return '"' + identifier + '"'


class _Builder:
    def __init__(self, expression: str, table: str, column: str):
        self.expression = expression
        self.table = table
        self.column = column
        self.segments = _Parser(expression).parse()
        self.params: dict = {}
        self._counter = 0

    def bind(self, value: Any) -> str:
        self._counter += 1
        name = f"jp{self._counter}"
        self.params[name] = value
        return f":{name}"

    def build(self) -> Tuple[str, str]:
        ctes = [
            f"w0 AS (SELECT rowid AS doc_id, json({self.column}) AS node, "
            "'$' AS pfx, '' AS ord "
            f"FROM {self.table} _q)"
        ]
        if not self.segments:
            body = self._final_cte("w0")
            ctes.append("w1 AS (" + body + ")")
            last = "w1"
        else:
            last = "w0"
            previous = None
            for depth, segment in enumerate(self.segments, start=1):
                body = self._step_body(last, segment, previous)
                previous = segment
                last = f"w{depth}"
                ctes.append(f"{last} AS ({body})")
        text_expr = ("json(" + last + ".node)") if not self.segments else (
            "json_quote(" + last + ".node)")
        scalar = (
            f"CASE {last}.type "
            "WHEN 'object' THEN json(" + last + ".node) "
            "WHEN 'array' THEN json(" + last + ".node) "
            "WHEN 'true' THEN 'true' "
            "WHEN 'false' THEN 'false' "
            "WHEN 'null' THEN 'null' "
            "WHEN 'text' THEN " + text_expr + " "
            "ELSE json(" + last + ".node) END")
        fragment = ("WITH " + ",\n".join(ctes)
                    + (f"\nSELECT doc_id, {last}.pfx AS path, "
                        f"{scalar} AS value, {last}.type AS json_type, "
                        f"{last}.ord AS ord FROM {last}"))
        sql = fragment + " ORDER BY doc_id, ord, path"
        return sql, fragment

    def _final_cte(self, prev: str) -> str:
        return (f"SELECT {prev}.doc_id AS doc_id, {prev}.node AS node, "
                f"{prev}.pfx AS pfx, {prev}.ord AS ord, "
                f"json_type({prev}.node) AS type FROM {prev}")

    def _step_body(self, prev: str, segment: _Segment,
                   previous=None) -> str:
        if isinstance(segment, _Field):
            key = self.bind(segment.name)
            prefix = self.bind(_field_prefix(segment.name))
            source = _each_source(f"{prev}.node", "object")
            return (
                f"SELECT {prev}.doc_id AS doc_id, e.value AS node, "
                f"{prev}.pfx || {prefix} AS pfx, {prev}.ord AS ord, "
                "e.type AS type "
                f"FROM {prev}, json_each({source}) e "
                f"WHERE {_is_json_text(f'{prev}.node', 'object')} "
                f"AND typeof(e.key) = 'text' AND e.key = {key}")
        if isinstance(segment, _Index):
            index = self.bind(segment.index)
            source = _each_source(f"{prev}.node", "array")
            return (
                f"SELECT {prev}.doc_id AS doc_id, e.value AS node, "
                f"{prev}.pfx || '[' || e.key || ']' AS pfx, "
                f"{prev}.ord || printf('%020d', e.key) AS ord, "
                "e.type AS type "
                f"FROM {prev}, json_each({source}) e "
                f"WHERE {_is_json_text(f'{prev}.node', 'array')} "
                f"AND e.key = CASE WHEN {index} >= 0 THEN {index} "
                f"ELSE json_array_length({prev}.node) + {index} END")
        if isinstance(segment, _Wildcard):
            source = _each_source(f"{prev}.node", "array")
            return (
                f"SELECT {prev}.doc_id AS doc_id, e.value AS node, "
                f"{prev}.pfx || '[' || e.key || ']' AS pfx, "
                f"{prev}.ord || printf('%020d', e.key) AS ord, "
                "e.type AS type "
                f"FROM {prev}, json_each({source}) e "
                f"WHERE {_is_json_text(f'{prev}.node', 'array')}")
        if isinstance(segment, _Filter):
            after_wildcard = isinstance(previous, _Wildcard)
            candidate = (f"{prev}.node" if after_wildcard else "e.value")
            predicate = self._filter_predicate(segment, candidate)
            if after_wildcard:
                return (
                    f"SELECT {prev}.doc_id AS doc_id, {prev}.node AS node, "
                    f"{prev}.pfx AS pfx, {prev}.ord AS ord, "
                    f"{prev}.type AS type "
                    f"FROM {prev} WHERE ({predicate})")
            source = _each_source(f"{prev}.node", "array")
            return (
                f"SELECT {prev}.doc_id AS doc_id, e.value AS node, "
                f"{prev}.pfx || '[' || e.key || ']' AS pfx, "
                f"{prev}.ord || printf('%020d', e.key) AS ord, "
                "e.type AS type "
                f"FROM {prev}, json_each({source}) e "
                f"WHERE {_is_json_text(f'{prev}.node', 'array')} "
                f"AND ({predicate})")
        raise JSONPathNotSupported(  # pragma: no cover
            f"unsupported segment: {segment!r}")

    def _filter_predicate(self, segment: _Filter, candidate: str) -> str:
        kind_expr, value_expr = self._relative_exprs(segment.rel, candidate)
        literal = segment.literal
        op = segment.op
        if literal is None:
            if op == "==":
                return f"{kind_expr} = 'null'"
            return f"{kind_expr} IS NOT NULL AND {kind_expr} <> 'null'"
        if isinstance(literal, bool):
            wanted = "true" if literal else "false"
            if op == "==":
                return f"{kind_expr} = '{wanted}'"
            return f"{kind_expr} IS NOT NULL AND {kind_expr} <> '{wanted}'"
        bound = self.bind(literal)
        if isinstance(literal, (int, float)):
            numeric = f"({kind_expr} IN ('integer', 'real'))"
            if op == "==":
                return f"{numeric} AND {value_expr} = {bound}"
            if op == "!=":
                return f"{numeric} AND {value_expr} <> {bound}"
            return f"{numeric} AND {value_expr} {op} {bound}"
        if op == "==":
            return f"{kind_expr} = 'text' AND {value_expr} = {bound}"
        if op == "!=":
            return f"{kind_expr} = 'text' AND {value_expr} <> {bound}"
        return f"{kind_expr} = 'text' AND {value_expr} {op} {bound}"

    def _relative_exprs(self, rel: Sequence[Tuple[str, Any]],
                        candidate: str = "e.value"):
        # A chain of correlated json_each lookups starting at the candidate
        # element. ``kind`` reads json_each's declared row type, so values
        # stored with a raw integer/real/NULL storage class (booleans, JSON
        # null, numbers) are classified without re-parsing them. Each
        # json_each argument is guarded to NULL unless the current value is a
        # JSON text container of the required kind.
        kind = "NULL"
        value = candidate
        depth = 0
        for step_kind, token in rel:
            depth += 1
            alias = f"r{depth}"
            expected = "object" if step_kind == "field" else "array"
            source = (f"(SELECT {value} WHERE "
                      f"{_is_json_text(value, expected)})")
            if step_kind == "field":
                bound = self.bind(token)
                match = (f"typeof({alias}.key) = 'text' "
                         f"AND {alias}.key = {bound}")
            else:
                bound = self.bind(token)
                length = (f"(SELECT json_array_length({value}) WHERE "
                          f"{_is_json_text(value, 'array')})")
                match = (
                    f"{alias}.key = CASE WHEN {bound} >= 0 THEN {bound} "
                    f"ELSE {length} + {bound} END")
            kind = (f"(SELECT {alias}.type FROM json_each({source}) {alias} "
                    f"WHERE {match})")
            value = (f"(SELECT {alias}.value FROM json_each({source}) "
                     f"{alias} WHERE {match})")
        return kind, value


def _is_json_text(expression: str, expected: Optional[str] = None) -> str:
    # Step CTEs carry json_each values verbatim: containers are stored as JSON
    # text while bare scalars have integer/real/NULL storage classes. Calling
    # json_valid()/json_type() on a non-text value raises "malformed JSON", so
    # every JSON predicate must be gated on typeof(x) = 'text'.
    suffix = f" AND json_type({expression}) = '{expected}'" if expected else ""
    return (f"(typeof({expression}) = 'text' AND json_valid({expression})"
            f"{suffix})")


def _each_source(expression: str, expected: str) -> str:
    # Make the json_each argument NULL unless the value is a JSON container of
    # the expected kind, so json_each never evaluates a bare scalar.
    guard = _is_json_text(expression, expected)
    return f"(SELECT {expression} WHERE {guard})"


def _field_prefix(name: str) -> str:
    # Normalized object member reference in the output path.
    return _quote_name(name)


def compile_jsonpath(expression: str, table: str,
                    column: str) -> CompiledJSONPath:
    """Compile ``expression`` into a parameterized SQLite statement.

    Args:
        expression: the supported JSONPath subset (see module docstring).
        table: controlled table name (identifier, not data).
        column: controlled JSON column name (identifier, not data).

    Returns:
        CompiledJSONPath with the full SQL, a reusable fragment and the
        bind parameters.
    """
    quoted_table = _quote_identifier(table, "table")
    quoted_column = _quote_identifier(column, "column")
    builder = _Builder(expression, quoted_table, quoted_column)
    sql, fragment = builder.build()
    return CompiledJSONPath(
        expression=expression,
        sql=sql,
        fragment=fragment,
        params=dict(builder.params),
        table=quoted_table,
        column=quoted_column,
    )
