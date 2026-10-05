from __future__ import annotations

import re
from typing import Any

from ..exceptions import JsonPathLexerError, JsonPathParserError
from .query import (
    BinaryLogical,
    ComparisonExpression,
    FilterQuery,
    FilterSelector,
    IndexSelector,
    LiteralExpression,
    NameSelector,
    PathQuery,
    CurrentQuery,
    RootQuery,
    Segment,
    SliceSelector,
    UnaryLogical,
    WildcardSelector,
)


def parse(expression: str):
    return Rfc9535Parser(expression).parse()


class Token:
    def __init__(self, kind: str, value: Any, position: int):
        self.kind = kind
        self.value = value
        self.position = position


class Rfc9535Parser:
    _integer = re.compile(r"-?(?:0|[1-9][0-9]*)")
    _number = re.compile(
        r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    )
    _name = re.compile(
        r"(?:[A-Za-z0-9_~]|[\u0080-\U0010FFFF])"
        r"(?:[A-Za-z0-9_~-]|[\u0080-\U0010FFFF])*"
    )
    _shorthand_name = re.compile(
        r"(?:[A-Za-z0-9_~-]|@[A-Za-z0-9_~.-]|[\u0080-\U0010FFFF])"
        r"(?:[A-Za-z0-9_~@-]|[\u0080-\U0010FFFF])*"
    )

    def __init__(self, expression: str):
        self.expression = expression

    def parse(self) -> PathQuery:
        tokens = list(self._tokenize(self.expression))
        parser = _TokenParser(self.expression, tokens)
        return parser.parse_top_query()

    def _tokenize(self, expression: str):
        index = 0
        shorthand_next = False
        while index < len(expression):
            char = expression[index]
            if char in " \t\n\r":
                yield Token("WS", char, index)
                index += 1
            elif expression.startswith(("==", "!=", "<=", ">=", "&&", "||"), index):
                yield Token(expression[index:index + 2], expression[index:index + 2], index)
                index += 2
            elif char in "$@.*[]:?,!<>()":
                yield Token(char, char, index)
                index += 1
                shorthand_next = char == "."
            elif char == '"':
                value, next_index = self._read_string(expression, index)
                yield Token("STRING", value, index)
                index = next_index
                shorthand_next = False
            elif char == "'":
                self._error(index, "single-quoted strings are not part of RFC 9535")
            elif shorthand_next and (
                char.isalnum() or char in "_~@-" or ord(char) >= 0x80
            ):
                match = self._shorthand_name.match(expression, index)
                if not match:
                    self._error(index, "invalid member name")
                yield Token("NAME", match.group(0), index)
                index = match.end()
                shorthand_next = False
            elif char == "-" or char.isdigit():
                match = self._number.match(expression, index)
                if not match or self._is_identifier_character(expression, match.end()):
                    self._error(index, "invalid number literal")
                text = match.group(0)
                kind = "INT" if self._integer.fullmatch(text) else "NUMBER"
                value = int(text) if kind == "INT" else float(text)
                yield Token(kind, value, index)
                index = match.end()
            elif char.isalpha() or char == "_":
                match = self._name.match(expression, index)
                if not match or self._is_identifier_character(expression, match.end()):
                    self._error(index, "invalid member name")
                yield Token("NAME", match.group(0), index)
                index = match.end()
            else:
                self._error(index, f"unexpected character {char!r}")

    def _read_string(self, expression: str, start: int) -> tuple[str, int]:
        index = start + 1
        result = []
        while index < len(expression):
            char = expression[index]
            if ord(char) < 0x20:
                self._error(index, "unescaped control character in string")
            if char == '"':
                return "".join(result), index + 1
            if char != "\\":
                result.append(char)
                index += 1
            else:
                value, index = self._read_escape(expression, index)
                result.append(value)
        self._error(start, "unterminated string")

    def _read_escape(self, expression: str, start: int) -> tuple[str, int]:
        if start + 1 >= len(expression):
            self._error(start, "unterminated string escape")
        escape = expression[start + 1]
        simple = {'"': '"', "\\": "\\", "/": "/", "b": "\b",
                  "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
        if escape in simple:
            return simple[escape], start + 2
        if escape != "u":
            self._error(start + 1, "invalid string escape")
        code = self._read_hex(expression, start + 2)
        index = start + 6
        if 0xD800 <= code <= 0xDBFF:
            if expression[index:index + 2] != "\\u":
                self._error(index, "invalid Unicode surrogate pair")
                return "", index
            low = self._read_hex(expression, index + 2)
            if not 0xDC00 <= low <= 0xDFFF:
                self._error(index, "invalid Unicode surrogate pair")
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
            index += 6
        elif 0xDC00 <= code <= 0xDFFF:
            self._error(start + 2, "invalid Unicode surrogate pair")
        return chr(code), index

    def _read_hex(self, expression: str, position: int) -> int:
        text = expression[position:position + 4]
        if len(text) != 4 or any(char not in "0123456789abcdefABCDEF" for char in text):
            self._error(position, "invalid Unicode escape")
        return int(text, 16)

    def _is_identifier_character(self, expression: str, position: int) -> bool:
        return position < len(expression) and (
            expression[position].isalnum() or expression[position] in "_-"
        )

    def _error(self, position: int, message: str):
        raise JsonPathLexerError(f"{message} at position {position}")


class _TokenParser:
    def __init__(self, expression: str, tokens: list[Token]):
        self.expression = expression
        self.tokens = tokens
        self.index = 0

    def parse_top_query(self) -> PathQuery:
        root = self._parse_query_root(skip_whitespace=False)
        if not isinstance(root, RootQuery):
            self._error(self._current_position(), "RFC 9535 queries must start with '$'")
        segments = self._parse_segments(allow_leading_space=False)
        token = self._peek_raw()
        if token is not None:
            self._unexpected(token)
        return PathQuery(root, tuple(segments))

    def _parse_filter_query(self):
        root = self._parse_query_root(skip_whitespace=True)
        if not isinstance(root, (RootQuery, CurrentQuery)):
            self._error(self._current_position(), "filter query must start with '$' or '@'")
        segments = self._parse_segments(allow_leading_space=False)
        return PathQuery(root, tuple(segments))

    def _parse_query_root(self, skip_whitespace=True):
        token = self._peek() if skip_whitespace else self._peek_raw()
        if token is not None and token.kind == "$":
            self.index += 1
            return RootQuery()
        if token is not None and token.kind == "@":
            self.index += 1
            return CurrentQuery()
        return None

    def _parse_segments(self, allow_leading_space):
        segments = []
        while True:
            if allow_leading_space:
                self._skip_ws()
            token = self._peek_raw()
            if token is not None and token.kind == "WS":
                break
            if token is None or token.kind == "]" or token.kind in ("==", "!=", "<=", ">=", "<", ">"):
                break
            if token.kind in ("&&", "||", "!", ")"):
                break
            descendant = False
            if token.kind == ".":
                self.index += 1
                if self._peek_kind() == ".":
                    descendant = True
                    self.index += 1
                spaced = (
                    self._peek_kind() == "WS"
                    and self._next_non_ws_kind() == "["
                )
                if not descendant and self._peek_kind() == "[":
                    self._error(token.position, "child bracketed segments cannot follow a dot")
            else:
                if token.kind != "[":
                    self._unexpected(token)
                spaced = True
            selectors = self._parse_segment_selectors(spaced)
            segments.append(Segment("child", tuple(selectors), descendant))
        return segments

    def _parse_segment_selectors(self, spaced: bool):
        token = self._peek() if spaced else self._peek_raw()
        if token is None:
            self._error(self._current_position(), "unexpected end of query")
        if token.kind == "[":
            return self._parse_bracket_selectors()
        if token.kind == "?":
            self._error(token.position, "filter selectors must use bracket notation")
        if token.kind == "*":
            self.index += 1
            return [WildcardSelector()]
        if token.kind == "NAME":
            self.index += 1
            name = token.value
            while self._is_continuation_dot():
                self.index += 1
                continuation = self.tokens[self.index]
                self.index += 1
                name += "." + continuation.value
            return [NameSelector(name)]
        self._unexpected(token)

    def _parse_bracket_selectors(self):
        opening = self._consume("[")
        selectors = []
        self._skip_ws()
        while self._peek_kind() != "]":
            if self._peek_kind() is None:
                self._error(opening.position, "unterminated bracketed segment")
            selectors.append(self._parse_bracket_selector())
            self._skip_ws()
            if self._peek_kind() != ",":
                break
            self.index += 1
            self._skip_ws()
            if self._peek_kind() == "]":
                self._error(self._current_position(), "trailing comma is not allowed")
        self._skip_ws()
        if not selectors and self._peek_kind() == "]":
            self._error(opening.position, "empty bracketed segment")
        self._consume("]")
        return selectors

    def _parse_bracket_selector(self):
        self._skip_ws()
        token = self._peek()
        if token is None:
            self._error(self._current_position(), "unterminated bracketed selector")
        if token.kind == "STRING":
            self.index += 1
            selector = NameSelector(token.value)
        elif token.kind == "*":
            self.index += 1
            selector = WildcardSelector()
        elif token.kind == "?":
            selector = self._parse_filter_selector()
        elif token.kind in ("INT", ":"):
            selector = self._parse_index_or_slice()
        else:
            self._unexpected(token)
        self._skip_ws()
        return selector

    def _parse_index_or_slice(self):
        token = self._peek_raw()
        start = token.value if token is not None and token.kind == "INT" else None
        if start is not None:
            self.index += 1
        if self._next_non_ws_kind() != ":":
            if start is None:
                self._unexpected(self._peek())
            return IndexSelector(start)
        self._skip_ws()
        self.index += 1
        self._skip_ws()
        end = self._parse_optional_int()
        self._skip_ws()
        step = 1
        if self._peek_kind() == ":":
            self.index += 1
            self._skip_ws()
            step = self._parse_optional_int()
            self._skip_ws()
        return SliceSelector(start, end, step)

    def _parse_optional_int(self):
        token = self._peek_raw()
        if token is not None and token.kind == "INT":
            self.index += 1
            return token.value
        return None

    def _parse_filter_selector(self):
        self._consume("?")
        expression = self._parse_or_expression()
        self._skip_ws()
        return FilterSelector(expression)

    def _parse_or_expression(self):
        expression = self._parse_and_expression()
        while self._peek() is not None and self._peek_kind() == "||":
            self.index += 1
            right = self._parse_and_expression()
            expression = BinaryLogical(expression, "||", right)
        return expression

    def _parse_and_expression(self):
        expression = self._parse_basic_expression()
        while self._peek() is not None and self._peek_kind() == "&&":
            self.index += 1
            right = self._parse_basic_expression()
            expression = BinaryLogical(expression, "&&", right)
        return expression

    def _parse_basic_expression(self):
        self._skip_ws()
        token = self._peek()
        if token is None:
            self._error(self._current_position(), "unexpected end of filter expression")
        if token.kind == "(":
            self.index += 1
            expression = self._parse_or_expression()
            self._skip_ws()
            self._consume(")")
            return expression
        if token.kind == "!":
            self.index += 1
            self._skip_ws()
            return UnaryLogical(self._parse_test_operand())
        return self._parse_comparison_or_exists()

    def _parse_test_operand(self):
        self._skip_ws()
        token = self._peek()
        if token is not None and token.kind == "!":
            self.index += 1
            self._skip_ws()
            return UnaryLogical(self._parse_test_operand())
        if token is not None and token.kind in ("$", "@"):
            return FilterQuery(self._parse_filter_query())
        if token is not None and (
            token.kind == "STRING"
            or token.kind in ("INT", "NUMBER")
            or token.kind == "NAME"
            and token.value in ("true", "false", "null")
        ):
            return self._parse_comparison_operand()
        self._unexpected(token)

    def _parse_comparison_or_exists(self):
        self._skip_ws()
        token = self._peek()
        if token is None:
            self._error(self._current_position(), "unexpected end of filter expression")
        if token.kind in ("$", "@"):
            query = self._parse_filter_query()
            left = FilterQuery(query)
        elif token.kind == "STRING" or token.kind in ("INT", "NUMBER") or (
            token.kind == "NAME"
            and token.value in ("true", "false", "null")
        ):
            left = self._parse_comparison_operand()
        else:
            self._unexpected(token)

        token = self._peek()
        if token is None or token.kind not in ("==", "!=", "<", "<=", ">", ">="):
            return self._as_exists(left)
        if isinstance(left, FilterQuery) and not left.singular:
            self._error(token.position, "comparison requires a singular query")
        operator = token.kind
        self.index += 1
        right = self._parse_comparison_operand()
        return ComparisonExpression(left, operator, right)

    def _as_exists(self, operand):
        if isinstance(operand, FilterQuery):
            return operand
        self._unexpected(self._peek())

    def _parse_comparison_operand(self):
        self._skip_ws()
        token = self._peek()
        if token is None:
            self._error(self._current_position(), "missing comparison operand")
        if token.kind in ("$", "@"):
            query = self._parse_filter_query()
            if not query.is_singular:
                self._error(token.position, "comparison requires a singular query")
            return FilterQuery(query)
        if token.kind == "STRING":
            self.index += 1
            return LiteralExpression(token.value)
        if token.kind in ("INT", "NUMBER"):
            self.index += 1
            return LiteralExpression(token.value)
        if token.kind == "NAME" and token.value in ("true", "false", "null"):
            self.index += 1
            value = None if token.value == "null" else token.value == "true"
            return LiteralExpression(value)
        self._unexpected(token)

    def _skip_ws(self) -> None:
        while self._peek_kind() == "WS":
            self.index += 1

    def _peek(self) -> Token | None:
        self._skip_ws()
        return self._peek_raw()

    def _peek_raw(self) -> Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _peek_kind(self) -> str | None:
        token = self.tokens[self.index] if self.index < len(self.tokens) else None
        return token.kind if token is not None else None

    def _next_non_ws_kind(self) -> str | None:
        index = self.index
        while index < len(self.tokens) and self.tokens[index].kind == "WS":
            index += 1
        return self.tokens[index].kind if index < len(self.tokens) else None

    def _is_continuation_dot(self) -> bool:
        if self._peek_kind() != ".":
            return False
        following = self.tokens[self.index + 1] if self.index + 1 < len(self.tokens) else None
        return following is not None and following.kind == "NAME"

    def _consume(self, kind: str) -> Token:
        self._skip_ws()
        token = self._peek()
        if token is None or token.kind != kind:
            self._unexpected(token)
        self.index += 1
        return token

    def _current_position(self) -> int:
        token = self.tokens[self.index] if self.index < len(self.tokens) else None
        return token.position if token is not None else len(self.expression)

    def _error(self, position: int, message: str):
        raise JsonPathParserError(f"{message} at position {position}")

    def _unexpected(self, token: Token | None):
        if token is None:
            raise JsonPathParserError("Parse error near the end of string!")
        raise JsonPathParserError(
            f"Parse error at position {token.position} near token {token.value!r} ({token.kind})"
        )
