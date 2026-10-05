from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence, Tuple

from jsonpath_ng.exceptions import JsonPathParserError

from . import ast


_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_NAME_START_RE = re.compile(r"[A-Za-z_]|[^\x00-\x7F]")
_NAME_PART_RE = re.compile(r"(?:[A-Za-z0-9_-]|[^\x00-\x7F])*")


class Token:
    __slots__ = ("type", "value", "pos")

    def __init__(self, type: str, value: Any, pos: int):
        self.type = type
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:
        return f"Token({self.type!r}, {self.value!r}, {self.pos})"


def tokenize(expression: str) -> List[Token]:
    tokens: List[Token] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char in " \t\n\r":
            start = index
            index += 1
            while index < len(expression) and expression[index] in " \t\n\r":
                index += 1
            tokens.append(Token("WS", expression[start:index], start))
            continue
        if expression.startswith("..", index):
            tokens.append(Token("..", "..", index))
            index += 2
            continue
        if expression.startswith("&&", index):
            tokens.append(Token("&&", "&&", index))
            index += 2
            continue
        if expression.startswith("||", index):
            tokens.append(Token("||", "||", index))
            index += 2
            continue
        for operator in ("==", "!=", "<=", ">="):
            if expression.startswith(operator, index):
                tokens.append(Token(operator, operator, index))
                index += 2
                break
        else:
            if char in ".[*?]:,()!<>=$@&|":
                tokens.append(Token(char, char, index))
                index += 1
            elif char in "'\"":
                start = index
                value, index = _read_string(expression, index)
                tokens.append(Token("STRING", value, start))
            elif char == "-" or char.isdigit():
                match = _NUMBER_RE.match(expression, index)
                if match is None or match.start() != index:
                    raise JsonPathParserError(f"Invalid number at position {index}")
                if match.end() < len(expression) and (
                    expression[match.end()].isdigit() or expression[match.end()] == "."
                ):
                    raise JsonPathParserError(f"Invalid number at position {index}")
                raw = match.group(0)
                value = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
                tokens.append(Token("NUMBER", value, index))
                index = match.end()
            elif _NAME_START_RE.fullmatch(char):
                start = index
                index += 1
                match = _NAME_PART_RE.match(expression, index)
                index = match.end()
                value = expression[start:index]
                tokens.append(Token("NAME", value, start))
            else:
                raise JsonPathParserError(f"Unexpected character {char!r} at position {index}")
            continue
        continue
    tokens.append(Token("EOF", None, len(expression)))
    return tokens


def _read_string(expression: str, start: int) -> Tuple[str, int]:
    quote = expression[start]
    index = start + 1
    json_string = '"'
    while index < len(expression):
        char = expression[index]
        if char == quote:
            json_string += '"'
            try:
                return json.loads(json_string), index + 1
            except ValueError as exc:
                raise JsonPathParserError(f"Invalid string literal at position {start}") from exc
        if ord(char) < 0x20:
            raise JsonPathParserError(f"Unescaped control character at position {index}")
        if char == "\\":
            index += 1
            if index >= len(expression):
                break
            escaped = expression[index]
            if escaped in '"\\/bfnrtu':
                if escaped == '"':
                    json_string += '\\"'
                else:
                    json_string += "\\" + escaped
            elif escaped == "'":
                json_string += "'"
            else:
                raise JsonPathParserError(f"Invalid string escape at position {index - 1}")
            index += 1
        else:
            if char == '"':
                json_string += '\\"'
            else:
                json_string += char
            index += 1
    raise JsonPathParserError(f"Unterminated string literal at position {start}")


class JSONPathParser:
    def __init__(self, debug: bool = False, lexer_class=None):
        self.debug = debug
        self.lexer_class = lexer_class

    def parse(self, expression: str) -> ast.JSONPath:
        return _ParserState(tokenize(expression), expression).parse_top_query()


class _ParserState:
    def __init__(self, tokens: Sequence[Token], expression: str):
        self.tokens = tokens
        self.position = 0

    @property
    def token(self) -> Token:
        return self.tokens[self.position]

    def advance(self) -> Token:
        current = self.token
        self.position += 1
        return current

    def expect(self, type: str) -> Token:
        if self.token.type != type:
            self.error(f"Expected {type!r}, got {self.token.type!r}")
        return self.advance()

    def error(self, message: str) -> None:
        raise JsonPathParserError(f"{message} at position {self.token.pos}")

    def skip_whitespace(self) -> bool:
        saw = False
        while self.token.type == "WS":
            saw = True
            self.position += 1
        return saw

    def parse_top_query(self) -> ast.JSONPath:
        if self.token.type != "$":
            self.error("RFC 9535 JSONPath must start with '$'")
        query = self.parse_query(relative=False)
        self.skip_whitespace()
        if self.token.type != "EOF":
            self.error("Unexpected trailing input")
        return query

    def parse_query(self, relative: bool) -> ast.JSONPath:
        self.advance()
        segments: List[ast.Segment] = []
        while True:
            self.skip_whitespace()
            if self.token.type not in (".", "..", "["):
                break
            segments.append(self.parse_segment())
        return ast.JSONPath(segments, relative=relative)

    def parse_segment(self) -> ast.Segment:
        if self.token.type == "..":
            self.advance()
            if self.token.type != "[":
                self.skip_whitespace()
                if self.token.type == "*":
                    self.advance()
                    return ast.Segment((ast.WildcardSelector(),), descendant=True)
                if self.token.type != "NAME":
                    self.error("Expected descendant selector after '..'")
                return ast.Segment((ast.NameSelector(self.advance().value),), descendant=True)
            return ast.Segment((self.parse_bracket_selectors(),), descendant=True)
        if self.token.type == ".":
            self.advance()
            self.skip_whitespace()
            if self.token.type == "*":
                self.advance()
                return ast.Segment((ast.WildcardSelector(),))
            if self.token.type != "NAME":
                self.error("Expected name after '.'")
            name = self.advance().value
            return ast.Segment((ast.NameSelector(name),))
        return ast.Segment((self.parse_bracket_selectors(),))

    def parse_bracket_selectors(self) -> ast.Selector:
        self.expect("[")
        self.skip_whitespace()
        if self.token.type == "?":
            self.advance()
            expression = self.parse_or()
            self.skip_whitespace()
            self.expect("]")
            return ast.FilterSelector(expression)
        if self.token.type == "*":
            self.advance()
            self.skip_whitespace()
            self.expect("]")
            return ast.WildcardSelector()

        selectors: List[ast.Selector] = []
        while True:
            self.skip_whitespace()
            if self.token.type == "STRING":
                selectors.append(ast.NameSelector(self.advance().value))
            elif self.token.type == "NUMBER" or self.token.type == "-" or self.token.type == ":":
                selectors.append(self.parse_index_or_slice())
            else:
                self.error("Expected a name, index, wildcard, slice, or filter")
            self.skip_whitespace()
            if self.token.type == ",":
                self.advance()
                continue
            self.expect("]")
            break
        if len(selectors) == 1:
            return selectors[0]
        return _SelectorSequence(tuple(selectors))

    def parse_index_or_slice(self) -> ast.Selector:
        start = self.parse_optional_integer()
        self.skip_whitespace()
        if self.token.type != ":":
            if not isinstance(start, int):
                self.error("Expected array index")
            return ast.IndexSelector(start)
        self.advance()
        self.skip_whitespace()
        end = self.parse_optional_integer()
        self.skip_whitespace()
        step: Optional[int] = None
        if self.token.type == ":":
            self.skip_whitespace()
            self.advance()
            self.skip_whitespace()
            step = self.parse_optional_integer()
            if step == 0:
                self.error("Slice step cannot be zero")
        return ast.SliceSelector(start, end, step)

    def parse_optional_integer(self) -> Optional[int]:
        negative = False
        if self.token.type == "-":
            negative = True
            self.advance()
        if self.token.type != "NUMBER":
            if negative:
                self.error("Expected integer after '-'")
            return None
        if not isinstance(self.token.value, int):
            self.error("Expected an integer")
        value = self.advance().value
        return -value if negative else value

    def parse_or(self) -> ast.LogicalExpression:
        left = self.parse_and()
        while True:
            self.skip_whitespace()
            if self.token.type != "||":
                break
            self.advance()
            self.skip_whitespace()
            left = ast.BinaryLogical(left, self.parse_and(), "||")
        return left

    def parse_and(self) -> ast.LogicalExpression:
        left = self.parse_not()
        while True:
            self.skip_whitespace()
            if self.token.type != "&&":
                break
            self.advance()
            self.skip_whitespace()
            left = ast.BinaryLogical(left, self.parse_not(), "&&")
        return left

    def parse_not(self) -> ast.LogicalExpression:
        self.skip_whitespace()
        if self.token.type == "!":
            self.advance()
            self.skip_whitespace()
            return ast.Not(self.parse_not())
        return self.parse_basic()

    def parse_basic(self) -> ast.LogicalExpression:
        self.skip_whitespace()
        if self.token.type == "(":
            self.advance()
            query_position = self.position
            if self.token.type in ("$", "@"):
                query = self.parse_comparable()
                self.skip_whitespace()
                if self.token.type == ")":
                    self.advance()
                    if self.token.type in ("==", "!=", "<", ">", "<=", ">="):
                        self.error("Parenthesized queries can only test existence")
                    return ast.Exists(query)
                self.position = query_position
            expression = self.parse_or()
            self.skip_whitespace()
            self.expect(")")
            return expression

        left = self.parse_comparable()
        self.skip_whitespace()
        if self.token.type not in ("==", "!=", "<", ">", "<=", ">="):
            if isinstance(left, ast.JSONPath):
                return ast.Exists(left)
            if isinstance(left, ast.Literal) and isinstance(left.value, bool):
                return ast.BooleanLiteral(left.value)
            self.error("Filter test must be a query or boolean expression")
        operator = self.advance().type
        self.skip_whitespace()
        right = self.parse_comparable()
        if isinstance(left, ast.JSONPath) and not left.is_singular:
            self.error("Multi-value queries cannot be compared")
        if isinstance(right, ast.JSONPath) and not right.is_singular:
            self.error("Multi-value queries cannot be compared")
        return ast.Comparison(left, right, operator)

    def parse_comparable(self) -> Any:
        token = self.token
        if token.type in ("$", "@"):
            return self.parse_query(relative=token.type == "@")
        if token.type == "STRING":
            return ast.Literal(self.advance().value)
        if token.type == "NUMBER":
            return ast.Literal(self.advance().value)
        if token.type == "NAME" and token.value in ("true", "false", "null"):
            raw = self.advance().value
            return ast.Literal(True if raw == "true" else False if raw == "false" else None)
        self.error("Expected a query or literal in filter expression")


class _SelectorSequence(ast.Selector):
    def __init__(self, selectors: Tuple[ast.Selector, ...]):
        self.selectors = selectors

    def apply(self, node: ast.Match, root: ast.Match) -> Tuple[ast.Match, ...]:
        result: List[ast.Match] = []
        for selector in self.selectors:
            result.extend(selector.apply(node, root))
        return tuple(result)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SelectorSequence) and self.selectors == other.selectors

    def __hash__(self) -> int:
        return hash(("sequence", self.selectors))


def parse(expression: str) -> ast.JSONPath:
    return JSONPathParser().parse(expression)
