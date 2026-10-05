from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from ..jsonpath import DatumInContext, JSONPath


class _Nothing:
    def __repr__(self) -> str:
        return "Nothing"


Nothing = _Nothing()


@dataclass(frozen=True)
class RfcPath(JSONPath):
    segments: tuple[tuple[str, int] | tuple[str, str] | tuple[str], ...] = ()

    def child(self, segment):
        return RfcPath(self.segments + (segment,))

    def __str__(self) -> str:
        result = "$"
        for kind, value in self.segments:
            if kind == "index":
                result += f"[{value}]"
            elif kind == "name":
                result += "['" + _canonical_name_escape(value) + "']"
            else:
                result += "[*]"
        return result


def _canonical_name_escape(value: str) -> str:
    result = value.replace("\\", "\\\\").replace("'", "\\'")

    def escape_control(match):
        char = match.group(0)
        simple = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
        if char in simple:
            return simple[char]
        return "\\u%04x" % ord(char)

    return re.sub(r"[\x00-\x1f]", escape_control, result)


class RfcMatch(DatumInContext):
    def __init__(self, value: Any, path: RfcPath, root_value: Any = None):
        super().__init__(value, path=path, context=None)
        self.normalized_path = path
        self.root_value = value if root_value is None else root_value

    @property
    def full_path(self) -> JSONPath:  # type: ignore[override]
        return self.path


class RfcQuery(JSONPath):
    def find(self, data: Any) -> list[RfcMatch]:
        raise NotImplementedError


class RootQuery(RfcQuery):
    is_singular = True

    def find(self, data: Any) -> list[RfcMatch]:
        wrapped = data.root_value if isinstance(data, RfcMatch) else (
            data.value if isinstance(data, DatumInContext) else data
        )
        return [RfcMatch(wrapped, RfcPath())]

    def __str__(self) -> str:
        return "$"


class CurrentQuery(RfcQuery):
    is_singular = True

    def find(self, data: Any) -> list[RfcMatch]:
        if isinstance(data, RfcMatch):
            return [data]
        wrapped = data.value if isinstance(data, DatumInContext) else data
        return [RfcMatch(wrapped, RfcPath())]

    def __str__(self) -> str:
        return "@"


@dataclass(frozen=True)
class NameSelector:
    name: str


@dataclass(frozen=True)
class IndexSelector:
    index: int


@dataclass(frozen=True)
class WildcardSelector:
    pass


@dataclass(frozen=True)
class SliceSelector:
    start: int | None
    end: int | None
    step: int | None


@dataclass(frozen=True)
class FilterSelector:
    expression: Any

    def __str__(self) -> str:
        return str(self.expression)


@dataclass(frozen=True)
class Segment:
    kind: str
    selectors: tuple[Any, ...]
    descendant: bool = False


@dataclass(frozen=True)
class PathQuery(RfcQuery):
    root: Any
    segments: tuple[Segment, ...]

    @property
    def is_singular(self) -> bool:
        return all(
            not segment.descendant
            and len(segment.selectors) == 1
            and isinstance(segment.selectors[0], (NameSelector, IndexSelector))
            for segment in self.segments
        )

    def find(self, data: Any) -> list[RfcMatch]:
        matches = self.root.find(data)
        for segment in self.segments:
            matches = _apply_segment(matches, segment)
        return matches

    def __str__(self) -> str:
        result = str(self.root)
        for segment in self.segments:
            prefix = ".." if segment.descendant else ""
            result += prefix + _segment_to_string(segment)
        return result


def _segment_to_string(segment: Segment) -> str:
    if len(segment.selectors) == 1:
        selector = segment.selectors[0]
        if isinstance(selector, NameSelector):
            return "[" + json.dumps(selector.name, ensure_ascii=False, separators=(",", ":")) + "]"
        return _selector_to_string(selector)
    return "[" + ",".join(_selector_content(item) for item in segment.selectors) + "]"


def _selector_content(selector: Any) -> str:
    if isinstance(selector, NameSelector):
        return json.dumps(selector.name, ensure_ascii=False, separators=(",", ":"))
    if isinstance(selector, IndexSelector):
        return str(selector.index)
    if isinstance(selector, WildcardSelector):
        return "*"
    if isinstance(selector, SliceSelector):
        return _slice_to_string(selector)[1:-1]
    return f"?{selector}"


def _selector_to_string(selector: Any) -> str:
    if isinstance(selector, NameSelector):
        return "[" + json.dumps(selector.name, ensure_ascii=False, separators=(",", ":")) + "]"
    if isinstance(selector, IndexSelector):
        return f"[{selector.index}]"
    if isinstance(selector, WildcardSelector):
        return "[*]"
    if isinstance(selector, SliceSelector):
        return _slice_to_string(selector)
    return f"[?{selector}]"


def _slice_to_string(selector: SliceSelector) -> str:
    start = "" if selector.start is None else str(selector.start)
    end = "" if selector.end is None else str(selector.end)
    if selector.step in (None, 1):
        return f"[{start}:{end}]"
    return f"[{start}:{end}:{selector.step}]"


def _apply_segment(matches: Iterable[RfcMatch], segment: Segment) -> list[RfcMatch]:
    result: list[RfcMatch] = []
    for match in matches:
        contexts = _visit(match) if segment.descendant else [match]
        for context in contexts:
            for selector in segment.selectors:
                result.extend(_apply_selector(context, selector))
    return result


def _visit(match: RfcMatch) -> list[RfcMatch]:
    result = [match]
    value = match.value
    if isinstance(value, dict):
        for name, child in value.items():
            result.extend(_visit(RfcMatch(
                child, match.path.child(("name", name)), match.root_value
            )))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_visit(RfcMatch(
                child, match.path.child(("index", index)), match.root_value
            )))
    return result


def _apply_selector(match: RfcMatch, selector: Any) -> list[RfcMatch]:
    if isinstance(selector, NameSelector):
        if isinstance(match.value, dict) and selector.name in match.value:
            return [RfcMatch(
                match.value[selector.name],
                match.path.child(("name", selector.name)),
                match.root_value,
            )]
    elif isinstance(selector, IndexSelector) and isinstance(match.value, list):
        index = _normalize_index(selector.index, len(match.value))
        if index is not None:
            return [RfcMatch(
                match.value[index],
                match.path.child(("index", index)),
                match.root_value,
            )]
    elif isinstance(selector, WildcardSelector):
        return _children(match)
    elif isinstance(selector, SliceSelector) and isinstance(match.value, list):
        return _select_slice(match, selector)
    elif isinstance(selector, FilterSelector):
        return _filter_children(match, selector.expression)
    return []


def _children(match: RfcMatch) -> list[RfcMatch]:
    if isinstance(match.value, dict):
        return [
            RfcMatch(child, match.path.child(("name", name)), match.root_value)
            for name, child in match.value.items()
        ]
    if isinstance(match.value, list):
        return [
            RfcMatch(child, match.path.child(("index", index)), match.root_value)
            for index, child in enumerate(match.value)
        ]
    return []


def _filter_children(match: RfcMatch, expression: Any) -> list[RfcMatch]:
    if isinstance(match.value, list):
        result = []
        for index, child in enumerate(match.value):
            child_match = RfcMatch(
                child,
                match.path.child(("index", index)),
                match.root_value,
            )
            if _test_logical(expression, child_match):
                result.append(child_match)
        return result
    if _test_logical(expression, match):
        return [match]
    return []


def _normalize_index(index: int, length: int) -> int | None:
    if index < 0:
        index += length
    return index if 0 <= index < length else None


def _select_slice(match: RfcMatch, selector: SliceSelector) -> list[RfcMatch]:
    length = len(match.value)
    step = selector.step if selector.step is not None else 1
    if step == 0:
        return []
    indices = range(length)[selector.start:selector.end:step]
    result = []
    for index in indices:
        result.append(RfcMatch(
            match.value[index],
            match.path.child(("index", index)),
            match.root_value,
        ))
    return result


@dataclass(frozen=True)
class LiteralExpression:
    value: Any

    def __str__(self) -> str:
        return _literal_to_string(self.value)


@dataclass(frozen=True)
class FilterQuery:
    query: Any

    @property
    def singular(self) -> bool:
        return self.query.is_singular

    def __str__(self) -> str:
        return str(self.query)


@dataclass(frozen=True)
class UnaryLogical:
    operand: Any

    def __str__(self) -> str:
        return f"!{_logical_operand_to_string(self.operand)}"


@dataclass(frozen=True)
class BinaryLogical:
    left: Any
    operator: str
    right: Any

    def __str__(self) -> str:
        left = _logical_operand_to_string(self.left)
        right = _logical_operand_to_string(self.right)
        return f"{left} {self.operator} {right}"


@dataclass(frozen=True)
class ComparisonExpression:
    left: Any
    operator: str
    right: Any

    def __str__(self) -> str:
        return f"{self.left} {self.operator} {self.right}"


def _literal_to_string(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _logical_operand_to_string(expression: Any) -> str:
    text = str(expression)
    return f"({text})" if isinstance(expression, BinaryLogical) else text


def _test_logical(expression: Any, current: RfcMatch) -> bool:
    if isinstance(expression, ComparisonExpression):
        return _test_comparison(expression, current)
    if isinstance(expression, UnaryLogical):
        return not _test_logical(expression.operand, current)
    if isinstance(expression, BinaryLogical):
        if expression.operator == "&&":
            return _test_logical(expression.left, current) and _test_logical(expression.right, current)
        return _test_logical(expression.left, current) or _test_logical(expression.right, current)
    return _test_exists(expression, current)


def _test_exists(expression: Any, current: RfcMatch) -> bool:
    if isinstance(expression, FilterQuery):
        return bool(expression.query.find(current))
    if isinstance(expression, LiteralExpression):
        return expression.value is True
    return False


def _test_comparison(expression: ComparisonExpression, current: RfcMatch) -> bool:
    left = _comparison_value(expression.left, current)
    right = _comparison_value(expression.right, current)
    if left is Nothing or right is Nothing:
        return expression.operator == "!="

    if expression.operator in ("==", "!="):
        equal = _json_equal(left, right)
        return equal if expression.operator == "==" else not equal

    if not _is_number(left) or not _is_number(right):
        return False
    if expression.operator == "<":
        return left < right
    if expression.operator == "<=":
        return left <= right
    if expression.operator == ">":
        return left > right
    return left >= right


def _comparison_value(operand: Any, current: RfcMatch) -> Any:
    if isinstance(operand, LiteralExpression):
        return operand.value
    matches = operand.query.find(current)
    if not matches:
        return Nothing
    return matches[0].value


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _json_equal(left: Any, right: Any) -> bool:
    if _is_number(left) or _is_number(right):
        return _is_number(left) and _is_number(right) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right
