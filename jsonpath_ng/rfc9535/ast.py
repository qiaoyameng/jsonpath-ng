from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from .path import Match, NodeList, Nothing, NormalizedPath, is_nothing


class JSONPath:
    def __init__(self, segments: Sequence["Segment"], relative: bool = False):
        self.segments = tuple(segments)
        self.relative = relative

    @property
    def is_singular(self) -> bool:
        return all(segment.is_singular for segment in self.segments)

    def find(self, data: Any) -> List[Match]:
        root = Match(data, NormalizedPath(), data)
        return list(self.evaluate(root, root).nodes)

    def evaluate(self, current: Match, root: Match) -> NodeList:
        nodes = (current,)
        if not self.relative:
            nodes = (root,)
        for segment in self.segments:
            nodes = segment.apply(nodes, root)
        return NodeList(nodes)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, JSONPath)
            and self.relative == other.relative
            and self.segments == other.segments
        )

    def __hash__(self) -> int:
        return hash((self.relative, self.segments))


class Segment:
    def __init__(self, selectors: Sequence["Selector"], descendant: bool = False):
        self.selectors = tuple(selectors)
        self.descendant = descendant

    @property
    def is_singular(self) -> bool:
        return (
            not self.descendant
            and len(self.selectors) == 1
            and self.selectors[0].is_singular
        )

    def apply(self, nodes: Sequence[Match], root: Match) -> Tuple[Match, ...]:
        result: List[Match] = []
        for node in nodes:
            candidates = _descendants(node) if self.descendant else (node,)
            for candidate in candidates:
                for selector in self.selectors:
                    result.extend(selector.apply(candidate, root))
        return tuple(result)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Segment)
            and self.descendant == other.descendant
            and self.selectors == other.selectors
        )

    def __hash__(self) -> int:
        return hash((self.descendant, self.selectors))


class Selector:
    is_singular = False

    def apply(self, node: Match, root: Match) -> Tuple[Match, ...]:
        raise NotImplementedError


class NameSelector(Selector):
    is_singular = True

    def __init__(self, name: str):
        self.name = name

    def apply(self, node: Match, root: Match) -> Tuple[Match, ...]:
        if isinstance(node.value, dict) and self.name in node.value:
            child = Match(
                node.value[self.name],
                node.path.child(self.name),
                root.value,
                node,
            )
            return (child,)
        return ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NameSelector) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("name", self.name))


class IndexSelector(Selector):
    is_singular = True

    def __init__(self, index: int):
        self.index = index

    def apply(self, node: Match, root: Match) -> Tuple[Match, ...]:
        if isinstance(node.value, list):
            index = len(node.value) + self.index if self.index < 0 else self.index
            if 0 <= index < len(node.value):
                child = Match(
                    node.value[index],
                    node.path.child(index),
                    root.value,
                    node,
                )
                return (child,)
        return ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IndexSelector) and self.index == other.index

    def __hash__(self) -> int:
        return hash(("index", self.index))


class WildcardSelector(Selector):
    def apply(self, node: Match, root: Match) -> Tuple[Match, ...]:
        if isinstance(node.value, dict):
            return tuple(
                Match(value, node.path.child(name), root.value, node)
                for name, value in node.value.items()
            )
        if isinstance(node.value, list):
            return tuple(
                Match(value, node.path.child(index), root.value, node)
                for index, value in enumerate(node.value)
            )
        return ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WildcardSelector)

    def __hash__(self) -> int:
        return hash("wildcard")


class SliceSelector(Selector):
    def __init__(self, start: Any = None, end: Any = None, step: Any = None):
        self.start = start
        self.end = end
        self.step = 1 if step is None else step

    def apply(self, node: Match, root: Match) -> Tuple[Match, ...]:
        if not isinstance(node.value, list):
            return ()
        indices = range(len(node.value))[self.start:self.end:self.step]
        return tuple(
            Match(node.value[index], node.path.child(index), root.value, node)
            for index in indices
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SliceSelector)
            and self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

    def __hash__(self) -> int:
        return hash(("slice", self.start, self.end, self.step))


class FilterSelector(Selector):
    def __init__(self, expression: "LogicalExpression"):
        self.expression = expression

    def apply(self, node: Match, root: Match) -> Tuple[Match, ...]:
        if isinstance(node.value, dict):
            values = node.value.items()
            return tuple(
                Match(value, node.path.child(name), root.value, node)
                for name, value in values
                if self.expression.evaluate(
                    Match(value, node.path.child(name), root.value, node),
                    root,
                )
            )
        if isinstance(node.value, list):
            return tuple(
                Match(value, node.path.child(index), root.value, node)
                for index, value in enumerate(node.value)
                if self.expression.evaluate(
                    Match(value, node.path.child(index), root.value, node),
                    root,
                )
            )
        return ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FilterSelector) and self.expression == other.expression

    def __hash__(self) -> int:
        return hash(("filter", self.expression))


class LogicalExpression:
    def evaluate(self, current: Match, root: Match) -> bool:
        raise NotImplementedError


class Exists(LogicalExpression):
    def __init__(self, query: JSONPath):
        self.query = query

    def evaluate(self, current: Match, root: Match) -> bool:
        return bool(self.query.evaluate(current, root))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Exists) and self.query == other.query

    def __hash__(self) -> int:
        return hash(("exists", self.query))


class BooleanLiteral(LogicalExpression):
    def __init__(self, value: bool):
        self.value = value

    def evaluate(self, current: Match, root: Match) -> bool:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BooleanLiteral) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("boolean", self.value))


class Not(LogicalExpression):
    def __init__(self, expression: LogicalExpression):
        self.expression = expression

    def evaluate(self, current: Match, root: Match) -> bool:
        return not self.expression.evaluate(current, root)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Not) and self.expression == other.expression

    def __hash__(self) -> int:
        return hash(("not", self.expression))


class BinaryLogical(LogicalExpression):
    def __init__(self, left: LogicalExpression, right: LogicalExpression, operator: str):
        self.left = left
        self.right = right
        self.operator = operator

    def evaluate(self, current: Match, root: Match) -> bool:
        left = self.left.evaluate(current, root)
        right = self.right.evaluate(current, root)
        return left and right if self.operator == "&&" else left or right

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BinaryLogical)
            and self.operator == other.operator
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.operator, self.left, self.right))


class Comparison(LogicalExpression):
    def __init__(self, left: "Comparable", right: "Comparable", operator: str):
        self.left = left
        self.right = right
        self.operator = operator

    def evaluate(self, current: Match, root: Match) -> bool:
        left = _comparable_value(self.left, current, root)
        right = _comparable_value(self.right, current, root)
        if self.operator == "==":
            return _strict_equal(left, right)
        if self.operator == "!=":
            if is_nothing(left) or is_nothing(right):
                return False
            return not _strict_equal(left, right)
        if not _orderable(left) or not _orderable(right):
            return False
        if type(left) is not type(right) and not _both_numbers(left, right):
            return False
        if self.operator == "<":
            return left < right
        if self.operator == "<=":
            return left <= right
        if self.operator == ">":
            return left > right
        return left >= right

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Comparison)
            and self.operator == other.operator
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.operator, self.left, self.right))


class Literal:
    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Literal) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("literal", repr(self.value)))


Comparable = Any


def _comparable_value(operand: Any, current: Match, root: Match) -> Any:
    if isinstance(operand, Literal):
        return operand.value
    if not operand.is_singular:
        raise ValueError("only singular queries may be compared")
    return operand.evaluate(current, root).value()


def _strict_equal(left: Any, right: Any) -> bool:
    if is_nothing(left) or is_nothing(right):
        return is_nothing(left) and is_nothing(right)
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if _both_numbers(left, right):
        return left == right
    return type(left) is type(right) and left == right


def _orderable(value: Any) -> bool:
    return (
        not is_nothing(value)
        and value is not None
        and not isinstance(value, bool)
        and (isinstance(value, (int, float, str)) and not isinstance(value, bool))
    )


def _both_numbers(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    )


def _descendants(node: Match) -> Tuple[Match, ...]:
    result: List[Match] = []
    pending = [node]
    while pending:
        current = pending.pop()
        result.append(current)
        if isinstance(current.value, dict):
            children = [
                Match(value, current.path.child(name), current.root_value, current)
                for name, value in current.value.items()
            ]
            pending.extend(reversed(children))
        elif isinstance(current.value, list):
            children = [
                Match(value, current.path.child(index), current.root_value, current)
                for index, value in enumerate(current.value)
            ]
            pending.extend(reversed(children))
    return tuple(result)
