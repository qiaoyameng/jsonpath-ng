from __future__ import annotations

import typing

from jsonpath_ng.exceptions import JSONPathParameterError
from jsonpath_ng.jsonpath import BoundParameter
from jsonpath_ng.jsonpath import Child
from jsonpath_ng.jsonpath import Descendants
from jsonpath_ng.jsonpath import JSONPath
from jsonpath_ng.jsonpath import Placeholder
from jsonpath_ng.jsonpath import Where
from jsonpath_ng.jsonpath import WhereNot
from jsonpath_ng.jsonpath import _Intersect
from jsonpath_ng.jsonpath import _Union

if typing.TYPE_CHECKING:
    from jsonpath_ng.jsonpath import DatumInContext

# AST nodes that combine a left and a right sub-expression.
_BINARY_NODES: tuple[type, ...] = (
    Child,
    Descendants,
    Where,
    WhereNot,
    _Union,
    _Intersect,
)


def _child_nodes(node: JSONPath) -> typing.Iterable[JSONPath]:
    """Yield the JSONPath nodes nested directly inside ``node``.

    Only structural AST nodes are traversed; literal leaves such as strings
    and numbers are returned as-is elsewhere and never inspected.
    """

    if isinstance(node, _BINARY_NODES):
        yield node.left
        yield node.right
        return

    # Imported lazily to avoid importing the extension package at module load.
    from jsonpath_ng.ext.arithmetic import Operation
    from jsonpath_ng.ext.filter import Expression
    from jsonpath_ng.ext.filter import Filter
    from jsonpath_ng.ext.filter import _BooleanOperation
    from jsonpath_ng.ext.iterable import SortedThis

    if isinstance(node, Filter):
        yield from node.expressions
    elif isinstance(node, Expression):
        yield node.target
        # The right-hand side is a scalar, a query (JSONPath), or a Placeholder.
        if isinstance(node.value, JSONPath):
            yield node.value
    elif isinstance(node, _BooleanOperation):
        yield from node.operands
    elif isinstance(node, Operation):
        if isinstance(node.left, JSONPath):
            yield node.left
        if isinstance(node.right, JSONPath):
            yield node.right
    elif isinstance(node, SortedThis) and node.expressions:
        for field, _reverse in node.expressions:
            yield field


def _collect_placeholders(
    node: JSONPath, seen: dict[str, Placeholder] | None = None
) -> dict[str, Placeholder]:
    """Return every placeholder in the tree keyed by parameter name."""

    if seen is None:
        seen = {}
    if isinstance(node, Placeholder):
        seen.setdefault(node.name, node)
        return seen
    for child in _child_nodes(node):
        _collect_placeholders(child, seen)
    return seen


def _validate_value(name: str, value: typing.Any) -> None:
    """Reject values that are not JSON scalars, before any document access."""

    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        return
    raise JSONPathParameterError(
        f"Parameter ${name} has an unsupported value type "
        f"{type(value).__name__}; expected null, boolean, number, or string"
    )


def _bind_node(
    node: JSONPath, bindings: typing.Mapping[str, BoundParameter]
) -> JSONPath:
    """Return a copy of the AST with every placeholder replaced by its binding."""

    if isinstance(node, Placeholder):
        return bindings[node.name]

    from jsonpath_ng.ext.arithmetic import Operation
    from jsonpath_ng.ext.filter import Expression
    from jsonpath_ng.ext.filter import Filter
    from jsonpath_ng.ext.filter import _BooleanOperation
    from jsonpath_ng.ext.iterable import SortedThis

    if isinstance(node, _BINARY_NODES):
        return type(node)(
            _bind_node(node.left, bindings), _bind_node(node.right, bindings)
        )
    if isinstance(node, Filter):
        return Filter([_bind_node(expr, bindings) for expr in node.expressions])
    if isinstance(node, Expression):
        value: typing.Any = node.value
        if isinstance(value, JSONPath):
            value = _bind_node(value, bindings)
        return Expression(_bind_node(node.target, bindings), node.op, value)
    if isinstance(node, _BooleanOperation):
        return type(node)([_bind_node(operand, bindings) for operand in node.operands])
    if isinstance(node, Operation):
        left = (
            _bind_node(node.left, bindings)
            if isinstance(node.left, JSONPath)
            else node.left
        )
        right = (
            _bind_node(node.right, bindings)
            if isinstance(node.right, JSONPath)
            else node.right
        )
        return Operation(left, node.op_symbol, right)
    if isinstance(node, SortedThis) and node.expressions:
        return SortedThis(
            [
                (_bind_node(field, bindings), reverse)
                for field, reverse in node.expressions
            ]
        )
    # Leaf nodes without parameters (Fields, Index, Slice, Root, This, ...).
    return node


class ParameterizedJSONPath:
    """A parsed query template containing one or more ``$name`` placeholders.

    The template is parsed once; each :meth:`find` call receives its own
    parameter dictionary and evaluates against a fresh AST copy. The parsed
    tree and the placeholder nodes are never mutated, so the same template can
    be reused concurrently with different bindings.
    """

    def __init__(self, expression: JSONPath) -> None:
        placeholders = _collect_placeholders(expression)
        if not placeholders:
            raise JSONPathParameterError(
                "The query declares no parameters; use jsonpath_ng.ext.parse() "
                "to parse it directly"
            )
        self.expression = expression
        self.parameter_names: tuple[str, ...] = tuple(placeholders)

    def bind(self, parameters: typing.Mapping[str, typing.Any]) -> JSONPath:
        """Validate ``parameters`` and return a bound, ready-to-evaluate AST."""

        missing = [name for name in self.parameter_names if name not in parameters]
        if missing:
            rendered = ", ".join(f"${name}" for name in missing)
            raise JSONPathParameterError(f"Missing parameter(s): {rendered}")

        extra = sorted(set(parameters) - set(self.parameter_names))
        if extra:
            rendered = ", ".join(f"${name}" for name in extra)
            raise JSONPathParameterError(f"Unknown parameter(s): {rendered}")

        for name in self.parameter_names:
            _validate_value(name, parameters[name])

        bindings = {
            name: BoundParameter(name, parameters[name])
            for name in self.parameter_names
        }
        return _bind_node(self.expression, bindings)

    def find(
        self, data: typing.Any, parameters: typing.Mapping[str, typing.Any]
    ) -> list[DatumInContext]:
        """Execute the template on ``data`` with the given parameter values."""

        return self.bind(parameters).find(data)

    def __str__(self) -> str:
        return str(self.expression)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.expression!r})"

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, ParameterizedJSONPath)
            and self.expression == other.expression
        )


def parse_template(template: str) -> ParameterizedJSONPath:
    """Parse a parameterized query exactly once.

    Placeholders use the ``$name`` syntax and may only appear as a filter
    comparison value. Parsing rejects any other position, so a parameter can
    never inject a field name, an operator, or a path segment.
    """

    # Imported lazily: the extension package imports this module.
    from jsonpath_ng.ext.parser import ExtendedJsonPathParser

    expression = ExtendedJsonPathParser().parse(template)
    return ParameterizedJSONPath(expression)
