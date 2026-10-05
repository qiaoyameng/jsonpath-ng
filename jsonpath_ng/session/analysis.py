"""Static analysis of JSONPath queries for incremental maintenance.

A maintainable query is flattened into a *spine* of selection steps:

* ``("field", frozenset[str])``     - one or more object members (incl. ``*``)
* ``("index", tuple[int, ...])``    - non-negative array positions
* ``("slice", start, stop, step)``  - array slice, Python semantics
* ("desc", kind, ...)               - recursive descent over the step below
* ("filter", targets)               - ``[?@.field <op> value]`` expressions

Queries containing nodes outside the declarative subset are rejected with an
:class:`UnsupportedQuery`; the session then falls back to a complete ``find``
and reports the reason.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Tuple

from jsonpath_ng.jsonpath import (
    Child,
    Descendants,
    Fields,
    Index,
    Root,
    Slice,
    This,
)
from jsonpath_ng.ext.filter import Expression, Filter


class Step(NamedTuple):
    kind: str
    payload: Any = None


class UnsupportedQuery(ValueError):
    """The query uses a node the incremental engine cannot analyze."""


def _normalise_slice(node: Slice) -> Step:
    step = node.step if node.step is not None else 1
    if step == 0:
        raise UnsupportedQuery("slice step of zero")
    return Step("slice", (node.start, node.end, step))


def _normalise_index(node: Index) -> Step:
    indices = tuple(node.indices)
    if not indices or any(
        not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices
    ):
        raise UnsupportedQuery("only non-negative integer indices are supported")
    return Step("index", tuple(sorted(indices)))


def _normalise_fields(node: Fields) -> Step:
    names = tuple(node.fields)
    if not names:
        raise UnsupportedQuery("empty field selector")
    return Step(
        "field",
        ("*" if "*" in names else "set", frozenset(names)),
    )


def _filter_target(expr: Any) -> Tuple[str, str, Any]:
    if not isinstance(expr, Expression):
        raise UnsupportedQuery(f"unsupported filter expression {expr!r}")
    target = expr.target
    # supported: @.field  (This . Fields(single concrete name))
    if not (
        isinstance(target, Child)
        and isinstance(target.left, This)
        and isinstance(target.right, Fields)
        and "*" not in target.right.fields
        and len(target.right.fields) == 1
    ):
        raise UnsupportedQuery(
            "only '@.<field>' comparison/existence filters are supported"
        )
    field = target.right.fields[0]
    if expr.op is not None and expr.op not in (
        "==",
        "=",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        "=~",
    ):
        raise UnsupportedQuery(f"unsupported filter operator {expr.op!r}")
    return (field, expr.op, expr.value)


def _normalise_filter(node: Filter) -> Step:
    if not node.expressions:
        return Step("filter", ())
    targets = tuple(_filter_target(expr) for expr in node.expressions)
    return Step("filter", targets)


def _selector_step(node: Any) -> Step:
    if isinstance(node, Fields):
        return _normalise_fields(node)
    if isinstance(node, Index):
        return _normalise_index(node)
    if isinstance(node, Slice):
        return _normalise_slice(node)
    if isinstance(node, Filter):
        return _normalise_filter(node)
    raise UnsupportedQuery(f"unsupported selector node {type(node).__name__}")


def _flatten(node: Any, steps: List[Step], descendants_seen: int) -> None:
    if isinstance(node, Root) or isinstance(node, This):
        return
    if isinstance(node, Child):
        _flatten(node.left, steps, descendants_seen)
        steps.append(_selector_step(node.right))
        return
    if isinstance(node, Descendants):
        if descendants_seen:
            raise UnsupportedQuery("multiple recursive descents are not supported")
        _flatten(node.left, steps, descendants_seen + 1)
        inner = _selector_step(node.right)
        steps.append(Step("desc", inner))
        return
    # A bare selector without an explicit root is allowed (jsonpath-ng parses
    # "a.b" without a leading '$').
    steps.append(_selector_step(node))


def analyze(expr: Any) -> List[Step]:
    """Return the maintainable spine for *expr* or raise UnsupportedQuery."""
    steps: List[Step] = []
    _flatten(expr, steps, 0)
    if not steps:
        raise UnsupportedQuery("query matches only the root document")
    return steps




