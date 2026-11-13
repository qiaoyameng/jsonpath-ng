"""Optional diagnostics for JSONPath evaluation.

Diagnostics are collected from the actual evaluation chain:
instrumented ``find()`` implementations report into the active
:class:`QueryDiagnostics` collector while the query runs,
so no expression is ever re-evaluated to explain a result.

Usage::

    from jsonpath_ng.diagnostics import QueryDiagnostics
    from jsonpath_ng.ext.parser import parse

    with QueryDiagnostics() as diagnostics:
        matches = parse("$.store.book[?@.price < 10].title").find(data)

    print(diagnostics.render())
    print(diagnostics.first_failure)

Collection is opt-in and cheap when disabled: each ``find()`` checks
a module-level reference and skips all event bookkeeping when it is None.
"""

from __future__ import annotations

import itertools
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

# Failure reasons, reported separately so that a renamed field
# is distinguishable from a changed filter condition.
MISSING_FIELD = "missing_field"
INDEX_OUT_OF_RANGE = "index_out_of_range"
TYPE_MISMATCH = "type_mismatch"
PREDICATE_FALSE = "predicate_false"

FAILURE_REASONS = (MISSING_FIELD, INDEX_OUT_OF_RANGE, TYPE_MISMATCH, PREDICATE_FALSE)

# The active collector, or None when diagnostics are disabled.
_active: QueryDiagnostics | None = None

# Monotonic sequence stamping when each failure is recorded,
# so the first failure can be found regardless of when steps complete.
_failure_sequence = itertools.count()

_NOT_SET: typing.Any = object()


class DiagnosticEvent:
    """One step of an evaluation: an AST node applied to its input datum.

    Attributes:
        node: The AST node rendered as query syntax, e.g. ``$.'store'``.
        node_type: The AST class name, e.g. ``Fields``.
        path: The full path of the input datum, if it could be determined.
        input_count: Candidate datums the step started from
            (a filter counts its members; other steps count their one datum).
        output_count: Candidate datums the step produced,
            or None if the step did not complete (e.g. an exception).
        failure: The first locatable failure reason for the step, if any.
        detail: Extra context for the failure, e.g. the missing field name.
        sample_input/sample_output: Sample values, only when enabled.
        children: Steps evaluated as part of this step, in evaluation order.

    The node and path renderings are computed lazily, so an evaluation
    that is never inspected does not pay for them. Until then, events
    keep their input datum alive.
    """

    __slots__ = (
        "node_type",
        "input_count",
        "output_count",
        "failure",
        "detail",
        "sample_input",
        "sample_output",
        "children",
        "_collector",
        "_datum",
        "_node_obj",
        "_node_str",
        "_open",
        "_path_str",
        "_sample_values",
        "_failure_seq",
    )

    def __init__(
        self,
        collector: QueryDiagnostics,
        node: typing.Any,
        datum: typing.Any,
    ) -> None:
        self._collector = collector
        self._open = True
        self._sample_values = collector.include_values
        self._node_obj = node
        self._node_str: str | None = None
        self._datum = datum
        self._path_str: typing.Any = _NOT_SET
        self.node_type = type(node).__name__
        self.input_count = 1
        self.output_count: int | None = None
        self.failure: str | None = None
        self.detail: str | None = None
        self.sample_input = (
            _sample(_datum_value(datum)) if self._sample_values else None
        )
        self.sample_output: str | None = None
        self.children: list[DiagnosticEvent] = []
        self._failure_seq: int | None = None

    @property
    def node(self) -> str:
        """The AST node rendered as query syntax."""

        if self._node_str is None:
            self._node_str = _node_to_str(self._node_obj)
        return self._node_str

    @property
    def path(self) -> str | None:
        """The full path of the input datum, if it could be determined."""

        if self._path_str is _NOT_SET:
            self._path_str = _datum_path(self._datum)
            self._datum = None
        return self._path_str

    def set_output(self, result: typing.Any) -> None:
        try:
            self.output_count = len(result)
        except TypeError:
            self.output_count = 1 if result is not None else 0
        if self._sample_values:
            try:
                first = next(iter(result), None)
            except TypeError:
                first = result
            self.sample_output = _sample(_datum_value(first))

    def set_failure(self, reason: str, detail: str | None = None) -> None:
        # Only the first locatable failure of a step is kept.
        if self.failure is None:
            self.failure = reason
            self.detail = detail
            self._failure_seq = next(_failure_sequence)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_type} {self.node!r}, "
            f"in={self.input_count}, out={self.output_count}, "
            f"failure={self.failure!r})"
        )


class QueryDiagnostics:
    """Collects diagnostic events for one query evaluation.

    Use as a context manager around the ``find()`` call to inspect.
    The query result is unaffected; an exception raised by the query
    propagates unchanged, with the completed steps kept on the collector.

    Arguments:
        max_events: Budget of recorded events. Once reached, no further
            events are recorded and `truncated` is set.
        include_values: Sample input/output values on each event.
            Hidden by default so that diagnostics are safe to log.
    """

    def __init__(self, *, max_events: int = 1000, include_values: bool = False) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self.max_events = max_events
        self.include_values = include_values
        self.events: list[DiagnosticEvent] = []
        self.truncated = False
        self.error: BaseException | None = None
        self._stack: list[DiagnosticEvent] = []
        self._ordered: list[DiagnosticEvent] = []
        self._previous: QueryDiagnostics | None = None

    def __enter__(self) -> QueryDiagnostics:
        global _active
        self._previous = _active
        _active = self
        return self

    def __exit__(
        self, exc_type: typing.Any, exc: BaseException | None, tb: typing.Any
    ) -> bool:
        global _active
        _active = self._previous
        self._stack.clear()
        if exc is not None:
            # Keep the original exception; it is never converted to an
            # empty match, and the steps completed so far stay recorded.
            self.error = exc
        return False

    def _start(self, node: typing.Any, datum: typing.Any) -> DiagnosticEvent | None:
        if len(self._ordered) >= self.max_events:
            self.truncated = True
            return None
        event = DiagnosticEvent(self, node, datum)
        self._ordered.append(event)
        if self._stack:
            self._stack[-1].children.append(event)
        else:
            self.events.append(event)
        self._stack.append(event)
        return event

    def _finish(self, event: DiagnosticEvent, output: typing.Any = _NOT_SET) -> None:
        if not event._open:
            return
        event._open = False
        if output is not _NOT_SET:
            event.set_output(output)
        if self._stack and self._stack[-1] is event:
            self._stack.pop()

    def iter_events(self) -> Iterator[DiagnosticEvent]:
        """Yield every recorded event in evaluation (pre-order) order."""

        yield from self._ordered

    @property
    def event_count(self) -> int:
        return len(self._ordered)

    @property
    def first_failure(self) -> DiagnosticEvent | None:
        """The step whose failure was recorded first during evaluation."""

        failures = [event for event in self._ordered if event.failure is not None]
        if not failures:
            return None
        return min(failures, key=lambda event: typing.cast(int, event._failure_seq))

    @property
    def truncation_note(self) -> str | None:
        if not self.truncated:
            return None
        return (
            f"event budget of {self.max_events} reached; "
            "diagnostics truncated, query results unaffected"
        )

    def render(self) -> str:
        """Render the recorded steps as an indented tree of counts."""

        lines: list[str] = []

        def visit(event: DiagnosticEvent, depth: int) -> None:
            line = (
                f"{'  ' * depth}{event.node_type} {event.node} "
                f"in={event.input_count} out={event.output_count}"
            )
            if event.path is not None:
                line += f" at {event.path}"
            if event.failure is not None:
                line += f" FAILED {event.failure}"
                if event.detail:
                    line += f": {event.detail}"
            lines.append(line)
            for child in event.children:
                visit(child, depth + 1)

        for event in self.events:
            visit(event, 0)
        if self.truncated:
            lines.append(f"... {self.truncation_note}")
        if self.error is not None:
            lines.append(f"evaluation raised {type(self.error).__name__}: {self.error}")
        return "\n".join(lines)


def start(node: typing.Any, datum: typing.Any) -> DiagnosticEvent | None:
    """Start a diagnostic step for `node`, or return None when disabled."""

    collector = _active
    if collector is None:
        return None
    return collector._start(node, datum)


def finish(event: DiagnosticEvent | None, output: typing.Any = _NOT_SET) -> None:
    """Close a step, recording its output; safe to call more than once."""

    if event is not None:
        event._collector._finish(event, output)


def record_failure(
    event: DiagnosticEvent | None, reason: str, detail: str | None = None
) -> None:
    if event is not None:
        event.set_failure(reason, detail)


def _node_to_str(node: typing.Any) -> str:
    try:
        return str(node)
    except Exception:
        return repr(node)


def _datum_path(datum: typing.Any) -> str | None:
    full_path = getattr(datum, "full_path", None)
    if full_path is None:
        return None
    try:
        return str(full_path)
    except Exception:
        return None


def _datum_value(datum: typing.Any) -> typing.Any:
    return getattr(datum, "value", datum)


def _sample(value: typing.Any, limit: int = 60) -> str:
    text = repr(value)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text
