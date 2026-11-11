"""
Optional diagnostics for JSONPath evaluation.

The diagnostics answer the question "where did my query lose all of its
candidates?" without re-running a simplified interpreter.  Instead, the
real evaluation chain reports each AST step to a collector that lives in
a context variable, so a query evaluated with diagnostics produces
exactly the same results as one evaluated without them.

Usage::

    from jsonpath_ng.diagnostics import find_with_diagnostics

    matches, report = find_with_diagnostics(
        parse("$.store.book[0].price"), data
    )
    if not matches:
        print(report.first_failure())

Sample values are hidden by default so that reports are safe to log;
pass ``include_samples=True`` to opt in to them.
"""

from __future__ import annotations

import contextvars
import dataclasses
import typing

if typing.TYPE_CHECKING:
    from jsonpath_ng.jsonpath import DatumInContext
    from jsonpath_ng.jsonpath import JSONPath

# Reason codes for the first locatable failure of a step.
# These are deliberately coarse so that reports stay stable and greppable.
MISSING_FIELD = "missing-field"
INDEX_OUT_OF_RANGE = "index-out-of-range"
TYPE_MISMATCH = "type-mismatch"
PREDICATE_FALSE = "predicate-false"

#: The maximum number of events collected by default.
DEFAULT_EVENT_BUDGET: typing.Final[int] = 1000

#: Sentinel used when no sample value is recorded.
_NO_SAMPLE: typing.Final[object] = object()

#: The active collector, or ``None`` when diagnostics are disabled.
#: Evaluation nodes check this before reporting, so the overhead of a
#: disabled collector is a single context-variable lookup per node.
_active_collector: contextvars.ContextVar[Collector | None] = (
    contextvars.ContextVar("jsonpath_ng_diagnostics_collector", default=None)
)


@dataclasses.dataclass
class DiagnosticEvent:
    """One step of a JSONPath evaluation.

    ``node`` is the string form of the AST node that ran, ``path`` is
    the full path of the datum the node ran against, and
    ``input_count`` / ``output_count`` are the numbers of candidates
    entering and leaving the step.  ``reason`` is one of the module
    level reason codes, set only when the step produced no candidates
    and a cause could be determined.
    """

    node: str
    path: str
    input_count: int
    output_count: int
    reason: str | None = None
    detail: str | None = None
    sample: typing.Any = _NO_SAMPLE

    def __str__(self) -> str:
        parts = [
            f"node={self.node!r}",
            f"path={self.path!r}",
            f"in={self.input_count}",
            f"out={self.output_count}",
        ]
        if self.reason is not None:
            parts.append(f"reason={self.reason}")
        if self.detail is not None:
            parts.append(f"detail={self.detail}")
        if self.sample is not _NO_SAMPLE:
            parts.append(f"sample={self.sample!r}")
        return "DiagnosticEvent(" + ", ".join(parts) + ")"


@dataclasses.dataclass
class DiagnosticReport:
    """The diagnostics gathered during one ``find_with_diagnostics()`` call."""

    events: list[DiagnosticEvent]
    truncated: bool = False
    event_budget: int = DEFAULT_EVENT_BUDGET
    error: BaseException | None = None

    def first_failure(self) -> DiagnosticEvent | None:
        """Return the first step that produced no candidates.

        The first empty step is the point at which the query first lost
        all of its candidates.  Steps that ran afterwards can never
        restore them, so they are not considered.
        """

        for event in self.events:
            if event.output_count == 0 and event.input_count > 0:
                return event
        return None

    def __iter__(self) -> typing.Iterator[DiagnosticEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)


class Collector:
    """Accumulates diagnostic events while a single query is evaluated."""

    def __init__(
        self,
        event_budget: int = DEFAULT_EVENT_BUDGET,
        include_samples: bool = False,
    ) -> None:
        self.events: list[DiagnosticEvent] = []
        self.truncated = False
        self.event_budget = event_budget
        self.include_samples = include_samples
        self.error: BaseException | None = None

    def record(
        self,
        node: JSONPath,
        datum: typing.Any,
        input_count: int,
        output_count: int,
        reason: str | None = None,
        detail: str | None = None,
        sample: typing.Any = _NO_SAMPLE,
    ) -> None:
        """Record one evaluation step, stopping once the budget is spent."""

        if len(self.events) >= self.event_budget:
            self.truncated = True
            return

        if sample is _NO_SAMPLE or not self.include_samples:
            recorded_sample = _NO_SAMPLE
        else:
            recorded_sample = sample

        self.events.append(
            DiagnosticEvent(
                node=str(node),
                path=_datum_path(datum),
                input_count=input_count,
                output_count=output_count,
                reason=reason,
                detail=detail,
                sample=recorded_sample,
            )
        )

    def report(self) -> DiagnosticReport:
        return DiagnosticReport(
            events=list(self.events),
            truncated=self.truncated,
            event_budget=self.event_budget,
            error=self.error,
        )


def _datum_path(datum: typing.Any) -> str:
    """Render the full path of *datum*, or ``$`` for the query root."""

    from jsonpath_ng.jsonpath import DatumInContext

    if isinstance(datum, DatumInContext):
        try:
            rendered = str(datum.full_path)
        except Exception:
            return "<unrenderable path>"
        # A bare wrapped datum carries the default `This()` path.
        if rendered == "`this`":
            return "$"
        return rendered
    return "$"


def get_collector() -> Collector | None:
    """Return the active collector, or ``None`` when diagnostics are off."""

    return _active_collector.get()


def record_step(
    node: JSONPath,
    datum: typing.Any,
    input_count: int,
    output_count: int,
    reason: str | None = None,
    detail: str | None = None,
    sample: typing.Any = _NO_SAMPLE,
) -> None:
    """Record a step on the active collector, if any."""

    collector = _active_collector.get()
    if collector is not None:
        collector.record(
            node,
            datum,
            input_count,
            output_count,
            reason=reason,
            detail=detail,
            sample=sample,
        )


def find_with_diagnostics(
    path: JSONPath,
    data: typing.Any,
    *,
    event_budget: int = DEFAULT_EVENT_BUDGET,
    include_samples: bool = False,
) -> tuple[list[DatumInContext], DiagnosticReport]:
    """Evaluate *path* against *data*, collecting diagnostics as it runs.

    The query is evaluated exactly once, through the same call chain as
    :meth:`JSONPath.find`; diagnostics are an observer and never change
    the returned matches.  If evaluation raises, the steps that
    completed before the exception are retained on a report attached to
    the exception as ``exc.jsonpath_diagnostics``, and the original
    exception is re-raised unchanged.
    """

    collector = Collector(
        event_budget=event_budget, include_samples=include_samples
    )
    token = _active_collector.set(collector)
    try:
        matches = path.find(data)
    except BaseException as exc:
        collector.error = exc
        # The exception propagates unchanged.  The steps that completed
        # before it stay available on the exception itself, so error
        # paths keep their diagnostics instead of an empty match list.
        exc.jsonpath_diagnostics = collector.report()  # type: ignore[attr-defined]
        raise
    finally:
        _active_collector.reset(token)
    return matches, collector.report()
