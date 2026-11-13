import typing


class JSONPathError(Exception):
    pass


class JsonPathLexerError(JSONPathError):
    pass


class JsonPathParserError(JSONPathError):
    pass


class UpdatePlanError(JSONPathError):
    """Base class for errors raised by the update planner."""


class UnsupportedUpdateDataError(UpdatePlanError, TypeError):
    """The document is not a plain, acyclic tree of dicts, lists, and scalars."""


class UpdatePlanStaleError(UpdatePlanError, RuntimeError):
    """The document changed after the plan was created."""


class UpdatePlanConflictError(UpdatePlanError):
    """One or more rules conflict, so the plan cannot be applied."""

    def __init__(
        self,
        conflicts: typing.Iterable[typing.Any],
        plan: typing.Any = None,
    ) -> None:
        self.conflicts = tuple(conflicts)
        self.plan = plan
        first = self.conflicts[0] if self.conflicts else None
        self.path = first.path if first is not None else None
        self.expression = first.expression if first is not None else None
        self.related_path = first.related_path if first is not None else None
        self.related_expression = (
            first.related_expression if first is not None else None
        )
        message = "update plan has conflicts: " + "; ".join(
            conflict.describe() for conflict in self.conflicts
        )
        super().__init__(message)


UpdatePlanConflict = UpdatePlanConflictError
UpdatePlanStale = UpdatePlanStaleError
StaleUpdatePlanError = UpdatePlanStaleError
ConflictUpdatePlanError = UpdatePlanConflictError
