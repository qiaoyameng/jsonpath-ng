class JSONPathError(Exception):
    pass


class JsonPathLexerError(JSONPathError):
    pass


class JsonPathParserError(JSONPathError):
    pass


class UpdatePlanError(JSONPathError):
    """The base class for errors raised while building or applying a plan."""


class UpdatePlanTypeError(UpdatePlanError, TypeError):
    """A plan was built for data containing unsupported container types."""


class UpdatePlanConflictError(UpdatePlanError):
    """Rules target the same position with incompatible operations."""


class UpdatePlanStaleError(UpdatePlanError):
    """A plan is applied after its source document changed."""
