class JSONPathError(Exception):
    pass


class JsonPathLexerError(JSONPathError):
    pass


class JsonPathParserError(JSONPathError):
    pass


class TemplateError(JSONPathError):
    """Base class for errors raised while preparing or binding a template."""


class DuplicateParameterError(TemplateError):
    """A parameter name is declared more than once in a template."""


class MissingParameterError(TemplateError):
    """A template executed without a value for one of its parameters."""


class UnsupportedParameterTypeError(TemplateError):
    """A parameter value is not a JSON null, boolean, number, or string."""
