class JSONPathError(Exception):
    pass


class JsonPathLexerError(JSONPathError):
    pass


class JsonPathParserError(JSONPathError):
    pass


class JSON1CompileError(JSONPathError):
    """Raised when a JSONPath AST cannot be compiled to a SQLite JSON1 query.

    The error is raised during compilation, before any SQL is executed,
    so an unsupported condition is never silently dropped.
    """
