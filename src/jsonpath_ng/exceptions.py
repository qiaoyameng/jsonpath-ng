class JSONPathError(Exception):
    pass


class JsonPathLexerError(JSONPathError):
    pass


class JsonPathParserError(JSONPathError):
    pass


class SQLiteCompilationError(JSONPathError):
    """A JSONPath AST cannot be compiled to SQLite JSON1 SQL."""
