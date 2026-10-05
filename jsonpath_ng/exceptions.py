class JSONPathError(Exception):
    pass


class JsonPathLexerError(JSONPathError):
    pass


class JsonPathParserError(JSONPathError):
    pass


class JsonPatchError(JSONPathError):
    """Raised when a JSON Patch (RFC 6902) operation cannot be applied."""
