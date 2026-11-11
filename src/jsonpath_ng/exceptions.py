class JSONPathError(Exception):
    pass


class JSONPathParameterError(JSONPathError):
    """A parameterized query was declared or bound incorrectly."""


class JsonPathLexerError(JSONPathError):
    pass


class JsonPathParserError(JSONPathError):
    pass
