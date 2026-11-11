Added
-----

*   Add parameterized query templates via ``jsonpath_ng.ext.parse_template()``.

    A template is parsed once and executed with a parameter dictionary for
    each run. Placeholders use the ``$name`` syntax and may only appear as a
    comparison value in a filter expression, so a parameter can never inject
    a field name, an operator, or a path segment. Parameter values may be
    null, booleans, numbers, or strings, and their types are preserved during
    comparison. Missing parameters, unknown parameters, and unsupported value
    types raise ``JSONPathParameterError`` before the document is accessed.
