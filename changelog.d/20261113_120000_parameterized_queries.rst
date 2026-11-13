Added
-----

*   Support parameterized queries through ``jsonpath_ng.ext.parse_template()``.
    A template is parsed once and reuses its AST on every execution,
    with ``:name`` placeholders accepted only as filter comparison values
    and bound from a parameter dictionary at execution time.
