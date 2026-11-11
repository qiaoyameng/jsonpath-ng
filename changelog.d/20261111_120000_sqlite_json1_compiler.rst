Added
-----

*   Add ``jsonpath_ng.ext.sqlite.compile_jsonpath()`` to compile supported
    JSONPath ASTs into SQLite JSON1 SQL.

    The entry point returns the SQL fragment, bind parameters, and a
    description of the supported scope, and produces each match with its
    value, JSON1 type, and normalized path. Field selectors, array indices
    and wildcards, scalar comparisons, and ``&&``/``||`` filters are
    supported; unsupported nodes raise ``SQLiteCompilationError`` before
    execution instead of being silently dropped.
