Added
-----

*   Add optional query diagnostics in ``jsonpath_ng.diagnostics``.

    Wrapping a ``find()`` call in ``QueryDiagnostics()`` records each
    evaluation step's input and output candidate counts and the first
    locatable failure reason (missing field, index out of range,
    type mismatch, or false predicate), collected from the real
    evaluation chain without re-evaluating the query.
