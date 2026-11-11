Added
-----

*   Add optional query diagnostics via ``jsonpath_ng.diagnostics.find_with_diagnostics()``.

    The real evaluation chain records, for every AST step, the number of
    candidates entering and leaving the step, plus the first locatable
    failure reason (``missing-field``, ``index-out-of-range``,
    ``type-mismatch``, or ``predicate-false``).  Sample values are hidden
    by default, collection stops at a configurable event budget, and
    exceptions keep the completed steps on ``exc.jsonpath_diagnostics``
    while propagating unchanged.
