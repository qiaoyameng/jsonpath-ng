"""Long-lived query sessions with incremental JSON Patch maintenance.

This package provides an explicit :class:`QuerySession` that keeps a document
and the current results of one or more JSONPath queries. After applying a
batch of JSON Patch (RFC 6902) operations the session reports the matches that
were added, removed, had their value changed or merely moved position.

Only a declarative subset of JSONPath can be maintained incrementally:

* field access (``$.a.b``),
* array indices (``[0]``, ``[0,2]``),
* slices (``[1:3]``, ``[::2]`` and ``[*]``),
* recursive descent (``..name``, at most one per query),
* the existing ``[?...]`` filters.

Queries containing nodes that cannot be statically analysed (``where``,
``parent``, unions, arithmetic targets, ...) fall back to a full ``find`` and
the reason is exposed in the session statistics.
"""

from .session import (
    AccessCounter,
    MatchDelta,
    PatchBatchError,
    QuerySession,
    SessionQuery,
    SessionStats,
)

__all__ = [
    "AccessCounter",
    "MatchDelta",
    "PatchBatchError",
    "QuerySession",
    "SessionQuery",
    "SessionStats",
]
