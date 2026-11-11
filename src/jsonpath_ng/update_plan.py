"""
Declarative update plans for JSONPath expressions.

A cleaning pipeline typically mixes rules that replace values with rules that
delete non-conforming items. Calling ``JSONPath.update()`` and
``JSONPath.filter()`` one after another is order-sensitive: each step mutates
the document, so later expressions hit different nodes and array indices shift.

An ``UpdatePlan`` resolves every expression against the *same* original
document first, records the concrete positions each rule targets, and only then
applies all the resolved changes in one pass. The plan can be inspected before
it is applied, and applying it either succeeds completely or leaves the input
document untouched.

Only documents built from ordinary ``dict`` and ``list`` containers (plus JSON
scalar values) are supported; replacement and deletion operations are supported.
"""

from __future__ import annotations

import hashlib
import json
import typing

from jsonpath_ng.exceptions import UpdatePlanConflictError
from jsonpath_ng.exceptions import UpdatePlanStaleError
from jsonpath_ng.exceptions import UpdatePlanTypeError
from jsonpath_ng.jsonpath import DatumInContext
from jsonpath_ng.jsonpath import Fields
from jsonpath_ng.jsonpath import Index
from jsonpath_ng.jsonpath import JSONPath

# A path component is either a dict field name or a non-negative list index.
PathPart: typing.TypeAlias = str | int
# A resolved target location. The empty tuple identifies the document root.
CanonicalPath: typing.TypeAlias = tuple[PathPart, ...]
# Mirrors the callable signature supported by ``JSONPath.update()``.
ReplaceCallback: typing.TypeAlias = typing.Callable[
    [typing.Any, typing.Any, typing.Any], typing.Any
]
PlanCallback: typing.TypeAlias = typing.Callable[["PlannedUpdate"], typing.Any]

_DELETE = object()


def canonical_path(datum: DatumInContext) -> CanonicalPath:
    """Return the concrete, normalized location of *datum* in its document."""

    parts: list[PathPart] = []
    node: DatumInContext | None = datum
    while node is not None and node.context is not None:
        segment = node.path
        if isinstance(segment, Index):
            index = segment.indices[0]
            if index < 0:
                # Normalize negative indices against the original parent array
                # so the stored position is always a concrete non-negative one.
                index %= len(node.context.value)
            parts.append(index)
        elif isinstance(segment, Fields):
            parts.append(segment.fields[0])
        else:
            msg = f"Unsupported path segment {segment!r} in an update plan."
            raise UpdatePlanTypeError(msg)
        node = node.context
    return tuple(reversed(parts))


def render_path(path: CanonicalPath) -> str:
    """Render a canonical path in JSONPath-like bracket notation."""

    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f"[{json.dumps(part, ensure_ascii=False)}]"
    return rendered


def document_digest(document: typing.Any) -> str:
    """Return a stable SHA-256 digest that fingerprints *document*."""

    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_document(value: typing.Any) -> None:
    """Reject documents that are not plain dict/list JSON structures."""

    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                msg = (
                    "Update plans only support string object keys; found "
                    f"{type(key).__name__} key {key!r}."
                )
                raise UpdatePlanTypeError(msg)
            validate_document(item)
    elif isinstance(value, list):
        for item in value:
            validate_document(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        msg = (
            "Update plans only support JSON values built from dict and list "
            f"containers; found {type(value).__name__}."
        )
        raise UpdatePlanTypeError(msg)


class PlannedUpdate:
    """One approved position that a plan will change.

    ``path`` is the concrete location in the original document, ``sources``
    names every expression that resolved to this position, and ``value`` is the
    replacement for a ``replace`` operation (``None`` for a ``delete``).
    """

    __slots__ = ("operation", "path", "sources", "value")

    def __init__(
        self,
        operation: typing.Literal["replace", "delete"],
        path: CanonicalPath,
        sources: tuple[str, ...],
        value: typing.Any = None,
    ) -> None:
        self.operation = operation
        self.path = path
        self.sources = sources
        self.value = value

    @property
    def rendered_path(self) -> str:
        return render_path(self.path)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(operation={self.operation!r}, "
            f"path={self.rendered_path!r}, sources={self.sources!r}, "
            f"value={self.value!r})"
        )


class PlanRule:
    """A rule that targets every match of *expression* in the source document.

    With neither *value* nor *callback* the rule deletes its targets. A
    callback receives the same arguments as a ``JSONPath.update()`` callable:
    the current value, the parent container, and the key or index within it.
    Returning ``None`` keeps the value unchanged, mirroring ``update()``.
    """

    __slots__ = ("expression", "operation", "value", "callback")

    def __init__(
        self,
        expression: JSONPath,
        value: typing.Any = _DELETE,
        callback: ReplaceCallback | None = None,
    ) -> None:
        if value is not _DELETE and callback is not None:
            msg = "A rule accepts either a value or a callback, not both."
            raise UpdatePlanTypeError(msg)
        self.expression = expression
        self.value = None if value is _DELETE else value
        self.callback = callback
        self.operation: typing.Literal["replace", "delete"] = (
            "delete" if value is _DELETE and callback is None else "replace"
        )

    @classmethod
    def replace(cls, expression: JSONPath, value: typing.Any) -> PlanRule:
        """Build a replacement rule, even when the replacement is ``None``."""

        return cls(expression, value=value)

    @classmethod
    def replace_with(cls, expression: JSONPath, callback: ReplaceCallback) -> PlanRule:
        """Build a replacement rule driven by a callback."""

        return cls(expression, callback=callback)

    @classmethod
    def delete(cls, expression: JSONPath) -> PlanRule:
        """Build a deletion rule."""

        return cls(expression)

    def resolve(self, document: typing.Any) -> list[DatumInContext]:
        return self.expression.find(document)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(expression={self.expression!r}, "
            f"operation={self.operation!r})"
        )


class UpdatePlan:
    """Rules resolved against one document and applied in a single pass.

    Build plans with :meth:`build`, inspect :attr:`updates` before applying,
    and call :meth:`apply` to mutate the document. Every expression is
    evaluated against the original document, so reordering non-conflicting
    rules never changes which positions are hit or the final result.
    """

    def __init__(
        self,
        document: typing.Any,
        rules: typing.Sequence[PlanRule],
        updates: list[PlannedUpdate],
        digest: str,
    ) -> None:
        self.document = document
        self.rules = tuple(rules)
        self.updates = updates
        self._digest = digest

    # -------------------------------------------------------------- building

    @classmethod
    def build(
        cls, document: typing.Any, rules: typing.Sequence[PlanRule]
    ) -> UpdatePlan:
        """Resolve every rule against *document* and compile the plan.

        All rules are evaluated before any change is made. Conflicting rules
        raise :class:`UpdatePlanConflictError`; unsupported documents raise
        :class:`UpdatePlanTypeError`. In either case *document* is untouched.
        """

        validate_document(document)
        digest = document_digest(document)

        targets: dict[CanonicalPath, list[PlanRule]] = {}
        for rule in rules:
            for datum in rule.resolve(document):
                targets.setdefault(canonical_path(datum), []).append(rule)

        conflicts = cls._detect_conflicts(targets)
        if conflicts:
            raise UpdatePlanConflictError(
                "Incompatible update plan rules: " + "; ".join(conflicts)
            )

        updates: list[PlannedUpdate] = []
        for path, hit_rules in targets.items():
            rule = hit_rules[0]
            sources = tuple(dict.fromkeys(str(hit.expression) for hit in hit_rules))
            if rule.operation == "delete":
                updates.append(PlannedUpdate("delete", path, sources))
            elif rule.callback is not None:
                updates.append(
                    PlannedUpdate("replace", path, sources, value=rule.callback)
                )
            else:
                updates.append(
                    PlannedUpdate("replace", path, sources, value=rule.value)
                )
        # A plan describes positions in one fixed document, so its presentation
        # must not depend on the order in which rules were supplied.
        updates.sort(key=lambda update: update.rendered_path)
        return cls(document, list(rules), updates, digest)

    @classmethod
    def _detect_conflicts(
        cls, targets: dict[CanonicalPath, list[PlanRule]]
    ) -> list[str]:
        conflicts: list[str] = []
        paths = list(targets)
        for path in paths:
            hit_rules = targets[path]
            operations = {rule.operation for rule in hit_rules}
            sources = _render_sources(hit_rules)
            rendered = render_path(path)

            if len(operations) > 1:
                conflicts.append(
                    f"{rendered} is both replaced and deleted by {sources}"
                )
                continue

            if "replace" in operations and cls._incompatible_values(hit_rules):
                values = [
                    rule.callback if rule.callback is not None else rule.value
                    for rule in hit_rules
                ]
                rendered_values = ", ".join(repr(value) for value in values)
                conflicts.append(
                    f"{rendered} receives incompatible replacement values "
                    f"({rendered_values}) from {sources}"
                )

            # A container target also removes or overwrites every descendant,
            # so any rule targeting a nested position conflicts with it.
            for other_path in paths:
                if other_path == path or not _is_relative(other_path, path):
                    continue
                nested_sources = _render_sources(targets[other_path])
                conflicts.append(
                    f"{render_path(other_path)} is nested inside {rendered}, "
                    f"which a {next(iter(operations))} targets; outer sources: "
                    f"{sources}; nested sources: {nested_sources}"
                )
        return conflicts

    @staticmethod
    def _incompatible_values(hit_rules: list[PlanRule]) -> bool:
        values = [
            rule.callback if rule.callback is not None else rule.value
            for rule in hit_rules
        ]
        seen: list[typing.Any] = []
        for value in values:
            for existing in seen:
                if callable(value) or callable(existing):
                    if value is existing:
                        break
                elif _json_equal(value, existing):
                    break
            else:
                seen.append(value)
        return len(seen) > 1

    # ------------------------------------------------------------- inspection

    @property
    def digest(self) -> str:
        """SHA-256 digest of the document the plan was built for."""

        return self._digest

    def is_current(self, document: typing.Any | None = None) -> bool:
        """Return whether *document* still matches the plan's source digest."""

        target = self.document if document is None else document
        try:
            return document_digest(target) == self._digest
        except (TypeError, ValueError):
            return False

    def paths(self) -> list[str]:
        """Render every targeted position, in plan order."""

        return [update.rendered_path for update in self.updates]

    def __len__(self) -> int:
        return len(self.updates)

    def __iter__(self) -> typing.Iterator[PlannedUpdate]:
        return iter(self.updates)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(updates={len(self.updates)}, "
            f"digest={self._digest[:12]!r})"
        )

    # --------------------------------------------------------------- applying

    def apply(
        self,
        callback: PlanCallback | None = None,
        *,
        document: typing.Any | None = None,
    ) -> typing.Any:
        """Apply every planned change once and return the mutated document.

        *callback*, when given, is invoked once per approved position with the
        :class:`PlannedUpdate` before that position is changed. Raising from the
        callback (or from any replacement callback) aborts the whole plan and
        leaves the document unchanged. Applying a plan after its document
        changed raises :class:`UpdatePlanStaleError`.
        """

        target = self.document if document is None else document
        if not self.is_current(target):
            msg = (
                "Refusing to apply an update plan: the document no longer "
                "matches the digest captured when the plan was built."
            )
            raise UpdatePlanStaleError(msg)

        # Resolve replacement values and run approval callbacks before any
        # mutation, so a failure aborts the plan with the document untouched.
        staged: list[tuple[PlannedUpdate, typing.Any]] = []
        for update in self.updates:
            if callback is not None:
                callback(update)
            if update.operation == "delete":
                staged.append((update, _DELETE))
            elif callable(update.value):
                parent, key = _navigate_parent(target, update.path)
                staged.append((update, update.value(parent[key], parent, key)))
            else:
                staged.append((update, update.value))

        return _commit(target, staged)


def _render_sources(hit_rules: list[PlanRule]) -> str:
    return ", ".join(sorted({str(rule.expression) for rule in hit_rules}))


def _is_relative(path: CanonicalPath, ancestor: CanonicalPath) -> bool:
    return len(path) > len(ancestor) and path[: len(ancestor)] == ancestor


def _json_equal(left: typing.Any, right: typing.Any) -> bool:
    """Structural equality for plain JSON values, with bool distinct from int."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) is not type(right):
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _navigate_parent(
    root: typing.Any, path: CanonicalPath
) -> tuple[typing.Any, PathPart]:
    """Return the parent container of *path* and its key within it."""

    if not path:
        msg = "The document root has no parent container."
        raise UpdatePlanConflictError(msg)
    container = root
    for part in path[:-1]:
        container = container[part]
    return container, path[-1]


def _commit(
    root: typing.Any, staged: list[tuple[PlannedUpdate, typing.Any]]
) -> typing.Any:
    """Apply staged changes to *root*, interpreting indices against the origin.

    Replacements are performed first (they preserve container shape), then
    deletions run deepest-first and with descending array indices so every
    position keeps meaning the index it had in the original document.
    """

    root_replacement = _DELETE
    root_deleted = False

    for update, value in staged:
        if not update.path:
            if update.operation == "delete":
                root_deleted = True
            else:
                root_replacement = value
            continue
        if update.operation == "replace" and value is not None:
            parent, key = _navigate_parent(root, update.path)
            parent[key] = value

    deletions = [
        update
        for update, value in staged
        if update.operation == "delete" and update.path
    ]
    # Deepest paths first; among same-depth paths, descending indices for lists
    # guarantees earlier (smaller) indices stay valid until they are removed.
    deletions.sort(key=lambda update: (len(update.path), _sort_key(update.path)))
    for update in deletions:
        parent, key = _navigate_parent(root, update.path)
        if isinstance(parent, list):
            del parent[key]
        else:
            del parent[key]

    if root_deleted:
        return None
    if root_replacement is _DELETE or root_replacement is None:
        # No root replacement, or a root callback returning None: keep root.
        return root
    return root_replacement


def _sort_key(path: CanonicalPath) -> tuple[typing.Any, ...]:
    # Deletions within the same list must run from the highest index down.
    # Field deletions are order-independent; keep a deterministic order.
    return tuple(
        (
            -part if isinstance(part, int) else 0,
            "" if isinstance(part, int) else part,
        )
        for part in path
    )
