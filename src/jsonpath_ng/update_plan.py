from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import typing
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field

from jsonpath_ng.exceptions import ConflictUpdatePlanError
from jsonpath_ng.exceptions import StaleUpdatePlanError
from jsonpath_ng.exceptions import UnsupportedUpdateDataError
from jsonpath_ng.exceptions import UpdatePlanConflict
from jsonpath_ng.exceptions import UpdatePlanConflictError
from jsonpath_ng.exceptions import UpdatePlanError
from jsonpath_ng.exceptions import UpdatePlanStale
from jsonpath_ng.exceptions import UpdatePlanStaleError
from jsonpath_ng.jsonpath import DatumInContext
from jsonpath_ng.jsonpath import Fields
from jsonpath_ng.jsonpath import Index
from jsonpath_ng.jsonpath import JSONPath
from jsonpath_ng.jsonpath import Root

UpdateAction: typing.TypeAlias = typing.Literal["replace", "delete"]
Replacement: typing.TypeAlias = typing.Any
Address: typing.TypeAlias = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class UpdateConflict:
    """A pair of rules that cannot be applied together."""

    kind: str
    path: str
    expression: str
    related_path: str | None = None
    related_expression: str | None = None
    expressions: tuple[str, ...] = ()
    related_expressions: tuple[str, ...] = ()

    @property
    def type(self) -> str:
        """An alias for ``kind``."""

        return self.kind

    @property
    def canonical_path(self) -> str:
        return self.path

    @property
    def source_expression(self) -> str:
        """The primary source expression."""

        return self.expression

    @property
    def sources(self) -> tuple[str, ...]:
        return self.expressions or (self.expression,)

    @property
    def related_sources(self) -> tuple[str, ...]:
        if self.related_expressions:
            return self.related_expressions
        if self.related_expression is not None:
            return (self.related_expression,)
        return ()

    def describe(self) -> str:
        if self.kind == "replace-delete":
            expressions = ", ".join(repr(item) for item in self.expressions)
            return (
                f"replace/delete conflict at {self.path} "
                f"from expressions {expressions}"
            )
        if self.kind == "replace-replace":
            expressions = ", ".join(repr(item) for item in self.expressions)
            return (
                f"replace/replace conflict at {self.path} "
                f"from expressions {expressions}"
            )
        if self.kind == "root-delete":
            return f"cannot delete the document root at {self.path}"
        return (
            f"parent/child conflict at {self.path} and {self.related_path} "
            f"from expressions {self.expression!r} and "
            f"{self.related_expression!r}"
        )

    def __str__(self) -> str:
        return self.describe()


@dataclass(frozen=True, slots=True)
class UpdateOperation:
    """One approved write to one physical container position."""

    action: UpdateAction
    path: str
    expression: str
    sources: tuple[str, ...]
    value: Replacement = None
    key: str | int | None = None
    source: str | None = None
    _address: Address = field(default=(), repr=False, compare=False)
    _target_value: typing.Any = field(default=None, repr=False, compare=False)
    _ancestors: tuple[int, ...] = field(
        default=(), repr=False, compare=False
    )

    @property
    def canonical_path(self) -> str:
        return self.path

    @property
    def absolute_path(self) -> str:
        return self.path

    @property
    def type(self) -> UpdateAction:
        return self.action

    @property
    def expressions(self) -> tuple[str, ...]:
        return self.sources

    @property
    def replacement(self) -> Replacement:
        return self.value

    @property
    def original_value(self) -> typing.Any:
        return self._target_value

    @property
    def is_replace(self) -> bool:
        return self.action == "replace"

    @property
    def is_delete(self) -> bool:
        return self.action == "delete"

    def to_dict(self) -> dict[str, typing.Any]:
        result: dict[str, typing.Any] = {
            "action": self.action,
            "path": self.path,
            "expressions": self.sources,
        }
        if self.action == "replace":
            result["value"] = self.value
        return result

    def __str__(self) -> str:
        if self.action == "delete":
            return f"delete {self.path}"
        return f"replace {self.path} with {self.value!r}"


@dataclass(frozen=True, slots=True)
class UpdateRule:
    """A declarative replacement or deletion rule."""

    action: UpdateAction
    expression: str
    value: Replacement = None
    source: str | None = None
    path: JSONPath = field(repr=False, compare=False, default_factory=Root)


@dataclass(frozen=True, slots=True)
class _Candidate:
    rule: UpdateRule
    address: Address
    key: str | int | None
    path: str
    physical_key: tuple[int, str | int] | tuple[str]
    order: int
    value: typing.Any
    ancestors: tuple[int, ...]


_SIMPLE_FIELD = re.compile(r"^@?[A-Za-z_][A-Za-z0-9_@-]*$")


def _canonical_path(address: Address) -> str:
    path = "$"
    for part in address:
        if isinstance(part, int):
            path += f"[{part}]"
        elif _SIMPLE_FIELD.match(part):
            path += f".{part}"
        else:
            path += f"[{json.dumps(part, ensure_ascii=False)}]"
    return path


def _canonical_path_for_validation(path: str, part: str | int) -> str:
    if isinstance(part, int):
        return f"{path}[{part}]"
    return f"{path}[{part!r}]"


def _validate_data(value: typing.Any, *, path: str = "$") -> None:
    """Reject data outside the supported first-version data model."""

    active: set[int] = set()

    def visit(item: typing.Any, current_path: str) -> None:
        if type(item) is dict:
            if id(item) in active:
                raise UnsupportedUpdateDataError(
                    f"cycle detected at {current_path}"
                )
            active.add(id(item))
            for key, child in item.items():
                if not isinstance(key, str):
                    raise UnsupportedUpdateDataError(
                        f"dict key at {current_path} is not a string: {key!r}"
                    )
                visit(child, _canonical_path_for_validation(current_path, key))
            active.remove(id(item))
            return
        if type(item) is list:
            if id(item) in active:
                raise UnsupportedUpdateDataError(
                    f"cycle detected at {current_path}"
                )
            active.add(id(item))
            for index, child in enumerate(item):
                visit(
                    child,
                    _canonical_path_for_validation(current_path, index),
                )
            active.remove(id(item))
            return
        if item is None:
            return
        if type(item) not in (bool, int, float, str):
            raise UnsupportedUpdateDataError(
                f"unsupported value at {current_path}: "
                f"{type(item).__name__}"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise UnsupportedUpdateDataError(
                f"non-finite number at {current_path}: {item!r}"
            )

    if type(value) not in (dict, list):
        raise UnsupportedUpdateDataError(
            "the document root must be a plain dict or list"
        )
    visit(value, path)


def _digest(document: typing.Any) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_expression(expression: str | JSONPath) -> tuple[str, JSONPath]:
    if isinstance(expression, JSONPath):
        return str(expression), expression
    if not isinstance(expression, str):
        raise TypeError("expression must be a string or a JSONPath")

    from jsonpath_ng.ext import parse

    return expression, parse(expression)


_ACTION_ALIASES = {
    "delete": "delete",
    "filter": "delete",
    "remove": "delete",
    "replace": "replace",
    "set": "replace",
    "update": "replace",
}


def _normalize_rule(
    rule: UpdateRule | Mapping[str, typing.Any] | tuple,
) -> UpdateRule:
    if isinstance(rule, UpdateRule):
        if (
            not isinstance(rule.action, str)
            or rule.action not in _ACTION_ALIASES
        ):
            raise ValueError(f"unsupported update action: {rule.action!r}")
        action = _ACTION_ALIASES[rule.action]
        expression, path = _parse_expression(rule.expression)
        return UpdateRule(
            action=action,
            expression=expression,
            value=None if action == "delete" else rule.value,
            source=rule.source,
            path=path,
        )

    if isinstance(rule, Mapping):
        action = (
            rule.get("action")
            or rule.get("operation")
            or rule.get("op")
            or rule.get("type")
        )
        expression = (
            rule.get("expression")
            or rule.get("path")
            or rule.get("jsonpath")
        )
        value = rule.get("value", rule.get("replacement"))
        source = rule.get("source", rule.get("rule"))
    elif isinstance(rule, (tuple, list)):
        if len(rule) == 3:
            first, second, third = rule
            first_is_action = (
                isinstance(first, str) and first in _ACTION_ALIASES
            )
            second_is_action = (
                isinstance(second, str) and second in _ACTION_ALIASES
            )
            if first_is_action:
                action, expression, value = first, second, third
            elif second_is_action:
                action, expression, value = second, first, third
            else:
                raise ValueError(f"unsupported update action: {first!r}")
            source = None
        elif len(rule) == 2:
            first, second = rule
            first_is_action = (
                isinstance(first, str) and first in _ACTION_ALIASES
            )
            if first_is_action:
                action, expression = first, second
                value = None
            else:
                action, expression, value = "replace", first, second
            source = None
        else:
            raise TypeError(
                "a rule tuple must contain (action, expression, value)"
            )
    else:
        raise TypeError(f"unsupported rule type: {type(rule).__name__}")

    if not isinstance(action, str) or action not in _ACTION_ALIASES:
        raise ValueError(f"unsupported update action: {action!r}")
    action = _ACTION_ALIASES[action]
    if expression is None:
        raise ValueError("a rule must provide an expression")
    if action == "delete":
        value = None
    parsed_expression, path = _parse_expression(expression)
    return UpdateRule(
        action=action,
        expression=parsed_expression,
        value=value,
        source=source,
        path=path,
    )


def _same_replacement(left: typing.Any, right: typing.Any) -> bool:
    if callable(left) or callable(right):
        return left is right
    return left == right


class UpdatePlan:
    """
    A resolved collection of replacement and deletion rules.

    Expressions are evaluated against a private snapshot taken when the plan
    is created. ``apply()`` evaluates no expressions again; it applies the
    already-approved operations to a working copy and commits only after every
    callback succeeds.
    """

    def __init__(
        self,
        document: typing.Any,
        rules: Iterable[
            UpdateRule | Mapping[str, typing.Any] | tuple
        ] | None = None,
    ) -> None:
        _validate_data(document)
        self._document = document
        self._snapshot = copy.deepcopy(document)
        self._digest = _digest(document)
        self._rules: list[UpdateRule] = []
        self._operations: tuple[UpdateOperation, ...] = ()
        self._conflicts: tuple[UpdateConflict, ...] = ()
        self._resolved = False
        self._order = 0
        if rules is not None:
            for rule in rules:
                self.add_rule(rule)

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def document_digest(self) -> str:
        return self._digest

    @property
    def source_digest(self) -> str:
        return self._digest

    @property
    def document(self) -> typing.Any:
        return self._document

    @property
    def original(self) -> typing.Any:
        return self._document

    @property
    def source(self) -> typing.Any:
        return self._document

    @property
    def snapshot(self) -> typing.Any:
        return copy.deepcopy(self._snapshot)

    @property
    def rules(self) -> tuple[UpdateRule, ...]:
        return tuple(self._rules)

    @property
    def operations(self) -> tuple[UpdateOperation, ...]:
        self._resolve()
        return self._operations

    @property
    def actions(self) -> tuple[UpdateOperation, ...]:
        return self.operations

    @property
    def changes(self) -> tuple[UpdateOperation, ...]:
        return self.operations

    @property
    def entries(self) -> tuple[UpdateOperation, ...]:
        return self.operations

    @property
    def conflicts(self) -> tuple[UpdateConflict, ...]:
        self._resolve()
        return self._conflicts

    @property
    def valid(self) -> bool:
        return not self.conflicts

    @property
    def is_valid(self) -> bool:
        return self.valid

    def add_rule(
        self,
        rule: UpdateRule | Mapping[str, typing.Any] | tuple,
    ) -> UpdatePlan:
        normalized = _normalize_rule(rule)
        self._rules.append(normalized)
        self._resolved = False
        return self

    def add(
        self,
        action: UpdateAction,
        expression: str | JSONPath,
        value: Replacement = None,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.add_rule(
            UpdateRule(action, str(expression), value=value, source=source)
        )

    def replace(
        self,
        expression: str | JSONPath,
        value: Replacement,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.add_rule(
            UpdateRule("replace", str(expression), value=value, source=source)
        )

    def update(
        self,
        expression: str | JSONPath,
        value: Replacement,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.replace(expression, value, source=source)

    def add_replace(
        self,
        expression: str | JSONPath,
        value: Replacement,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.replace(expression, value, source=source)

    def delete(
        self,
        expression: str | JSONPath,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.add_rule(
            UpdateRule("delete", str(expression), source=source)
        )

    def add_delete(
        self,
        expression: str | JSONPath,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.delete(expression, source=source)

    def remove(
        self,
        expression: str | JSONPath,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.delete(expression, source=source)

    def set(
        self,
        expression: str | JSONPath,
        value: Replacement,
        *,
        source: str | None = None,
    ) -> UpdatePlan:
        return self.replace(expression, value, source=source)

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self) -> typing.Iterator[UpdateOperation]:
        return iter(self.operations)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(digest={self._digest!r}, "
            f"operations={self.operations!r})"
        )

    def resolve(self) -> UpdatePlan:
        self._resolve()
        return self

    def validate(self) -> UpdatePlan:
        self._resolve()
        if self._conflicts:
            raise UpdatePlanConflictError(self._conflicts, plan=self)
        return self

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "digest": self._digest,
            "operations": [
                operation.to_dict() for operation in self.operations
            ],
            "conflicts": [
                {
                    "kind": conflict.kind,
                    "path": conflict.path,
                    "expression": conflict.expression,
                    "related_path": conflict.related_path,
                    "related_expression": conflict.related_expression,
                }
                for conflict in self.conflicts
            ],
        }

    def describe(self) -> list[str]:
        self._resolve()
        lines = [str(operation) for operation in self._operations]
        lines.extend(conflict.describe() for conflict in self._conflicts)
        return lines

    def apply(
        self,
        document: typing.Any = None,
        *,
        in_place: bool = True,
    ) -> typing.Any:
        """
        Apply the plan once.

        The document's digest must still match the plan's original digest.
        A different object is accepted when its contents have the same digest.
        A callback failure discards the working copy, leaving both the document
        and plan intact.
        """

        self.validate()
        target = self._document if document is None else document
        _validate_data(target)
        if _digest(target) != self._digest:
            raise UpdatePlanStaleError(
                "the document changed after the update plan was created"
            )

        working = copy.deepcopy(target)
        updated_root = self._apply_operations(working)
        _validate_data(updated_root)

        if not in_place:
            return updated_root
        if updated_root is not target:
            self._commit(target, updated_root)
        return target

    def apply_to(
        self,
        document: typing.Any,
        *,
        in_place: bool = False,
    ) -> typing.Any:
        return self.apply(document, in_place=in_place)

    def commit(
        self,
        document: typing.Any = None,
        *,
        in_place: bool = True,
    ) -> typing.Any:
        return self.apply(document, in_place=in_place)

    execute = commit

    def _resolve(self) -> None:
        if self._resolved:
            return

        candidates: list[_Candidate] = []
        for rule in self._rules:
            try:
                matches = rule.path.find(self._snapshot)
            except Exception as exc:
                raise UpdatePlanError(
                    f"could not evaluate expression {rule.expression!r}"
                ) from exc
            for match in matches:
                candidates.append(self._make_candidate(rule, match))

        grouped: dict[
            tuple[int, str | int] | tuple[str], list[_Candidate]
        ] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.physical_key, []).append(candidate)

        operations: list[UpdateOperation] = []
        conflicts: list[UpdateConflict] = []
        for group in grouped.values():
            actions = {candidate.rule.action for candidate in group}
            if len(actions) > 1:
                conflicts.append(
                    UpdateConflict(
                        kind="replace-delete",
                        path=group[0].path,
                        expression=group[0].rule.expression,
                        expressions=tuple(
                            dict.fromkeys(
                                candidate.rule.expression
                                for candidate in group
                            )
                        ),
                    )
                )
                continue

            action = group[0].rule.action
            if action == "replace":
                replacements = [candidate.rule.value for candidate in group]
                if not all(
                    _same_replacement(replacements[0], replacement)
                    for replacement in replacements[1:]
                ):
                    conflicts.append(
                        UpdateConflict(
                            kind="replace-replace",
                            path=group[0].path,
                            expression=group[0].rule.expression,
                            expressions=tuple(
                                dict.fromkeys(
                                    candidate.rule.expression
                                    for candidate in group
                                )
                            ),
                        )
                    )
                    continue

            first = group[0]
            operations.append(
                UpdateOperation(
                    action=action,
                    path=first.path,
                    expression=first.rule.expression,
                    sources=tuple(
                        dict.fromkeys(
                            candidate.rule.expression for candidate in group
                        )
                    ),
                    value=first.rule.value,
                    key=first.key,
                    source=first.rule.source,
                    _address=first.address,
                    _target_value=first.value,
                    _ancestors=first.ancestors,
                )
            )

        for operation in operations:
            if operation.action == "delete" and not operation._address:
                conflicts.append(
                    UpdateConflict(
                        kind="root-delete",
                        path=operation.path,
                        expression=operation.expression,
                        expressions=operation.sources,
                    )
                )

        for index, left in enumerate(operations):
            for right in operations[index + 1 :]:
                relation = self._parent_child_relation(left, right)
                if relation is not None:
                    parent, child = relation
                    conflicts.append(
                        UpdateConflict(
                            kind="parent-child",
                            path=parent.path,
                            expression=parent.expression,
                            related_path=child.path,
                            related_expression=child.expression,
                            expressions=parent.sources,
                            related_expressions=child.sources,
                        )
                    )

        self._operations = tuple(operations)
        self._conflicts = tuple(conflicts)
        self._resolved = True

    def _make_candidate(
        self, rule: UpdateRule, datum: DatumInContext
    ) -> _Candidate:
        address, key, ancestors, parent_value = self._target_address(datum)
        if address:
            physical_key: tuple[int, str | int] | tuple[str] = (
                id(parent_value),
                key,
            )
        else:
            physical_key = ("root",)

        order = self._order
        self._order += 1
        return _Candidate(
            rule=rule,
            address=address,
            key=key,
            path=_canonical_path(address),
            physical_key=physical_key,
            order=order,
            value=datum.value,
            ancestors=ancestors,
        )

    def _target_address(
        self, datum: DatumInContext
    ) -> tuple[Address, str | int | None, tuple[int, ...], typing.Any]:
        address_parts: list[str | int] = []
        ancestors: list[int] = [id(self._snapshot)]
        current = datum
        key: str | int | None = None
        parent_value: typing.Any = None
        found_target = False

        while current.context is not None:
            parent = current.context
            path = current.path
            if isinstance(path, Index):
                if len(path.indices) != 1 or type(parent.value) is not list:
                    raise UnsupportedUpdateDataError(
                        "the expression selected a non-standard array position"
                    )
                raw_key = path.indices[0]
                if not -len(parent.value) <= raw_key < len(parent.value):
                    raise UnsupportedUpdateDataError(
                        f"array index {raw_key} is not valid in the original "
                        "document"
                    )
                selected_key: str | int = raw_key % len(parent.value)
            elif isinstance(path, Fields):
                if len(path.fields) != 1 or type(parent.value) is not dict:
                    raise UnsupportedUpdateDataError(
                        "the expression selected a non-standard object member"
                    )
                selected_key = path.fields[0]
                if selected_key not in parent.value:
                    raise UnsupportedUpdateDataError(
                        f"object member {selected_key!r} is not present "
                        "in the original document"
                    )
            else:
                raise UnsupportedUpdateDataError(
                    "the expression selected an unsupported path segment: "
                    f"{path}"
                )

            address_parts.append(selected_key)
            if not found_target:
                key = selected_key
                parent_value = parent.value
                found_target = True
            if isinstance(parent.value, (dict, list)):
                ancestors.append(id(parent.value))
            current = parent

        address = tuple(reversed(address_parts))
        return (
            address,
            key,
            tuple(dict.fromkeys(ancestors)),
            parent_value,
        )

    @staticmethod
    def _parent_child_relation(
        left: UpdateOperation, right: UpdateOperation
    ) -> tuple[UpdateOperation, UpdateOperation] | None:
        if id(left._target_value) in right._ancestors:
            return left, right
        if id(right._target_value) in left._ancestors:
            return right, left
        return None

    def _apply_operations(self, working: typing.Any) -> typing.Any:
        root_replacement: typing.Any = working
        root_was_replaced = False

        for operation in self._operations:
            if operation.action != "replace":
                continue
            if not operation._address:
                root_value = operation.value
                if callable(root_value):
                    root_value = root_value(working, None, None)
                    if root_value is None:
                        continue
                root_replacement = copy.deepcopy(root_value)
                root_was_replaced = True
                continue

            parent = _resolve_address(working, operation._address[:-1])
            key = operation.key
            current_value = parent[key]
            replacement = operation.value
            if callable(replacement):
                replacement = replacement(current_value, parent, key)
                if replacement is None:
                    continue
            parent[key] = copy.deepcopy(replacement)

        delete_batches: dict[
            int, tuple[typing.Any, list[str | int]]
        ] = {}
        for operation in self._operations:
            if operation.action != "delete":
                continue
            parent = _resolve_address(working, operation._address[:-1])
            key = operation.key
            batch = delete_batches.setdefault(id(parent), (parent, []))
            batch[1].append(key)

        for parent, keys in delete_batches.values():
            if isinstance(parent, list):
                positions = sorted({int(key) for key in keys}, reverse=True)
                for position in positions:
                    del parent[position]
            else:
                for key in keys:
                    parent.pop(key, None)

        return root_replacement if root_was_replaced else working

    @staticmethod
    def _commit(document: typing.Any, replacement: typing.Any) -> None:
        if isinstance(document, dict) and isinstance(replacement, dict):
            document.clear()
            document.update(replacement)
            return
        if isinstance(document, list) and isinstance(replacement, list):
            document[:] = replacement
            return
        raise UnsupportedUpdateDataError(
            "the replacement document has a different root container type"
        )


def _resolve_address(document: typing.Any, address: Address) -> typing.Any:
    current = document
    for part in address:
        if type(current) not in (dict, list):
            raise UpdatePlanError(
                "the document structure no longer matches the update plan"
            )
        current = current[part]
    return current


def build_update_plan(
    document: typing.Any,
    rules: Iterable[
        UpdateRule | Mapping[str, typing.Any] | tuple
    ] | None = None,
) -> UpdatePlan:
    """Build an :class:`UpdatePlan` for *document* and optional *rules*."""

    return UpdatePlan(document, rules)


def apply_update_plan(
    plan: UpdatePlan,
    document: typing.Any = None,
    *,
    in_place: bool = True,
) -> typing.Any:
    """Apply *plan* and return the resulting document."""

    return plan.apply(document, in_place=in_place)


update_plan = build_update_plan
make_update_plan = build_update_plan
create_update_plan = build_update_plan
plan_update = build_update_plan
UpdatePlanConflict = UpdatePlanConflictError
UpdatePlanStale = UpdatePlanStaleError
StaleUpdatePlanError = UpdatePlanStaleError
ConflictUpdatePlanError = UpdatePlanConflictError
UpdatePlanner = UpdatePlan
