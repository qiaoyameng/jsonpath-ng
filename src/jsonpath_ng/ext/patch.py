"""Versioned JSON Patch (RFC 6902 style) application.

A patch is a list of operations.  Every operation is validated against the
document before anything is mutated, so a malformed patch (bad base version or
illegal path) leaves the document untouched.  Successfully applied patches are
reported as a list of :class:`OperationEvent` objects describing the concrete
nodes that changed; the incremental subscription engine consumes those events
instead of re-running its queries.
"""

from __future__ import annotations

import copy
import typing


class PatchError(ValueError):
    """Raised when a patch cannot be applied.

    The document is guaranteed to be unchanged when this is raised.
    """


class VersionMismatchError(PatchError):
    """Raised when a patch's base version does not match the document."""


Segment: typing.TypeAlias = "str | int"
ConcretePath: typing.TypeAlias = tuple[Segment, ...]


def _decode_pointer(token: str) -> str:
    # RFC 6901: '~1' must be unescaped before '~0'.
    return token.replace("~1", "/").replace("~0", "~")


def parse_json_pointer(pointer: str) -> ConcretePath:
    """Parse an RFC 6901 JSON Pointer into a concrete materialized path.

    Integer-looking tokens stay strings here: field names such as ``"1"``
    are legal object keys, and the document decides whether a segment is an
    array index.  The empty pointer resolves to the document root.
    """

    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        msg = f"Invalid JSON Pointer {pointer!r}: pointers must start with '/'"
        raise PatchError(msg)
    return tuple(_decode_TOKEN(token) for token in pointer.split("/")[1:])


def _decode_TOKEN(token: str) -> Segment:
    decoded = _decode_pointer(token)
    if decoded.lstrip("-").isdigit():
        return int(decoded)
    return decoded


class OperationEvent:
    """A concrete, already-applied structural change.

    Attributes:
        op: One of ``"add"``, ``"remove"`` or ``"replace"``.
        path: The concrete path of the affected node.  For array inserts and
            removals this is the index where the element was inserted or
            removed; an empty tuple means the whole document was replaced.
        old_value: The value before the operation (``_ABSENT`` when nothing
            was there before, as with an ``add``).
        new_value: The value after the operation (``_ABSENT`` for ``remove``).
        shifted_from: For array inserts/removals, the concrete paths of the
            elements that shifted, mapped to their new paths.
    """

    __slots__ = ("op", "path", "old_value", "new_value", "shifted_from", "shifted_to")

    def __init__(
        self,
        op: str,
        path: ConcretePath,
        old_value: typing.Any,
        new_value: typing.Any,
        shifted_from: list[ConcretePath] | None = None,
        shifted_to: list[ConcretePath] | None = None,
    ) -> None:
        self.op = op
        self.path = path
        self.old_value = old_value
        self.new_value = new_value
        self.shifted_from = shifted_from or []
        self.shifted_to = shifted_to or []

    def __repr__(self) -> str:
        return (
            f"OperationEvent(op={self.op!r}, path={self.path!r}, "
            f"shifted={len(self.shifted_from)})"
        )


_ABSENT = object()

#: Sentinel marking "no value existed" (the ``old_value`` of an ``add`` and
#: the ``new_value`` of a ``remove`` event).
ABSENT = _ABSENT


class VersionedDocument:
    """Holds a JSON document and a monotonically increasing version."""

    def __init__(self, document: typing.Any) -> None:
        self._document = copy.deepcopy(document)
        self._version = 0

    @property
    def document(self) -> typing.Any:
        return self._document

    @property
    def version(self) -> int:
        return self._version

    def snapshot(self) -> typing.Any:
        """Return a deep copy of the current document."""

        return copy.deepcopy(self._document)

    def apply_patch(
        self,
        patch: typing.Sequence[dict[str, typing.Any]],
        base_version: int | None = None,
    ) -> list[OperationEvent]:
        """Apply an RFC 6902-style patch atomically.

        ``base_version`` optionally pins the patch to a document version; a
        mismatch raises :class:`VersionMismatchError` without mutating the
        document.  Any illegal path or unknown operation raises
        :class:`PatchError` without mutating the document.
        """

        if base_version is not None and base_version != self._version:
            msg = (
                f"Patch base version {base_version} does not match "
                f"document version {self._version}"
            )
            raise VersionMismatchError(msg)

        operations = [self._normalize(operation) for operation in patch]

        # Validate every operation on a scratch copy first.  Nothing the
        # caller can see is touched until all operations are known to be legal.
        scratch = copy.deepcopy(self._document)
        # The scratch copy is mutated as operations are validated so that
        # index validation sees the effects of earlier operations in the
        # same patch (an insert shifts later indices, and so on).
        for operation in operations:
            self._validate(scratch, operation)
            _apply_operation(scratch, operation)

        events: list[OperationEvent] = []
        for operation in operations:
            if not operation["path"]:
                event = _apply_operation(self._document, operation)
                self._document = (
                    None if event.new_value is _ABSENT else event.new_value
                )
                events.append(event)
            else:
                events.append(_apply_operation(self._document, operation))
        self._version += 1
        return events

    @staticmethod
    def _normalize(operation: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if not isinstance(operation, dict) or "op" not in operation:
            msg = (
                "Patch operation must be an object with an 'op' member: "
                f"{operation!r}"
            )
            raise PatchError(msg)
        name = operation["op"]
        if name not in {"add", "remove", "replace"}:
            msg = f"Unsupported patch operation {name!r}"
            raise PatchError(msg)
        if "path" not in operation:
            msg = f"Patch operation {name!r} is missing its 'path'"
            raise PatchError(msg)
        normalized = {
            "op": name,
            "path": parse_json_pointer(operation["path"]),
        }
        if name in {"add", "replace"}:
            if "value" not in operation:
                msg = f"Patch operation {name!r} is missing its 'value'"
                raise PatchError(msg)
            normalized["value"] = copy.deepcopy(operation["value"])
        return normalized

    def _validate(self, document: typing.Any, operation: dict[str, typing.Any]) -> None:
        name = operation["op"]
        path = operation["path"]
        if name == "add":
            self._validate_add(document, path)
        elif name == "remove":
            self._validate_existing(document, path)
        else:  # replace
            self._validate_existing(document, path)

    def _validate_existing(
        self, document: typing.Any, path: ConcretePath
    ) -> None:
        if not path:
            return
        parent, key = _parent_and_key(document, path)
        if isinstance(parent, list):
            if not isinstance(key, int) or not -len(parent) <= key < len(parent):
                msg = f"Path {path!r} does not exist in the document"
                raise PatchError(msg)
        elif isinstance(parent, dict):
            if key not in parent:
                msg = f"Path {path!r} does not exist in the document"
                raise PatchError(msg)
        else:
            msg = f"Path {path!r} traverses a scalar value"
            raise PatchError(msg)

    def _validate_add(
        self, document: typing.Any, path: ConcretePath
    ) -> None:
        if not path:
            return
        parent, key = _parent_and_key(document, path)
        if isinstance(parent, list):
            if key == "-" or (
                isinstance(key, int) and -len(parent) <= key <= len(parent)
            ):
                return
            msg = f"Array index {key!r} is out of bounds for path {path!r}"
            raise PatchError(msg)
        if isinstance(parent, dict):
            return
        msg = f"Path {path!r} traverses a scalar value"
        raise PatchError(msg)

def _parent_and_key(
    document: typing.Any, path: ConcretePath
) -> tuple[typing.Any, Segment]:
    parent: typing.Any = document
    for segment in path[:-1]:
        if isinstance(parent, list):
            if not isinstance(segment, int) or not -len(parent) <= segment < len(
                parent
            ):
                msg = f"Path {path!r} does not exist in the document"
                raise PatchError(msg)
            parent = parent[segment]
        elif isinstance(parent, dict):
            if segment not in parent:
                msg = f"Path {path!r} does not exist in the document"
                raise PatchError(msg)
            parent = parent[segment]
        else:
            msg = f"Path {path!r} traverses a scalar value"
            raise PatchError(msg)
    return parent, path[-1]


def _apply_operation(
    document: typing.Any, operation: dict[str, typing.Any]
) -> OperationEvent:
    name = operation["op"]
    path = operation["path"]
    if not path:
        old = document if name != "add" else _ABSENT
        new = operation["value"] if name != "remove" else _ABSENT
        return OperationEvent(name, (), old, new)

    parent, key = _parent_and_key(document, path)
    if name == "add":
        return _apply_add(parent, key, path, operation["value"])
    if name == "remove":
        return _apply_remove(parent, key, path)
    if isinstance(parent, list) and isinstance(key, int) and key < 0:
        key = key + len(parent)
    old_value = parent[key]
    parent[key] = copy.deepcopy(operation["value"])
    return OperationEvent(
        "replace", path, old_value, copy.deepcopy(operation["value"])
    )


def _apply_add(
    parent: typing.Any,
    key: Segment,
    path: ConcretePath,
    value: typing.Any,
) -> OperationEvent:
        if isinstance(parent, list):
            index = len(parent) if key == "-" else key
            if index < 0:
                index += len(parent) + 1
            parent.insert(index, value)
            shifted_from = [path[:-1] + (i,) for i in range(index, len(parent) - 1)]
            shifted_to = [path[:-1] + (i + 1,) for i in range(index, len(parent) - 1)]
            return OperationEvent(
                "add", path, _ABSENT, copy.deepcopy(value), shifted_from, shifted_to
            )
        existed = key in parent
        old_value = parent[key] if existed else _ABSENT
        parent[key] = value
        op = "replace" if existed else "add"
        return OperationEvent(op, path, old_value, copy.deepcopy(value))

def _apply_remove(
    parent: typing.Any,
    key: Segment,
    path: ConcretePath,
) -> OperationEvent:
        if isinstance(parent, list):
            index = key if key >= 0 else key + len(parent)
            old_value = parent[index]
            shifted_from = [path[:-1] + (i,) for i in range(index + 1, len(parent))]
            shifted_to = [path[:-1] + (i - 1,) for i in range(index + 1, len(parent))]
            del parent[index]
            return OperationEvent(
                "remove", path, old_value, _ABSENT, shifted_from, shifted_to
            )
        old_value = parent.pop(key)
        return OperationEvent("remove", path, old_value, _ABSENT)
