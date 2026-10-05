"""Minimal RFC 6902 (JSON Patch) engine used by query sessions.

The engine mutates the document in place and records a *semantic event* for
every change. Events are the input of the incremental matcher:

* ``("value", path, old, new)``
    an object member (or the root, or an array element via "replace") changed.
    ``old is None`` means the member was added; ``new is None`` means removed.
* ``("insert", list_path, index, value, move_id)``
    an element was inserted into an array.
* ``("remove", list_path, index, old, move_id)``
    an element was removed from an array.
* ``("move_link", from_path, to_path)``
    a non-array (or cross-container) "move" for pointer pairing.

If any operation fails, every earlier mutation is rolled back and
:class:`PatchBatchError` is raised; no event is published.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, Tuple

from .pointer import Pointer, parse_pointer, resolve

Event = Tuple[Any, ...]


class PatchBatchError(ValueError):
    """A patch batch could not be applied.

    The document is left unchanged; no partial result is observable.
    """

    def __init__(self, index: int, op: Any, reason: str):
        self.index = index
        self.op = op
        self.reason = reason
        super().__init__(
            f"patch operation {index} ({op!r}) failed: {reason}"
        )


def json_equal(left: Any, right: Any) -> bool:
    """JSON type-aware equality (``True`` is not equal to ``1``)."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if left is None or right is None:
        return left is None and right is None
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            json_equal(left[k], right[k]) for k in left
        )
    return left == right


def _is_int_token(token: Any) -> bool:
    return isinstance(token, int) and not isinstance(token, bool)


class _BatchApplier:
    def __init__(self, doc: Any, patches: List[dict]):
        self.doc = doc
        self.patches = patches
        self.events: List[Event] = []
        self._undo: List[tuple] = []
        self._move_seq = 0

    # -- mutation primitives -------------------------------------------------

    def _set_member(self, path: Pointer, value: Any):
        parent = resolve(self.doc, path[:-1])
        token = path[-1]
        if isinstance(parent, list):
            if not _is_int_token(token) or not (0 <= token < len(parent)):
                raise PointerIndexError()
            old = parent[token]
            parent[token] = value
            self._undo.append(("index-set", path[:-1], token, old))
            self.events.append(("value", path, deepcopy(old), value))
        else:
            existed = token in parent
            old = deepcopy(parent[token]) if existed else None
            parent[token] = value
            self._undo.append(("member", path, existed, old))
            self.events.append(("value", path, old, value))

    def _add(self, raw_path: Any, value: Any, move_id: Optional[str] = None):
        path = parse_pointer(raw_path)
        if not path:
            self._replace_root(value, deepcopy(self.doc), record=True)
            return
        parent = resolve(self.doc, path[:-1])
        token = path[-1]
        if isinstance(parent, list):
            index = self._array_insert_index(token, len(parent))
            parent.insert(index, value)
            self._undo.append(("list-pop", path[:-1], index))
            self.events.append(
                ("insert", path[:-1], index, value, move_id)
            )
        elif isinstance(parent, dict):
            existed = token in parent
            old = deepcopy(parent[token]) if existed else None
            parent[token] = value
            self._undo.append(("member", path, existed, old))
            if existed:
                self.events.append(("value", path, old, value))
            else:
                self.events.append(("value", path, None, value))
        else:
            raise PointerTargetError()

    def _remove(self, raw_path: Any, move_id: Optional[str] = None) -> Any:
        path = parse_pointer(raw_path)
        if not path:
            raise PointerTargetError("the document root cannot be removed")
        parent = resolve(self.doc, path[:-1])
        token = path[-1]
        if isinstance(parent, list):
            index = self._array_existing_index(token, len(parent))
            old = parent.pop(index)
            self._undo.append(("list-insert", path[:-1], index, old))
            self.events.append(("remove", path[:-1], index, deepcopy(old), move_id))
            return old
        elif isinstance(parent, dict):
            if token not in parent:
                raise PointerTargetError()
            old = parent.pop(token)
            self._undo.append(("member", path, True, deepcopy(old)))
            self.events.append(("value", path, deepcopy(old), None))
            return old
        raise PointerTargetError()

    def _replace(self, raw_path: Any, value: Any):
        path = parse_pointer(raw_path)
        if not path:
            old = deepcopy(self.doc)
            self._replace_root(value, old, record=True)
            return
        parent = resolve(self.doc, path[:-1])
        token = path[-1]
        if isinstance(parent, list):
            if not _is_int_token(token) or not (0 <= token < len(parent)):
                raise PointerTargetError()
            old = parent[token]
            parent[token] = value
            self._undo.append(("index-set", path[:-1], token, old))
            self.events.append(("value", path, deepcopy(old), value))
        elif isinstance(parent, dict):
            if token not in parent:
                raise PointerTargetError()
            old = parent[token]
            parent[token] = value
            self._undo.append(("member", path, True, deepcopy(old)))
            self.events.append(("value", path, deepcopy(old), value))
        else:
            raise PointerTargetError()

    def _replace_root(self, value: Any, old: Any, record: bool):
        # Mutate in place so callers holding a reference keep seeing the doc.
        doc = self.doc
        doc.clear()
        if isinstance(value, dict):
            doc.update(value)
        elif isinstance(value, list):
            doc.extend(value)
        else:
            raise PointerTargetError("root must be an object or array")
        if record:
            self._undo.append(("root", old))
            self.events.append(("value", (), old, value))

    @staticmethod
    def _array_insert_index(token: Any, length: int) -> int:
        if token == "-":
            return length
        if _is_int_token(token) and 0 <= token <= length:
            return token
        raise PointerIndexError()

    @staticmethod
    def _array_existing_index(token: Any, length: int) -> int:
        if _is_int_token(token) and 0 <= token < length:
            return token
        raise PointerIndexError()

    # -- operations ----------------------------------------------------------

    def _test(self, op: dict):
        if "value" not in op:
            raise MalformedPatch("'test' requires a 'value' member")
        actual = resolve(self.doc, parse_pointer(op.get("path", "")))
        if not json_equal(actual, op["value"]):
            raise TestFailed()

    def _move(self, op: dict):
        frm = parse_pointer(op.get("from", _MISSING_FROM))
        to = parse_pointer(op.get("path", _MISSING_FROM))
        if frm is None or to is None:
            raise MalformedPatch("'move' requires 'from' and 'path'")
        if frm == to[: len(frm)] and frm != to:
            raise PointerTargetError("cannot move a value into itself")
        self._move_seq += 1
        move_id = f"move-{self._move_seq}"
        value = self._take(frm, move_id)
        self._place(to, value, move_id, frm)

    def _copy(self, op: dict):
        frm = parse_pointer(op.get("from", _MISSING_FROM))
        to = parse_pointer(op.get("path", _MISSING_FROM))
        if frm is None or to is None:
            raise MalformedPatch("'copy' requires 'from' and 'path'")
        value = deepcopy(resolve(self.doc, frm))
        self._place(to, value, None, frm)

    def _take(self, frm: Pointer, move_id: str) -> Any:
        if not frm:
            raise PointerTargetError("the document root cannot be moved")
        parent = resolve(self.doc, frm[:-1])
        token = frm[-1]
        if isinstance(parent, list):
            index = self._array_existing_index(token, len(parent))
            old = parent.pop(index)
            self._undo.append(("list-insert", frm[:-1], index, old))
            self.events.append(("remove", frm[:-1], index, deepcopy(old), move_id))
            return old
        if isinstance(parent, dict):
            if token not in parent:
                raise PointerTargetError()
            old = parent.pop(token)
            self._undo.append(("member", frm, True, deepcopy(old)))
            self.events.append(("value", frm, deepcopy(old), None))
            return old
        raise PointerTargetError()

    def _place(self, to: Pointer, value: Any, move_id: Optional[str], frm: Pointer):
        if not to:
            raise PointerTargetError("cannot place a value at the root")
        parent = resolve(self.doc, to[:-1])
        token = to[-1]
        if isinstance(parent, list):
            index = self._array_insert_index(token, len(parent))
            parent.insert(index, value)
            self._undo.append(("list-pop", to[:-1], index))
            self.events.append(("insert", to[:-1], index, value, move_id))
        elif isinstance(parent, dict):
            existed = token in parent
            old = deepcopy(parent[token]) if existed else None
            parent[token] = value
            self._undo.append(("member", to, existed, old))
            if existed:
                self.events.append(("value", to, old, value))
            else:
                self.events.append(("value", to, None, value))
            if move_id is not None:
                self.events.append(("move_link", frm, to))
        else:
            raise PointerTargetError()

    # -- driver --------------------------------------------------------------

    def run(self) -> List[Event]:
        if not isinstance(self.patches, list):
            raise PatchBatchError(-1, self.patches, "patch must be a list")
        for index, op in enumerate(self.patches):
            try:
                self._apply_one(op)
            except _PatchFailure as exc:
                self._rollback()
                raise PatchBatchError(index, op, str(exc)) from exc
            except (PointerError, KeyError, TypeError, ValueError) as exc:
                self._rollback()
                raise PatchBatchError(index, op, str(exc)) from exc
        return self.events

    def _apply_one(self, op: Any):
        if not isinstance(op, dict):
            raise MalformedPatch("operation must be an object")
        name = op.get("op")
        if not isinstance(name, str):
            raise MalformedPatch("missing or invalid 'op'")
        if "path" not in op:
            raise MalformedPatch("missing 'path'")
        parse_pointer(op["path"])  # validate syntax up-front
        if name == "test":
            self._test(op)
        elif name == "remove":
            self._remove(op["path"])
        elif name == "add":
            if "value" not in op:
                raise MalformedPatch("'add' requires a 'value' member")
            self._add(op["path"], op["value"])
        elif name == "replace":
            if "value" not in op:
                raise MalformedPatch("'replace' requires a 'value' member")
            self._replace(op["path"], op["value"])
        elif name == "move":
            self._move(op)
        elif name == "copy":
            self._copy(op)
        else:
            raise MalformedPatch(f"unsupported op {name!r}")

    def _rollback(self):
        for action in reversed(self._undo):
            kind = action[0]
            if kind == "root":
                old = action[1]
                doc = self.doc
                doc.clear()
                if isinstance(old, dict):
                    doc.update(deepcopy(old))
                else:
                    doc.extend(deepcopy(old))
            elif kind == "member":
                _, path, existed, old = action
                container = resolve(self.doc, path[:-1])
                token = path[-1]
                if existed:
                    container[token] = deepcopy(old)
                else:
                    container.pop(token, None)
            elif kind == "index-set":
                _, lp, index, old = action
                resolve(self.doc, lp)[index] = old
            elif kind == "list-pop":
                _, lp, index = action
                resolve(self.doc, lp).pop(index)
            elif kind == "list-insert":
                _, lp, index, old = action
                resolve(self.doc, lp).insert(index, old)
        self.events = []


class _PatchFailure(Exception):
    pass


class MalformedPatch(_PatchFailure):
    pass


class PointerTargetError(_PatchFailure):
    pass


class PointerIndexError(_PatchFailure):
    def __str__(self):
        return "array index out of range (or '-' used on an existing element)"


class TestFailed(_PatchFailure):
    pass


class PointerError(_PatchFailure):
    pass


_MISSING_FROM = object()



def apply_batch(doc: Any, patches: List[dict]) -> List[Event]:
    """Apply *patches* to *doc* in place; roll everything back on failure."""
    return _BatchApplier(doc, patches).run()
