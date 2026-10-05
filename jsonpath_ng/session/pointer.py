"""JSON Pointer (RFC 6901) helpers and document access counting.

Pointers are represented as tuples of reference tokens (integers for array
positions, strings for object members). The empty tuple is the document root.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional, Tuple

Pointer = Tuple[Any, ...]

_MISSING = object()


class PointerError(ValueError):
    """A malformed JSON Pointer."""


def parse_pointer(pointer: Any) -> Pointer:
    """Parse an RFC 6901 pointer string into a token tuple.

    Non-string input is normalized into a tuple, or wrapped as a single token.
    """
    if isinstance(pointer, tuple):
        return pointer
    if isinstance(pointer, list):
        return tuple(pointer)
    if pointer is None or pointer == "":
        return ()
    if not isinstance(pointer, str):
        raise PointerError(f"pointer must be a string, got {type(pointer)!r}")
    if not pointer.startswith("/"):
        raise PointerError(
            f"non-empty JSON pointer must start with '/': {pointer!r}"
        )
    tokens: List[Any] = []
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if token == "-":
            tokens.append("-")
        elif token.lstrip("-").isdigit() and not (
            len(token) > 1 and token[0] == "0"
        ):
            tokens.append(int(token))
        else:
            tokens.append(token)
    return tuple(tokens)


def format_pointer(tokens: Pointer) -> str:
    """Render a token tuple as an RFC 6901 string."""
    out = []
    for token in tokens:
        if token == "-":
            rendered = "-"
        elif isinstance(token, int):
            rendered = str(token)
        else:
            rendered = str(token).replace("~", "~0").replace("/", "~1")
        out.append("/" + rendered)
    return "".join(out)


def pointer_prefix(tokens: Pointer, length: int) -> Pointer:
    return tuple(tokens[:length])


def is_prefix(prefix: Pointer, path: Pointer) -> bool:
    return path[: len(prefix)] == prefix


def _index_ok(container, index: int) -> bool:
    return isinstance(container, list) and 0 <= index < len(container)


def resolve(doc: Any, tokens: Pointer, default: Any = _MISSING) -> Any:
    """Resolve *tokens* in *doc*; return *default* when the target is missing."""
    current = doc
    for depth, token in enumerate(tokens):
        if isinstance(token, int):
            if not isinstance(current, list) or not _index_ok(current, token):
                if default is not _MISSING:
                    return default
                raise PointerError(
                    f"index {token} out of range at {format_pointer(tokens[:depth])}"
                )
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                if default is not _MISSING:
                    return default
                raise PointerError(
                    f"member {token!r} missing at {format_pointer(tokens[:depth])}"
                )
            current = current[token]
    return current


def exists(doc: Any, tokens: Pointer) -> bool:
    return resolve(doc, tokens, _MISSING) is not _MISSING


def parent(doc: Any, tokens: Pointer) -> Optional[Any]:
    if not tokens:
        return None
    return resolve(doc, tokens[:-1])


def snap(doc: Any, tokens: Pointer) -> Any:
    """Deep copy of the node at *tokens*, or ``None`` if it does not exist."""
    value = resolve(doc, tokens, None)
    return deepcopy(value)


class AccessCounter:
    """Counts document nodes *visited* during a query evaluation.

    A node is counted the first time it is entered (root included). Wrapping a
    node for evaluation is O(1): children are wrapped lazily on access, so a
    query that only looks at a small subtree does not touch the rest.
    """

    def __init__(self) -> None:
        self.visited = 0
        self._seen: set = set()

    def reset(self) -> None:
        self.visited = 0
        self._seen.clear()

    def record(self, node_id: int) -> None:
        if node_id not in self._seen:
            self._seen.add(node_id)
            self.visited += 1

    def wrap(self, node: Any) -> Any:
        if isinstance(node, _CountedNode):
            return node
        if isinstance(node, dict):
            return _CountedDict(node, self)
        if isinstance(node, list):
            return _CountedList(node, self)
        return node

    def unwrap(self, value: Any) -> Any:
        if isinstance(value, _CountedNode):
            return value.__raw__
        return value


class _CountedNode:
    __slots__ = ("__weakref__",)


class _CountedDict(dict, _CountedNode):
    def __init__(self, raw: dict, counter: AccessCounter):
        counter.record(id(raw))
        object.__setattr__(self, "__raw__", raw)
        object.__setattr__(self, "__counter__", counter)

    def __getitem__(self, key):
        raw = object.__getattribute__(self, "__raw__")
        counter = object.__getattribute__(self, "__counter__")
        return counter.wrap(raw[key])

    def __contains__(self, key):
        raw = object.__getattribute__(self, "__raw__")
        return key in raw

    def get(self, key, default=None):
        raw = object.__getattribute__(self, "__raw__")
        counter = object.__getattribute__(self, "__counter__")
        if key in raw:
            return counter.wrap(raw[key])
        return default

    def keys(self):
        return object.__getattribute__(self, "__raw__").keys()

    def values(self):
        raw = object.__getattribute__(self, "__raw__")
        counter = object.__getattribute__(self, "__counter__")
        return (counter.wrap(v) for v in raw.values())

    def items(self):
        raw = object.__getattribute__(self, "__raw__")
        counter = object.__getattribute__(self, "__counter__")
        return ((k, counter.wrap(v)) for k, v in raw.items())

    def __iter__(self):
        return iter(object.__getattribute__(self, "__raw__").keys())

    def __len__(self):
        return len(object.__getattribute__(self, "__raw__"))

    def __eq__(self, other):
        if isinstance(other, _CountedDict):
            other = other.__raw__
        return object.__getattribute__(self, "__raw__") == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(object.__getattribute__(self, "__raw__"))

    def __bool__(self):
        return bool(object.__getattribute__(self, "__raw__"))

    def __hash__(self):
        return id(self)


class _CountedList(list, _CountedNode):
    def __init__(self, raw: list, counter: AccessCounter):
        counter.record(id(raw))
        object.__setattr__(self, "__raw__", raw)
        object.__setattr__(self, "__counter__", counter)

    def __getitem__(self, index):
        raw = object.__getattribute__(self, "__raw__")
        counter = object.__getattribute__(self, "__counter__")
        if isinstance(index, slice):
            return [counter.wrap(v) for v in raw[index]]
        return counter.wrap(raw[index])

    def __iter__(self):
        raw = object.__getattribute__(self, "__raw__")
        counter = object.__getattribute__(self, "__counter__")
        for value in raw:
            yield counter.wrap(value)

    def __len__(self):
        return len(object.__getattribute__(self, "__raw__"))

    def __contains__(self, item):
        return object.__getattribute__(self, "__raw__").__contains__(item)

    def __eq__(self, other):
        if isinstance(other, _CountedList):
            other = other.__raw__
        return object.__getattribute__(self, "__raw__") == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(object.__getattribute__(self, "__raw__"))

    def __bool__(self):
        return bool(object.__getattribute__(self, "__raw__"))

    def __hash__(self):
        return id(self)
