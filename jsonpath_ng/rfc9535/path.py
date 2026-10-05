from __future__ import annotations

from typing import Any, Iterable, Optional


class _Nothing:
    def __bool__(self):
        return False

    def __repr__(self):
        return "Nothing"


Nothing = _Nothing()


def is_nothing(value: Any) -> bool:
    return value is Nothing


class NormalizedPath:
    def __init__(self, segments: Optional[tuple[Any, ...]] = None):
        self.segments = () if segments is None else segments

    def child(self, segment: Any) -> "NormalizedPath":
        return NormalizedPath(self.segments + (segment,))

    def __str__(self) -> str:
        result = "$"
        for segment in self.segments:
            if isinstance(segment, int):
                result += f"[{segment}]"
            else:
                result += f"[{_quote_name(segment)}]"
        return result

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NormalizedPath) and self.segments == other.segments

    def __hash__(self) -> int:
        return hash(self.segments)

    def __repr__(self) -> str:
        return f"NormalizedPath({self!s})"


class Match:
    def __init__(
        self,
        value: Any,
        path: NormalizedPath,
        root_value: Any,
        context: Optional["Match"] = None,
    ):
        self.value = value
        self.path = path
        self.full_path = path
        self.root_value = root_value
        self.context = context

    def in_context(self, value: Any, segment: Any, root_value: Any) -> "Match":
        return Match(value, self.path.child(segment), root_value, self)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Match)
            and self.value == other.value
            and self.path == other.path
        )

    def __repr__(self) -> str:
        return f"Match(value={self.value!r}, path={self.path!s})"


class NodeList:
    def __init__(self, nodes: Iterable[Match] = ()):
        self.nodes = tuple(nodes)

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def value(self) -> Any:
        if len(self.nodes) != 1:
            return Nothing
        return self.nodes[0].value

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NodeList) and self.nodes == other.nodes

    def __repr__(self) -> str:
        return f"NodeList({self.nodes!r})"


def _quote_name(value: str) -> str:
    replacements = {
        "\\": "\\\\",
        "'": "\\'",
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    escaped = "".join(
        replacements.get(char)
        or (f"\\u{ord(char):04x}" if ord(char) < 0x20 else char)
        for char in value
    )
    return f"'{escaped}'"
