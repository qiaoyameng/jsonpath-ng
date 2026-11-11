"""Tests for the lazy ``JSONPath.iter_find()`` method.

The contract checked here is that ``iter_find()``:

* streams matches lazily, so closing an iterator early stops the traversal
  and releases every datum that has not been delivered;
* yields the same ``DatumInContext`` objects, in the same order and with the
  same duplicates, as ``find()`` returns as a list;
* keeps delivered objects' positions (``full_path``) stable after closing;
* propagates filter exceptions identically to ``find()``.
"""

import pytest

import jsonpath_ng
import jsonpath_ng.ext
from jsonpath_ng import Child
from jsonpath_ng import DatumInContext
from jsonpath_ng import Fields
from jsonpath_ng import JSONPath
from jsonpath_ng.ext.filter import Filter


class RecordingMapping(dict):
    """A dict that records every key read through ``get`` or ``[]``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed.append(key)
        return super().get(key, default)


# A short tree with an unvisited tail: if traversal stops after two matches,
# neither the root's later members nor the deep tail may be touched.
def recording_tree():
    tail = RecordingMapping({"v": 3})
    tree = RecordingMapping(
        {
            "a": [
                RecordingMapping({"v": 1}),
                RecordingMapping({"v": 2}),
            ],
            "b": RecordingMapping({"v": 99, "tail": tail}),
            "c": "never visited",
        }
    )
    return tree, tail


def test_find_still_returns_list():
    result = jsonpath_ng.parse("a[*].v").find({"a": [{"v": 1}, {"v": 2}]})
    assert isinstance(result, list)
    assert [datum.value for datum in result] == [1, 2]


def test_iter_find_returns_iterator():
    iterator = jsonpath_ng.parse("a").iter_find({"a": 1})
    assert iter(iterator) is iterator


def test_early_close_stops_traversal_through_children():
    tree, tail = recording_tree()
    iterator = jsonpath_ng.ext.parse("a[*].v").iter_find(tree)

    first = next(iterator)
    second = next(iterator)
    assert [first.value, second.value] == [1, 2]
    iterator.close()

    # Only the "a" branch was needed; the later root members remain untouched.
    assert tree.accessed == ["a"]
    assert tail.accessed == []


def test_early_close_stops_descendant_walk():
    tree, tail = recording_tree()
    iterator = jsonpath_ng.ext.parse("$..v").iter_find(tree)

    first = next(iterator)
    second = next(iterator)
    assert [first.value, second.value] == [1, 2]
    iterator.close()

    # The deep tail lives past the two delivered matches and must stay unread.
    assert tail.accessed == []


def test_delivered_datum_keeps_position_after_close():
    tree, _ = recording_tree()
    iterator = jsonpath_ng.ext.parse("a[*].v").iter_find(tree)

    delivered = next(iterator)
    delivered_full_path = str(delivered.full_path)
    iterator.close()

    assert delivered.value == 1
    assert delivered_full_path == "a.[0].v"
    assert str(delivered.full_path) == delivered_full_path


def test_closed_iterator_is_exhausted():
    iterator = jsonpath_ng.parse("a[*]").iter_find({"a": [1, 2, 3]})
    next(iterator)
    iterator.close()
    with pytest.raises(StopIteration):
        next(iterator)


# Expressions covering fields, indices, slices, wildcards, descendants,
# filters, and combinations of them.
PARSERS = pytest.mark.parametrize(
    "parse",
    (jsonpath_ng.parse, jsonpath_ng.ext.parse),
    ids=lambda function: function.__module__,
)

PARITY_CASES = (
    "a",
    "a.b",
    "a,b",
    "a[0]",
    "a[1:3]",
    "a[*].v",
    "a[0:2].v",
    "$..v",
    "a..v",
    "a[*]..v",
    "$..[*].v",
)

# Filters are only supported by the extended parser.
EXT_PARITY_CASES = (
    "$[?@.v > 1].v",
    "a[?@.v > 1][0].v",
)


def _parity_data():
    return {
        "a": [
            {"v": 1, "w": "one"},
            {"v": 2, "w": "two"},
            {"v": 1, "w": "one-duplicate"},
        ],
        "b": {"v": 4},
    }


@PARSERS
@pytest.mark.parametrize("expression", PARITY_CASES)
def test_full_iteration_matches_find(parse, expression):
    data = _parity_data()
    path = parse(expression)

    find_results = path.find(data)
    iter_results = list(path.iter_find(data))

    assert [datum.value for datum in iter_results] == [
        datum.value for datum in find_results
    ]
    assert [str(datum.full_path) for datum in iter_results] == [
        str(datum.full_path) for datum in find_results
    ]


@pytest.mark.parametrize("expression", EXT_PARITY_CASES)
def test_full_iteration_matches_find_ext(expression):
    data = _parity_data()
    path = jsonpath_ng.ext.parse(expression)

    find_results = path.find(data)
    iter_results = list(path.iter_find(data))

    assert [datum.value for datum in iter_results] == [
        datum.value for datum in find_results
    ]
    assert [str(datum.full_path) for datum in iter_results] == [
        str(datum.full_path) for datum in find_results
    ]


def test_iter_find_preserves_duplicate_hits():
    # A descendant query reaches the same leaves through multiple routes,
    # and iter_find must keep every duplicate that find keeps.
    data = {"a": {"x": [{"v": 1}, {"v": 2}]}, "b": {"x": [{"v": 1}]}}
    path = jsonpath_ng.ext.parse("$..x[*].v")

    find_values = [datum.value for datum in path.find(data)]
    iter_values = [datum.value for datum in path.iter_find(data)]

    assert iter_values == find_values
    assert len(iter_values) == len(find_values)


def test_datum_objects_are_distinct_per_hit():
    data = {"a": [{"v": 1}, {"v": 1}]}
    path = jsonpath_ng.parse("a[*].v")
    matches = list(path.iter_find(data))

    assert len(matches) == 2
    assert matches[0] is not matches[1]
    assert all(isinstance(datum, DatumInContext) for datum in matches)




@PARSERS
@pytest.mark.parametrize(
    "expression",
    (
        "missing",
        "a.missing.deeply.nested",
        "a[99].v",
        "$..nowhere",
    ),
)
def test_empty_results_match(parse, expression):
    data = _parity_data()
    path = parse(expression)

    assert path.find(data) == []
    assert list(path.iter_find(data)) == []


def test_empty_filter_results_match():
    data = _parity_data()
    path = jsonpath_ng.ext.parse("a[?@.v > 1000]")

    assert path.find(data) == []
    assert list(path.iter_find(data)) == []


class _RaisingPath(JSONPath):
    """A path whose evaluation always raises."""

    def find(self, data):
        raise RuntimeError("boom from filter expression")


def test_filter_exception_is_identical_to_find():
    path = Child(Fields("items"), Filter([_RaisingPath()]))
    data = {"items": [{"n": 1}, {"n": 2}]}

    with pytest.raises(RuntimeError, match="boom from filter expression"):
        path.find(data)

    iterator = path.iter_find(data)
    # Creating the iterator must not evaluate anything.
    with pytest.raises(RuntimeError, match="boom from filter expression"):
        next(iterator)


def test_filter_streams_candidates_lazily():
    first = RecordingMapping({"ok": True})
    second = RecordingMapping({"ok": False})
    third = RecordingMapping({"ok": True})
    data = {"p1": first, "p2": second, "p3": third}

    iterator = jsonpath_ng.ext.parse("$[?@.ok]").iter_find(data)
    assert next(iterator).value is first
    iterator.close()

    # The third candidate is never evaluated once the iterator is closed.
    assert third.accessed == []
    assert first.accessed == ["ok"]


def test_sortedthis_blocks_then_streams_sorted_values():
    data = {"items": [3, 1, 2]}
    path = jsonpath_ng.ext.parse("items.`sorted`[*]")

    assert [datum.value for datum in path.iter_find(data)] == [
        datum.value for datum in path.find(data)
    ] == [1, 2, 3]


def test_default_iter_find_delegates_to_find():
    # A node that only implements find() still works through iter_find(),
    # even when find() returns nothing.
    class _OnlyFind(JSONPath):
        def find(self, data):
            return []

    assert list(_OnlyFind().iter_find(object())) == []
