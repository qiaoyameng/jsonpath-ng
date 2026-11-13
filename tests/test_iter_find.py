import gc
import weakref

import pytest

import jsonpath_ng
import jsonpath_ng.ext
from jsonpath_ng import Fields
from jsonpath_ng.ext.filter import Filter


def _snapshot(datums):
    return [(datum.value, str(datum.full_path)) for datum in datums]


FIND_CASES = (
    pytest.param(jsonpath_ng.parse, "$", {"a": 1}, id="root"),
    pytest.param(jsonpath_ng.parse, "$.a", {"a": 1}, id="field"),
    pytest.param(jsonpath_ng.parse, "$.missing", {"a": 1}, id="empty-field"),
    pytest.param(jsonpath_ng.parse, "$[0]", [1, 2], id="index"),
    pytest.param(jsonpath_ng.parse, "$[0,1,0]", [1, 2], id="duplicate-index"),
    pytest.param(jsonpath_ng.parse, "$[0:4:2]", [1, 2, 3, 4], id="slice"),
    pytest.param(jsonpath_ng.parse, "$[9]", [1, 2], id="empty-index"),
    pytest.param(jsonpath_ng.parse, "$.*", {"a": {"x": 1}, "b": 2}, id="wildcard"),
    pytest.param(
        jsonpath_ng.parse,
        "$..a",
        {"first": {"a": 1}, "second": {"a": 2}, "other": 3},
        id="descendants",
    ),
    pytest.param(
        jsonpath_ng.parse,
        "$..*",
        {"a": [1, 2], "b": {"c": 3}},
        id="descendant-wildcard",
    ),
    pytest.param(
        jsonpath_ng.ext.parse,
        "$.items[?@.x > 0]",
        {"items": [{"x": 1}, {"x": 0}, {"x": 2}]},
        id="filter",
    ),
    pytest.param(
        jsonpath_ng.ext.parse,
        "$..[?@.x > 0]",
        {"a": [{"x": 1}, {"x": 0}], "b": [{"x": 2}]},
        id="descendant-filter",
    ),
    pytest.param(
        jsonpath_ng.ext.parse,
        "$.items[?@.x > 100]",
        {"items": [{"x": 1}, {"x": 2}]},
        id="empty-filter",
    ),
)


@pytest.mark.parametrize("parse, path, data", FIND_CASES)
def test_full_iteration_matches_find(parse, path, data):
    query = parse(path)

    expected = query.find(data)
    actual = list(query.iter_find(data))

    assert _snapshot(actual) == _snapshot(expected)


def test_duplicate_matches_and_full_paths_match_find():
    query = Fields("a", "a")
    data = {"a": {"b": 1}}

    expected = query.find(data)
    actual = list(query.iter_find(data))

    assert _snapshot(actual) == _snapshot(expected)
    assert [str(datum.full_path) for datum in actual] == ["a", "a"]


class _TrackingDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed = []

    def items(self):
        for key in super().keys():
            self.accessed.append(key)
            yield key, super().__getitem__(key)


@pytest.mark.parametrize(
    "path, expected",
    (
        pytest.param("$.*", [{"a": 1}, {"a": 2}], id="wildcard"),
        pytest.param("$..a", [1, 2], id="descendants"),
    ),
)
def test_early_close_does_not_access_later_members(path, expected):
    data = _TrackingDict(
        {
            "first": _TrackingDict({"a": 1}),
            "second": _TrackingDict({"a": 2}),
            "third": _TrackingDict({"a": 3}),
            "fourth": _TrackingDict({"a": 4}),
        }
    )
    query = jsonpath_ng.parse(path)

    iterator = query.iter_find(data)
    first_two = [next(iterator).value for _ in range(2)]
    iterator.close()

    assert first_two == expected
    assert "first" in data.accessed
    assert "second" in data.accessed
    assert "third" not in data.accessed
    assert "fourth" not in data.accessed


def test_closing_iterator_releases_unyielded_values():
    class Value:
        pass

    data = {"first": Value(), "second": Value()}
    query = jsonpath_ng.parse("$.*")
    iterator = query.iter_find(data)

    first = next(iterator)
    assert isinstance(first.value, Value)
    unreached = weakref.ref(data["second"])
    del data["second"]
    gc.collect()

    assert unreached() is None
    iterator.close()


def test_yielded_datum_is_not_repositioned_by_later_traversal():
    data = {"a": {"x": 1}, "b": {"x": 2}, "c": {"x": 3}}
    iterator = jsonpath_ng.parse("$..x").iter_find(data)

    first = next(iterator)
    path_before = str(first.full_path)
    remaining = list(iterator)

    assert path_before == "a.x"
    assert str(first.full_path) == path_before
    assert [datum.value for datum in remaining] == [2, 3]


class _ExplodingExpression:
    def __init__(self, explode_at):
        self.explode_at = explode_at
        self.seen = []

    def find(self, datum):
        self.seen.append(datum)
        if datum == self.explode_at:
            raise RuntimeError("filter exploded")
        return [datum]


def test_closing_filter_iterator_avoids_later_expression_errors():
    expression = _ExplodingExpression(explode_at=2)
    iterator = Filter([expression]).iter_find([1, 2])

    assert next(iterator).value == 1
    iterator.close()

    assert expression.seen == [1]


def test_filter_errors_match_find():
    expression = _ExplodingExpression(explode_at=2)
    query = Filter([expression])

    with pytest.raises(RuntimeError, match="filter exploded"):
        query.find([1, 2])

    with pytest.raises(RuntimeError, match="filter exploded"):
        list(query.iter_find([1, 2]))
