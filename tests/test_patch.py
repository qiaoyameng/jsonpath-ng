import copy

import pytest

from jsonpath_ng.ext import ABSENT
from jsonpath_ng.ext import PatchError
from jsonpath_ng.ext import VersionMismatchError
from jsonpath_ng.ext import VersionedDocument


def test_apply_patch_returns_versioned_events():
    store = VersionedDocument({"a": [1, 2, 3]})
    events = store.apply_patch(
        [{"op": "add", "path": "/a/0", "value": 0}]
    )
    assert store.version == 1
    assert store.document == {"a": [0, 1, 2, 3]}
    event = events[0]
    assert event.op == "add"
    assert event.path == ("a", 0)
    assert event.old_value is ABSENT
    assert event.shifted_from == [("a", 0), ("a", 1), ("a", 2)]
    assert event.shifted_to == [("a", 1), ("a", 2), ("a", 3)]


def test_remove_reports_shifts_and_old_value():
    store = VersionedDocument({"a": [1, 2, 3]})
    events = store.apply_patch([{"op": "remove", "path": "/a/1"}])
    event = events[0]
    assert event.op == "remove"
    assert event.old_value == 2
    assert event.new_value is ABSENT
    assert event.shifted_from == [("a", 2)]
    assert event.shifted_to == [("a", 1)]
    assert store.document == {"a": [1, 3]}


def test_replace_tracks_old_and_new_values():
    store = VersionedDocument({"a": {"b": 1}})
    events = store.apply_patch(
        [{"op": "replace", "path": "/a/b", "value": 2}]
    )
    event = events[0]
    assert event.op == "replace"
    assert event.old_value == 1
    assert event.new_value == 2


def test_base_version_mismatch_leaves_document_untouched():
    store = VersionedDocument({"a": 1})
    with pytest.raises(VersionMismatchError):
        store.apply_patch(
            [{"op": "replace", "path": "/a", "value": 2}], base_version=99
        )
    assert store.version == 0
    assert store.document == {"a": 1}


@pytest.mark.parametrize(
    "patch",
    (
        [{"op": "remove", "path": "/missing"}],
        [{"op": "replace", "path": "/a/0/b", "value": 1}],
        [{"op": "add", "path": "/a/99", "value": 1}],
        [{"op": "frobnicate", "path": "/a", "value": 1}],
        [{"op": "add", "path": "a", "value": 1}],
        [{"path": "/a", "value": 1}],
    ),
)
def test_illegal_patch_is_atomic(patch):
    original = {"a": [1, 2]}
    store = VersionedDocument(copy.deepcopy(original))
    with pytest.raises(PatchError):
        store.apply_patch(patch)
    assert store.document == original
    assert store.version == 0


def test_multi_operation_patch_validates_before_mutating():
    original = {"a": [1, 2, 3]}
    store = VersionedDocument(copy.deepcopy(original))
    with pytest.raises(PatchError):
        store.apply_patch(
            [
                {"op": "replace", "path": "/a/0", "value": 9},
                {"op": "remove", "path": "/a/99"},
            ]
        )
    assert store.document == original


def test_multi_operation_patch_applies_in_order():
    store = VersionedDocument({"a": [1, 2]})
    store.apply_patch(
        [
            {"op": "add", "path": "/a/0", "value": 0},
            {"op": "replace", "path": "/a/1", "value": 42},
        ]
    )
    assert store.document == {"a": [0, 42, 2]}


def test_escaped_json_pointer_tokens():
    store = VersionedDocument({"a/b": {"c~d": 1}})
    store.apply_patch(
        [{"op": "replace", "path": "/a~1b/c~0d", "value": 2}]
    )
    assert store.document == {"a/b": {"c~d": 2}}


def test_root_replacement():
    store = VersionedDocument({"a": 1})
    events = store.apply_patch(
        [{"op": "replace", "path": "", "value": {"b": 2}}]
    )
    assert events[0].path == ()
    assert store.document == {"b": 2}


def test_snapshot_is_independent_of_document():
    store = VersionedDocument({"a": [1]})
    snapshot = store.snapshot()
    store.apply_patch([{"op": "replace", "path": "/a/0", "value": 2}])
    assert snapshot == {"a": [1]}
