# -*- coding: utf-8 -*-
"""
test_locator
------------

Tests for `jsonpath_ng.locator` (stable locator generation).
"""

import random

import pytest

from jsonpath_ng import Fields, JSONPath
from jsonpath_ng.ext import parse as ext_parse
from jsonpath_ng.locator import (
    LocatorError,
    generate_locator,
)


def locate(document, result):
    """Evaluate a generated locator with the real query engine."""
    matches = ext_parse(result.locator).find(document)
    assert len(matches) == 1
    return matches[0].value


def test_identity_field_preferred_over_index():
    document = {'users': [
        {'id': 'u-1', 'name': 'amy'},
        {'id': 'u-2', 'name': 'bob'},
        {'id': 'u-3', 'name': 'cy'},
    ]}
    result = generate_locator(document, '$.users[1].name', ['id'])
    assert result.locator == "$['users'][?(@['id'] == 'u-2')]['name']"
    assert result.stable
    assert locate(document, result) == 'bob'


def test_locator_survives_array_shuffle():
    document = {'users': [
        {'id': 'u-1', 'name': 'amy'},
        {'id': 'u-2', 'name': 'bob'},
        {'id': 'u-3', 'name': 'cy'},
    ]}
    target = document['users'][1]
    result = generate_locator(document, ['users', 1], ['id'])

    rng = random.Random(20261005)
    for _ in range(20):
        rng.shuffle(document['users'])
        found = locate(document, result)
        assert found is target
        assert found['name'] == 'bob'


def test_duplicate_ids_require_field_combination():
    document = {'users': [
        {'id': 'a', 'name': 'x', 'tags': []},
        {'id': 'a', 'name': 'y', 'tags': []},
        {'id': 'b', 'name': 'x', 'tags': []},
    ]}
    # Neither 'id' nor 'name' is unique on its own, but the pair is.
    result = generate_locator(document, '$.users[0]', ['id', 'name'])
    assert result.locator == (
        "$['users'][?(@['id'] == 'a' & @['name'] == 'x')]")
    assert result.stable
    assert locate(document, result) is document['users'][0]


def test_combination_follows_allowlist_order_for_ties():
    document = {'items': [
        {'a': 'x', 'b': 'y', 'c': 1},
        {'a': 'x', 'b': 'z', 'c': 2},
        {'a': 'w', 'b': 'y', 'c': 3},
        {'a': 'v', 'b': 'q', 'c': 1},
    ]}
    # No single field is unique; ('a', 'b'), ('a', 'c') and ('b', 'c')
    # all are.  The allowlist order decides deterministically.
    result = generate_locator(document, '$.items[0]', ['a', 'b', 'c'])
    assert result.locator == (
        "$['items'][?(@['a'] == 'x' & @['b'] == 'y')]")

    result = generate_locator(document, '$.items[0]', ['b', 'c', 'a'])
    assert result.locator == (
        "$['items'][?(@['b'] == 'y' & @['c'] == 1)]")

    result = generate_locator(document, '$.items[0]', ['c', 'a', 'b'])
    assert result.locator == (
        "$['items'][?(@['c'] == 1 & @['a'] == 'x')]")


def test_shortest_combination_wins_over_allowlist_order():
    document = {'items': [
        {'a': 'x', 'b': 'y'},
        {'a': 'x', 'b': 'q'},
    ]}
    # 'b' alone is unique even though 'a' comes first in the allowlist.
    result = generate_locator(document, '$.items[0]', ['a', 'b'])
    assert result.locator == "$['items'][?(@['b'] == 'y')]"


def test_index_fallback_is_marked_unstable():
    document = {'users': [
        {'name': 'amy'},
        {'name': 'amy'},
    ]}
    result = generate_locator(document, '$.users[1]', ['id'])
    assert result.locator == "$['users'][1]"
    assert not result.stable
    assert len(result.unstable_segments) == 1
    segment = result.unstable_segments[0]
    assert segment.kind == 'index'
    assert segment.position == 1
    assert segment.reason == 'no unique identity field combination'
    assert locate(document, result) is document['users'][1]


def test_identical_twins_fall_back_to_index():
    # Two elements with equal content: no identity can separate them, and
    # the locator must not pretend to be stable.
    document = {'users': [
        {'id': 'same', 'name': 'amy'},
        {'id': 'same', 'name': 'amy'},
    ]}
    result = generate_locator(document, '$.users[0]', ['id', 'name'])
    assert result.locator == "$['users'][0]"
    assert not result.stable


def test_lookalike_values_are_not_treated_as_unique():
    # The engine coerces candidates with int() for integer literals, so
    # the string '5' also matches [?(@['id'] == 5)]: 'id' is not unique
    # here even though the values are distinct in the document.
    document = {'items': [
        {'id': 5, 'name': 'amy'},
        {'id': '5', 'name': 'bob'},
        {'id': 7, 'name': 'amy'},
    ]}
    result = generate_locator(document, '$.items[0]', ['id'])
    assert result.locator == "$['items'][0]"
    assert not result.stable

    # 'name' alone is not unique either; the combination is.
    result = generate_locator(document, '$.items[0]', ['id', 'name'])
    assert result.locator == (
        "$['items'][?(@['id'] == 5 & @['name'] == 'amy')]")
    assert locate(document, result) is document['items'][0]


def test_quoted_and_special_key_names():
    document = {
        'weird "key"': [
            {"it's": 'a', 'val': 1},
            {"it's": 'b', 'val': 2},
        ],
    }
    result = generate_locator(
        document, ["weird \"key\"", 1, 'val'], ["it's"])
    assert result.locator == (
        "$['weird \"key\"'][?(@['it\\'s'] == 'b')]['val']")
    assert result.stable
    assert locate(document, result) == 2


def test_values_with_quotes_backslashes_and_newlines():
    tricky = "quo'te\"d \\ backslash\nnewline"
    document = {'items': [
        {'id': tricky, 'n': 1},
        {'id': 'plain', 'n': 2},
    ]}
    result = generate_locator(document, '$.items[0]', ['id'])
    assert result.stable
    assert locate(document, result) is document['items'][0]


def test_identity_values_cannot_inject_expressions():
    # Values that look like JSONPath fragments must stay inert data.
    payload = "x')] | $['secret'] | @['id'] == '"
    document = {
        'items': [
            {'id': payload, 'n': 'target'},
            {'id': 'other', 'n': 'decoy'},
        ],
        'secret': ['must-not-leak'],
    }
    result = generate_locator(document, '$.items[0].n', ['id'])
    assert result.stable
    assert locate(document, result) == 'target'
    # The parsed locator must not reference any field beyond the path.
    referenced_fields = set()
    stack = [ext_parse(result.locator)]
    while stack:
        node = stack.pop()
        if isinstance(node, Fields):
            referenced_fields.update(node.fields)
        for attr in ('left', 'right', 'target'):
            child = getattr(node, attr, None)
            if isinstance(child, JSONPath):
                stack.append(child)
        stack.extend(getattr(node, 'expressions', None) or [])
    assert referenced_fields == {'items', 'id', 'n'}


def test_null_values_are_handled_safely():
    # None cannot be expressed as a filter literal; other fields must be
    # used instead, and None-valued siblings must not break matching.
    document = {'items': [
        {'id': None, 'name': 'amy'},
        {'id': 'i-2', 'name': 'bob'},
        {'id': 'i-3', 'name': 'cy'},
    ]}
    result = generate_locator(document, '$.items[1]', ['id', 'name'])
    assert result.locator == "$['items'][?(@['id'] == 'i-2')]"
    assert locate(document, result) is document['items'][1]

    # The None-valued element itself falls back to 'name'.
    result = generate_locator(document, '$.items[0]', ['id', 'name'])
    assert result.locator == "$['items'][?(@['name'] == 'amy')]"
    assert result.stable

    # With only 'id' allowed, the None-valued element has no identity.
    result = generate_locator(document, '$.items[0]', ['id'])
    assert result.locator == "$['items'][0]"
    assert not result.stable


def test_null_target_value():
    document = {'items': [{'id': 'a'}, {'id': 'b'}], 'extra': None}
    result = generate_locator(document, '$.extra', ['id'])
    assert result.locator == "$['extra']"
    assert locate(document, result) is None


def test_nested_arrays():
    document = {'groups': [
        {'gid': 'g1', 'members': [{'mid': 'm1', 'v': 1},
                                  {'mid': 'm2', 'v': 2}]},
        {'gid': 'g2', 'members': [{'mid': 'm1', 'v': 3},
                                  {'mid': 'm2', 'v': 4}]},
    ]}
    result = generate_locator(
        document, '$.groups[1].members[0].v', ['gid', 'mid'])
    assert result.locator == (
        "$['groups'][?(@['gid'] == 'g2')]"
        "['members'][?(@['mid'] == 'm1')]['v']")
    assert result.stable

    # Shuffling both levels still finds the original value.
    rng = random.Random(7)
    for _ in range(20):
        rng.shuffle(document['groups'])
        for group in document['groups']:
            rng.shuffle(group['members'])
        assert locate(document, result) == 3


def test_nested_arrays_without_identity_use_indices():
    document = {'matrix': [[1, 2], [3, 4]]}
    result = generate_locator(document, '$.matrix[1][0]', ['id'])
    assert result.locator == "$['matrix'][1][0]"
    assert not result.stable
    assert [s.reason for s in result.unstable_segments] == [
        'array element is not an object',
        'array element is not an object',
    ]
    assert locate(document, result) == 3


def test_bool_and_float_identity_values():
    document = {'items': [
        {'flag': True, 'ratio': 0.5, 'name': 'amy'},
        {'flag': False, 'ratio': 1.5, 'name': 'bob'},
    ]}
    result = generate_locator(document, '$.items[0]', ['flag'])
    assert result.locator == "$['items'][?(@['flag'] == true)]"
    assert locate(document, result) is document['items'][0]

    result = generate_locator(document, '$.items[1]', ['ratio'])
    assert result.locator == "$['items'][?(@['ratio'] == 1.5)]"
    assert locate(document, result) is document['items'][1]


def test_non_scalar_identity_values_are_ignored():
    document = {'items': [
        {'id': [1, 2], 'name': 'amy'},
        {'id': [1, 2], 'name': 'bob'},
    ]}
    result = generate_locator(document, '$.items[1]', ['id', 'name'])
    assert result.locator == "$['items'][?(@['name'] == 'bob')]"


def test_target_path_input_forms():
    document = {'a': [{'b': 1}]}
    expected = "$['a'][0]['b']"
    assert generate_locator(document, '$.a[0].b').locator == expected
    assert generate_locator(document, "$['a'][0]['b']").locator == expected
    assert generate_locator(document, 'a[0].b').locator == expected
    assert generate_locator(document, ['a', 0, 'b']).locator == expected
    assert generate_locator(document, '$').locator == '$'
    assert generate_locator(document, []).locator == '$'


def test_non_normalized_path_is_rejected():
    document = {'a': [{'b': 1}]}
    with pytest.raises(ValueError):
        generate_locator(document, '$..b')
    with pytest.raises(ValueError):
        generate_locator(document, '$.a[*]')
    with pytest.raises(ValueError):
        generate_locator(document, '$.a[0].b | $.a')
    with pytest.raises(TypeError):
        generate_locator(document, ['a', 0.5])


def test_missing_target_path_is_rejected():
    document = {'a': [{'b': 1}]}
    with pytest.raises(LocatorError):
        generate_locator(document, '$.missing')
    with pytest.raises(LocatorError):
        generate_locator(document, '$.a[5]')
    with pytest.raises(LocatorError):
        generate_locator(document, '$.a[0].b[1]')
    with pytest.raises(LocatorError):
        generate_locator(document, '$.a[0].b.c')


def test_wildcard_field_name_is_rejected():
    document = {'*': [{'id': 'a'}]}
    with pytest.raises(LocatorError):
        generate_locator(document, ['*', 0], ['id'])


def test_result_find_helper_and_str():
    document = {'users': [{'id': 'u-1'}, {'id': 'u-2'}]}
    result = generate_locator(document, '$.users[1]', ['id'])
    assert str(result) == result.locator
    matches = result.find(document)
    assert len(matches) == 1
    assert matches[0].value is document['users'][1]
