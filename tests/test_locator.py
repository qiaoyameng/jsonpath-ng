import random

import pytest

from jsonpath_ng import generate_locator, Locator, LocatorError
from jsonpath_ng.ext import parse


def find_one(expression, document):
    matches = parse(expression).find(document)
    assert len(matches) == 1, 'locator must match exactly one node: %r' % expression
    return matches[0]


def shuffled(seq, seed):
    seq = list(seq)
    random.Random(seed).shuffle(seq)
    return seq


class TestStableIdentity:
    def test_single_field_identity_survives_shuffle(self):
        users = [
            {'id': 'u1', 'name': 'amy'},
            {'id': 'u2', 'name': 'bob'},
            {'id': 'u3', 'name': 'cat'},
        ]
        document = {'users': users}
        locator = generate_locator(document, ['users', 1], ['id'])

        assert locator.stable
        assert locator.unstable_segments == ()
        assert locator.expression == "$['users'][?(@['id'] == 'u2')]"

        for seed in range(10):
            scrambled = dict(document, users=shuffled(users, seed))
            assert find_one(locator.expression, scrambled).value is users[1]

    def test_nested_arrays_survive_shuffle(self):
        rows = [
            [{'id': 'a', 'v': 1}, {'id': 'b', 'v': 2}],
            [{'id': 'c', 'v': 3}, {'id': 'd', 'v': 4}],
        ]
        document = {'rows': rows}
        locator = generate_locator(document, ['rows', 1, 0, 'v'], ['id'])

        # The outer level holds plain lists, so only the inner level is stable.
        assert not locator.stable
        assert locator.unstable_segments == (('rows', 1),)
        assert locator.expression == "$['rows'][1][?(@['id'] == 'c')]['v']"

        for seed in range(10):
            scrambled_rows = [shuffled(row, seed + i) for i, row in enumerate(rows)]
            scrambled = dict(document, rows=scrambled_rows)
            assert find_one(locator.expression, scrambled).value == 3

    def test_root_target(self):
        document = {'a': 1}
        locator = generate_locator(document, [], ['id'])
        assert locator.expression == '$'
        assert locator.stable
        assert find_one(locator.expression, document).value is document

    def test_scalar_and_null_targets(self):
        document = {'items': [{'id': 'x', 'v': None}, {'id': 'y', 'v': 0}]}
        locator = generate_locator(document, ['items', 0, 'v'], ['id'])
        assert locator.stable
        assert find_one(locator.expression, document).value is None


class TestDuplicateIdentities:
    def test_duplicate_ids_require_combination(self):
        items = [
            {'id': 'dup', 'sku': 's1'},
            {'id': 'dup', 'sku': 's2'},
            {'id': 'uniq', 'sku': 's2'},
        ]
        document = {'items': items}
        locator = generate_locator(document, ['items', 1], ['id', 'sku'])

        assert locator.stable
        assert "@['id'] == 'dup'" in locator.expression
        assert "@['sku'] == 's2'" in locator.expression
        assert '&' in locator.expression

        for seed in range(10):
            scrambled = dict(document, items=shuffled(items, seed))
            assert find_one(locator.expression, scrambled).value is items[1]

    def test_combination_size_grows_only_as_needed(self):
        items = [
            {'a': 1, 'b': 1, 'c': 1},
            {'a': 1, 'b': 1, 'c': 2},
            {'a': 1, 'b': 2, 'c': 1},
            {'a': 2, 'b': 1, 'c': 1},
        ]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], ['a', 'b', 'c'])
        # ('a', 'b') and ('a', 'c') are both ambiguous; only the triple works.
        assert locator.stable
        assert locator.expression.count('==') == 3

    def test_fixed_tie_breaking_rule(self):
        items = [{'x': 1, 'y': 2}, {'x': 9, 'y': 9}]
        document = {'items': items}
        # Both 'x' and 'y' are unique alone; the field listed first wins.
        locator = generate_locator(document, ['items', 0], ['y', 'x'])
        assert locator.expression == "$['items'][?(@['y'] == 2)]"
        locator = generate_locator(document, ['items', 0], ['x', 'y'])
        assert locator.expression == "$['items'][?(@['x'] == 1)]"

    def test_deterministic_output(self):
        items = [{'id': 'a', 'n': 1}, {'id': 'a', 'n': 2}]
        document = {'items': items}
        first = generate_locator(document, ['items', 0], ['id', 'n'])
        second = generate_locator(document, ['items', 0], ['id', 'n'])
        assert first == second


class TestIndexFallback:
    def test_identical_elements_fall_back_to_index(self):
        items = [{'id': 'same'}, {'id': 'same'}, {'id': 'same'}]
        document = {'items': items}
        locator = generate_locator(document, ['items', 1], ['id'])

        assert not locator.stable
        assert locator.unstable_segments == (('items', 1),)
        assert locator.expression == "$['items'][1]"
        assert find_one(locator.expression, document).value is items[1]

    def test_no_usable_fields_fall_back_to_index(self):
        items = [{'other': 1}, {'other': 2}]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], ['id'])
        assert not locator.stable
        assert locator.expression == "$['items'][0]"

    def test_non_dict_elements_fall_back_to_index(self):
        document = {'items': ['a', 'b', 'c']}
        locator = generate_locator(document, ['items', 2], ['id'])
        assert not locator.stable
        assert locator.expression == "$['items'][2]"

    def test_mixed_stable_and_unstable_segments(self):
        groups = [
            {'members': [{'id': 'm1'}, {'id': 'm2'}]},
            {'members': [{'id': 'm1'}, {'id': 'm2'}]},
        ]
        document = {'groups': groups}
        locator = generate_locator(document, ['groups', 1, 'members', 0], ['id'])

        assert not locator.stable
        assert locator.unstable_segments == (('groups', 1),)
        assert locator.expression == "$['groups'][1]['members'][?(@['id'] == 'm1')]"
        assert find_one(locator.expression, document).value is groups[1]['members'][0]

        # The stable inner segment still tracks its element after a shuffle.
        scrambled_members = shuffled(groups[1]['members'], 3)
        scrambled = {'groups': [groups[0], {'members': scrambled_members}]}
        assert find_one(locator.expression, scrambled).value is groups[1]['members'][0]


class TestSafeEncoding:
    def test_quoted_and_special_keys(self):
        weird_key = "we'ird ][' key"
        items = [{weird_key: 'k1', 'id': 'a'}, {weird_key: 'k2', 'id': 'b'}]
        document = {"o'uter": items}
        locator = generate_locator(document, ["o'uter", 1], [weird_key])

        assert locator.stable
        assert find_one(locator.expression, document).value is items[1]
        scrambled = {"o'uter": shuffled(items, 5)}
        assert find_one(locator.expression, scrambled).value is items[1]

    def test_hostile_string_value_cannot_inject(self):
        hostile = "x' & @['admin'] == true & @['id'] == '"
        items = [
            {'id': hostile, 'admin': False},
            {'id': 'x', 'admin': True},
        ]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], ['id'])

        assert locator.stable
        # The generated expression must still select exactly the first item.
        assert find_one(locator.expression, document).value is items[0]
        scrambled = {'items': shuffled(items, 7)}
        assert find_one(locator.expression, scrambled).value is items[0]

    def test_hostile_key_cannot_inject(self):
        hostile_key = "a'] == true & @['b"
        items = [{hostile_key: 1, 'id': 't'}, {hostile_key: 2, 'id': 'o'}]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], [hostile_key])
        assert locator.stable
        assert find_one(locator.expression, document).value is items[0]

    def test_value_types(self):
        items = [
            {'id': 7, 'f': 1.5, 'ok': True, 's': 'txt'},
            {'id': '7', 'f': 2.5, 'ok': False, 's': 'txt2'},
        ]
        document = {'items': items}
        # The query engine coerces '7' to 7 for `==`, so the int identity
        # is NOT unique here and the locator must not claim stability.
        locator = generate_locator(document, ['items', 0], ['id'])
        assert not locator.stable
        assert locator.expression == "$['items'][0]"
        assert find_one(locator.expression, document).value is items[0]

        items = [dict(items[0], id=7), dict(items[1], id=8)]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], ['id'])
        assert locator.stable
        assert "@['id'] == 7" in locator.expression
        assert find_one(locator.expression, document).value is items[0]

        for field, literal in [('f', '1.5'), ('ok', 'true'), ('s', "'txt'")]:
            loc = generate_locator(document, ['items', 0], [field])
            assert loc.stable
            assert literal in loc.expression
            assert find_one(loc.expression, document).value is items[0]


class TestNullAndMissingValues:
    def test_null_identity_field_is_not_used(self):
        items = [
            {'id': None, 'sku': 's1'},
            {'id': None, 'sku': 's2'},
        ]
        document = {'items': items}
        locator = generate_locator(document, ['items', 1], ['id', 'sku'])
        assert locator.stable
        assert "@['sku'] == 's2'" in locator.expression
        assert 'null' not in locator.expression

    def test_null_only_identity_falls_back(self):
        items = [{'id': None}, {'id': None}]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], ['id'])
        assert not locator.stable
        assert locator.expression == "$['items'][0]"

    def test_missing_field_is_skipped(self):
        items = [{'sku': 's1'}, {'id': 'i2', 'sku': 's2'}]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], ['id', 'sku'])
        assert locator.stable
        assert locator.expression == "$['items'][?(@['sku'] == 's1')]"


class TestVerificationAndErrors:
    def test_result_is_verified_unique_hit(self):
        document = {'items': [{'id': 'a'}, {'id': 'b'}]}
        locator = generate_locator(document, ['items', 1], ['id'])
        matches = parse(locator.expression).find(document)
        assert len(matches) == 1
        assert matches[0].value is document['items'][1]

    def test_bad_path_raises(self):
        document = {'items': [{'id': 'a'}]}
        with pytest.raises(LocatorError):
            generate_locator(document, ['missing'], ['id'])
        with pytest.raises(LocatorError):
            generate_locator(document, ['items', 5], ['id'])
        with pytest.raises(LocatorError):
            generate_locator(document, ['items', 0, 'id', 0], ['id'])

    def test_bad_arguments_raise(self):
        document = {'items': []}
        with pytest.raises(LocatorError):
            generate_locator(document, 'items', ['id'])
        with pytest.raises(LocatorError):
            generate_locator(document, [True], ['id'])
        with pytest.raises(LocatorError):
            generate_locator(document, [1.5], ['id'])
        with pytest.raises(LocatorError):
            generate_locator(document, [], 'id')

    def test_max_combination_size(self):
        items = [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}, {'a': 2, 'b': 1}]
        document = {'items': items}
        locator = generate_locator(document, ['items', 0], ['a', 'b'],
                                   max_combination_size=1)
        assert not locator.stable
        locator = generate_locator(document, ['items', 0], ['a', 'b'])
        assert locator.stable
