"""
Tests for `jsonpath_ng.session.QuerySession`: incremental JSONPath query
sessions over a stream of JSON Patch (RFC 6902) batches.
"""

import copy
import random

import pytest

from jsonpath_ng.exceptions import JSONPathError, JsonPatchError
from jsonpath_ng.ext import parse as ext_parse
from jsonpath_ng.session import (
    QuerySession,
    datum_path_tuple,
    path_to_pointer,
)


def find_map(expression, document):
    """The ground truth: a full ``find()`` as a {pointer: value} map."""
    if isinstance(expression, str):
        expression = ext_parse(expression)
    result = {}
    for datum in expression.find(document):
        try:
            path = datum_path_tuple(datum)
        except JSONPathError:
            path = str(datum.full_path)
        result[path_to_pointer(path)] = datum.value
    return result


def session_map(session):
    return {match.pointer: match.value for match in session.matches}


def assert_matches_find(session, expression):
    assert session_map(session) == find_map(expression, session.document)


# ---------------------------------------------------------------------------
# Incremental list insertion/removal
# ---------------------------------------------------------------------------

def test_array_insert_and_remove_match_full_find():
    document = {
        'items': [{'name': name, 'price': index}
                  for index, name in enumerate('abcde')],
        'meta': {'count': 5},
    }
    expression = '$.items[*].name'
    session = QuerySession(expression, document)

    diff = session.apply_patch([
        {'op': 'add', 'path': '/items/1', 'value': {'name': 'x', 'price': 99}},
    ])
    assert [match.pointer for match in diff.added] == ['/items/1/name']
    assert not diff.removed and not diff.changed
    assert {(m.old_pointer, m.new_pointer) for m in diff.moved} == {
        ('/items/1/name', '/items/2/name'),
        ('/items/2/name', '/items/3/name'),
        ('/items/3/name', '/items/4/name'),
        ('/items/4/name', '/items/5/name'),
    }
    assert_matches_find(session, expression)

    diff = session.apply_patch([{'op': 'remove', 'path': '/items/0'}])
    assert [match.pointer for match in diff.removed] == ['/items/0/name']
    assert not diff.added and not diff.changed
    assert {(m.old_pointer, m.new_pointer) for m in diff.moved} == {
        ('/items/2/name', '/items/1/name'),
        ('/items/3/name', '/items/2/name'),
        ('/items/4/name', '/items/3/name'),
        ('/items/5/name', '/items/4/name'),
        ('/items/1/name', '/items/0/name'),
    }
    assert_matches_find(session, expression)

    assert session.stats['batches_incremental'] == 2
    assert session.stats['full_requeries'] == 0


def test_slice_and_index_steps():
    document = {'items': [10, 20, 30, 40]}

    session = QuerySession('$.items[1:3]', document)
    assert session_map(session) == {'/items/1': 20, '/items/2': 30}
    diff = session.apply_patch([{'op': 'add', 'path': '/items/0', 'value': 5}])
    # The list is now [5, 10, 20, 30, 40]; the slice window shifted.
    assert session_map(session) == {'/items/1': 10, '/items/2': 20}
    assert [match.pointer for match in diff.added] == ['/items/1']
    assert {(m.old_pointer, m.new_pointer) for m in diff.moved} == {
        ('/items/1', '/items/2'),
    }
    assert [match.pointer for match in diff.removed] == ['/items/2']
    assert_matches_find(session, '$.items[1:3]')

    session = QuerySession('$.items[0]', {'items': [10, 20]})
    diff = session.apply_patch([{'op': 'replace', 'path': '/items/0', 'value': 99}])
    assert [(c.pointer, c.old_value, c.new_value) for c in diff.changed] == [
        ('/items/0', 10, 99),
    ]
    assert not diff.added and not diff.removed and not diff.moved


# ---------------------------------------------------------------------------
# Filter dependency fields trigger recomputation
# ---------------------------------------------------------------------------

def test_filter_condition_field_triggers_recompute():
    document = {'items': [
        {'name': 'a', 'stock': 0},
        {'name': 'b', 'stock': 3},
        {'name': 'c', 'stock': 0},
    ]}
    expression = '$.items[?(@.stock > 0)].name'
    session = QuerySession(expression, document)
    assert session_map(session) == {'/items/1/name': 'b'}

    # `stock` is not part of the output path, but the predicate reads it.
    diff = session.apply_patch([
        {'op': 'replace', 'path': '/items/2/stock', 'value': 5},
    ])
    assert [match.pointer for match in diff.added] == ['/items/2/name']
    assert not diff.removed and not diff.changed and not diff.moved
    assert_matches_find(session, expression)

    diff = session.apply_patch([
        {'op': 'replace', 'path': '/items/1/stock', 'value': 0},
    ])
    assert [match.pointer for match in diff.removed] == ['/items/1/name']
    assert not diff.added and not diff.changed and not diff.moved
    assert_matches_find(session, expression)

    assert session.stats['batches_incremental'] == 2
    assert session.stats['full_requeries'] == 0


# ---------------------------------------------------------------------------
# Position changes vs. value changes
# ---------------------------------------------------------------------------

def test_move_reports_position_not_value_changes():
    document = {'items': [{'id': i} for i in range(4)]}
    expression = '$.items[*].id'
    session = QuerySession(expression, document)

    diff = session.apply_patch([
        {'op': 'move', 'from': '/items/0', 'path': '/items/2'},
    ])
    assert document['items'] == [{'id': 1}, {'id': 2}, {'id': 0}, {'id': 3}]
    assert not diff.added and not diff.removed and not diff.changed
    assert {(m.old_pointer, m.new_pointer) for m in diff.moved} == {
        ('/items/0/id', '/items/2/id'),
        ('/items/1/id', '/items/0/id'),
        ('/items/2/id', '/items/1/id'),
    }
    assert_matches_find(session, expression)


def test_replace_reports_value_change():
    document = {'items': [{'name': 'a'}, {'name': 'b'}]}
    session = QuerySession('$.items[*].name', document)
    diff = session.apply_patch([
        {'op': 'replace', 'path': '/items/1/name', 'value': 'B'},
    ])
    assert [(c.pointer, c.old_value, c.new_value) for c in diff.changed] == [
        ('/items/1/name', 'b', 'B'),
    ]
    assert not diff.added and not diff.removed and not diff.moved


def test_copy_operation():
    document = {'items': [{'name': 'a'}]}
    session = QuerySession('$.items[*].name', document)
    diff = session.apply_patch([
        {'op': 'copy', 'from': '/items/0', 'path': '/items/-'},
    ])
    assert [match.pointer for match in diff.added] == ['/items/1/name']
    assert_matches_find(session, '$.items[*].name')


# ---------------------------------------------------------------------------
# Descendants
# ---------------------------------------------------------------------------

def test_descendants_incremental():
    document = {'a': {'b': {'price': 1}}, 'c': [{'price': 2}, {'x': 3}], 'z': 0}
    expression = '$..price'
    session = QuerySession(expression, document)
    assert session_map(session) == {'/a/b/price': 1, '/c/0/price': 2}

    diff = session.apply_patch([
        {'op': 'replace', 'path': '/c/0/price', 'value': 9},
    ])
    assert [(c.pointer, c.old_value, c.new_value) for c in diff.changed] == [
        ('/c/0/price', 2, 9),
    ]
    assert_matches_find(session, expression)

    diff = session.apply_patch([
        {'op': 'add', 'path': '/c/1/price', 'value': 7},
    ])
    assert [match.pointer for match in diff.added] == ['/c/1/price']
    assert_matches_find(session, expression)

    diff = session.apply_patch([{'op': 'remove', 'path': '/a/b'}])
    assert [match.pointer for match in diff.removed] == ['/a/b/price']
    assert_matches_find(session, expression)

    assert session.stats['full_requeries'] == 0


# ---------------------------------------------------------------------------
# Access counting: unrelated subtrees are not rescanned
# ---------------------------------------------------------------------------

class _AccessCounter:
    def __init__(self):
        self.count = 0


def _wrap_counted(value, counter):
    if isinstance(value, dict):
        return _CountingDict(value, counter)
    if isinstance(value, list):
        return _CountingList(value, counter)
    return value


class _CountingBase:
    def __init__(self, value, counter):
        super().__init__(value)
        self._counter = counter

    def _tick(self):
        self._counter.count += 1

    def __getitem__(self, key):
        self._tick()
        return super().__getitem__(key)

    def __iter__(self):
        self._tick()
        return super().__iter__()

    def __len__(self):
        self._tick()
        return super().__len__()

    def __contains__(self, key):
        self._tick()
        return super().__contains__(key)

    def items(self):
        self._tick()
        return super().items()

    def keys(self):
        self._tick()
        return super().keys()

    def values(self):
        self._tick()
        return super().values()

    def get(self, key, default=None):
        self._tick()
        return super().get(key, default)


class _CountingDict(_CountingBase, dict):
    pass


class _CountingList(_CountingBase, list):
    pass


def test_unrelated_subtree_patch_does_not_rescan():
    counter = _AccessCounter()
    books = [{'title': f'book-{i}', 'price': i} for i in range(50)]
    document = {
        'store': {
            'book': _wrap_counted(books, counter),
            'bicycle': {'color': 'red', 'price': 19.95},
        },
    }
    expression = '$.store.book[*].title'
    session = QuerySession(expression, document)
    assert len(session.matches) == 50
    counter.count = 0

    diff = session.apply_patch([
        {'op': 'replace', 'path': '/store/bicycle/color', 'value': 'blue'},
        {'op': 'add', 'path': '/store/bicycle/brand', 'value': 'trek'},
    ])
    assert diff.empty
    assert counter.count == 0, 'patching an unrelated subtree must not rescan'
    assert session.stats['batches_skipped'] == 1
    assert session.stats['batches_incremental'] == 0
    assert session.stats['full_requeries'] == 0

    # A field that is neither output nor filter input is also skipped.
    counter.count = 0
    diff = session.apply_patch([
        {'op': 'replace', 'path': '/store/book/0/price', 'value': 999},
    ])
    assert diff.empty
    # The query is not re-evaluated at all; only the patch itself navigates
    # to its target (a handful of accesses, not a scan of the 50 books).
    assert counter.count < 20
    assert session.stats['batches_skipped'] == 2

    # A relevant patch is incremental: it touches only a bounded number of
    # nodes, nowhere near a full scan of the 50 books.
    counter.count = 0
    diff = session.apply_patch([
        {'op': 'replace', 'path': '/store/book/0/title', 'value': 'new-title'},
    ])
    assert [(c.pointer, c.old_value, c.new_value) for c in diff.changed] == [
        ('/store/book/0/title', 'book-0', 'new-title'),
    ]
    assert session.stats['batches_incremental'] == 1
    assert session.stats['full_requeries'] == 0
    assert 0 < counter.count < 20
    assert_matches_find(session, expression)


# ---------------------------------------------------------------------------
# Unanalyzable extension nodes fall back to full queries
# ---------------------------------------------------------------------------

def test_unanalyzable_union_falls_back_to_full_query():
    document = {'items': [1, 2], 'other': {'x': 1}}
    expression = '($.items[0]) | ($.other.x)'
    session = QuerySession(expression, document)
    assert session.fallback
    assert session.stats['fallback_reasons']

    diff = session.apply_patch([{'op': 'replace', 'path': '/items/0', 'value': 9}])
    assert session.stats['full_requeries'] == 1
    assert [(c.pointer, c.old_value, c.new_value) for c in diff.changed] == [
        ('/items/0', 1, 9),
    ]
    assert_matches_find(session, expression)

    session.apply_patch([{'op': 'replace', 'path': '/other/x', 'value': 2}])
    assert session.stats['full_requeries'] == 2
    assert_matches_find(session, expression)


def test_unanalyzable_extension_keeps_static_prefix_for_skipping():
    document = {'items': [3, 1, 2], 'other': {'x': 1}}
    expression = '$.items.`sorted`'
    session = QuerySession(expression, document)
    assert session.fallback
    assert session.stats['fallback_reasons']

    # An unrelated patch is still skipped thanks to the analyzable prefix.
    diff = session.apply_patch([{'op': 'replace', 'path': '/other/x', 'value': 2}])
    assert diff.empty
    assert session.stats['batches_skipped'] == 1
    assert session.stats['full_requeries'] == 0

    # A relevant patch falls back to a full query, noted in the stats.
    diff = session.apply_patch([{'op': 'replace', 'path': '/items/0', 'value': 9}])
    assert session.stats['full_requeries'] == 1
    assert session.stats['batches_incremental'] == 0
    assert_matches_find(session, expression)

# ---------------------------------------------------------------------------
# Batch validation, ordering and atomicity
# ---------------------------------------------------------------------------

def test_failed_batch_publishes_no_partial_results():
    document = {'a': {'x': 1}, 'b': [1, 2, 3]}
    session = QuerySession('$.a.x', document)
    before = copy.deepcopy(document)

    with pytest.raises(JsonPatchError):
        session.apply_patch([
            {'op': 'replace', 'path': '/a/x', 'value': 9},
            {'op': 'remove', 'path': '/b/7'},
        ])
    assert document == before
    assert session_map(session) == {'/a/x': 1}
    assert session.stats['batches_failed'] == 1
    assert session.stats['batches_applied'] == 0

    with pytest.raises(JsonPatchError):
        session.apply_patch([
            {'op': 'add', 'path': '/b/-', 'value': 4},
            {'op': 'test', 'path': '/a/x', 'value': 999},
        ])
    assert document == before
    assert session_map(session) == {'/a/x': 1}
    assert session.stats['batches_failed'] == 2

    with pytest.raises(JsonPatchError):
        session.apply_patch([{'op': 'levitate', 'path': '/a'}])
    assert document == before
    assert session.stats['batches_failed'] == 3


def test_operations_apply_in_order():
    document = {'a': 1}
    session = QuerySession('$.a', document)
    diff = session.apply_patch([
        {'op': 'replace', 'path': '/a', 'value': 2},
        {'op': 'test', 'path': '/a', 'value': 2},
        {'op': 'add', 'path': '/b', 'value': 3},
    ])
    assert document == {'a': 2, 'b': 3}
    assert [(c.old_value, c.new_value) for c in diff.changed] == [(1, 2)]


def test_replace_document_root():
    document = {'a': 1}
    session = QuerySession('$.a', document)
    diff = session.apply_patch([
        {'op': 'replace', 'path': '', 'value': {'a': 2, 'b': 3}},
    ])
    assert session.document == {'a': 2, 'b': 3}
    assert [(c.old_value, c.new_value) for c in diff.changed] == [(1, 2)]
    assert_matches_find(session, '$.a')


# ---------------------------------------------------------------------------
# Randomized cross-check against full find()
# ---------------------------------------------------------------------------

def _random_operation(rng, document):
    items = document['items']
    nested = document['nested']['a']['b']
    index = rng.randrange(len(items))
    choice = rng.choice([
        'price', 'stock', 'name', 'insert', 'remove', 'append', 'move', 'copy',
        'nested_value', 'nested_add', 'nested_remove', 'nested_move',
        'flag', 'other',
    ])
    if choice == 'price':
        return {'op': 'replace', 'path': f'/items/{index}/price',
                'value': rng.randint(0, 9)}
    if choice == 'stock':
        return {'op': 'replace', 'path': f'/items/{index}/stock',
                'value': rng.randint(0, 4)}
    if choice == 'name':
        return {'op': 'replace', 'path': f'/items/{index}/name',
                'value': f'n{rng.randint(0, 9)}'}
    if choice == 'insert':
        return {'op': 'add', 'path': f'/items/{index}',
                'value': {'name': f'x{rng.randint(0, 9)}',
                          'price': rng.randint(0, 9),
                          'stock': rng.randint(0, 4)}}
    if choice == 'remove':
        if len(items) > 1:
            return {'op': 'remove', 'path': f'/items/{index}'}
        return {'op': 'replace', 'path': f'/items/{index}/price',
                'value': rng.randint(0, 9)}
    if choice == 'append':
        return {'op': 'add', 'path': '/items/-',
                'value': {'name': 'z', 'price': rng.randint(0, 9),
                          'stock': rng.randint(0, 4)}}
    if choice == 'move':
        return {'op': 'move', 'from': f'/items/{index}',
                'path': f'/items/{rng.randrange(len(items))}'}
    if choice == 'copy':
        return {'op': 'copy', 'from': f'/items/{index}',
                'path': f'/items/{rng.randrange(len(items))}'}
    if choice == 'nested_value':
        return {'op': 'replace', 'path': f'/nested/a/b/{rng.randrange(len(nested))}',
                'value': rng.randint(0, 9)}
    if choice == 'nested_add':
        return {'op': 'add', 'path': '/nested/a/b/-', 'value': rng.randint(0, 9)}
    if choice == 'nested_remove':
        if len(nested) > 1:
            return {'op': 'remove',
                    'path': f'/nested/a/b/{rng.randrange(len(nested))}'}
        return {'op': 'replace', 'path': '/nested/flag', 'value': False}
    if choice == 'nested_move':
        return {'op': 'move', 'from': f'/nested/a/b/{rng.randrange(len(nested))}',
                'path': f'/nested/a/b/{rng.randrange(len(nested))}'}
    if choice == 'flag':
        return {'op': 'replace', 'path': '/nested/flag',
                'value': rng.choice([True, False])}
    return {'op': 'replace', 'path': '/other', 'value': rng.randint(0, 5)}


def test_randomized_batches_match_full_find():
    rng = random.Random(20261005)
    expressions = [
        '$.items[*].price',
        '$.items[?(@.stock > 0)].name',
        '$.items[?(@.price >= 3)]',
        '$..price',
        '$.nested.a.b[*]',
        '$.items[1:4].name',
        '$.items[0].name',
        '$.items[*]',
        '$.nested',
        '$.items[?(@.name == "n1")].price',
        '$.items[?(@.price > 1 & @.stock < 3)]',
    ]
    for _ in range(10):
        base = {
            'items': [
                {'name': f'n{i}', 'price': rng.randint(0, 9),
                 'stock': rng.randint(0, 4)}
                for i in range(rng.randint(1, 6))
            ],
            'nested': {
                'a': {'b': [rng.randint(0, 9) for _ in range(rng.randint(1, 4))]},
                'flag': True,
            },
            'other': rng.randint(0, 5),
        }
        documents = {expression: copy.deepcopy(base) for expression in expressions}
        sessions = {
            expression: QuerySession(expression, documents[expression])
            for expression in expressions
        }
        for expression in expressions:
            assert_matches_find(sessions[expression], expression)
        for _ in range(15):
            operations = [
                _random_operation(rng, documents[expressions[0]])
                for _ in range(rng.randint(1, 3))
            ]
            outcomes = {}
            for expression in expressions:
                try:
                    outcomes[expression] = sessions[expression].apply_patch(operations)
                except JsonPatchError:
                    outcomes[expression] = None
            # All sessions observe the same document, so they must agree on
            # whether a batch is valid; failed batches change nothing.
            assert (all(outcome is None for outcome in outcomes.values())
                    or all(outcome is not None for outcome in outcomes.values()))
            for expression in expressions:
                assert_matches_find(sessions[expression], expression)
        for expression in expressions[1:]:
            assert documents[expression] == documents[expressions[0]]
