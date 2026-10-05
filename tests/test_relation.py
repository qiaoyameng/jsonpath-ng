"""Tests for jsonpath_ng.relation (static containment checking)."""

import json

import pytest

from jsonpath_ng import parse
from jsonpath_ng.jsonpath import (
    Child,
    Descendants,
    Fields,
    Index,
    Root,
    Slice,
    This,
    Union,
)
from jsonpath_ng.relation import (
    Counterexample,
    Relation,
    check_relation,
    is_covered,
    replay,
)


# ---------------------------------------------------------------------------
# Direct verdict checks
# ---------------------------------------------------------------------------

CONTAINED_CASES = [
    # identity and reflexivity
    ('$', ['$']),
    ('$.a', ['$.a']),
    ('$.a.b', ['$.a.b']),
    ('$.a[0]', ['$.a[0]']),
    ('$..a', ['$..a']),
    # fields and wildcards (field names are atoms, never text prefixes)
    ('$.a', ['$.*']),
    ('$.ab', ['$.*']),
    ('$.a', ['$.a', '$.ab']),
    ('$.a,b', ['$.*']),
    ('$.a.b', ['$.a.*']),
    ('$.a.b', ['$.*.b']),
    # indices and slices: array-length boundaries
    ('$.a[0]', ['$.a[*]']),
    ('$.a[0]', ['$.a[0:1]']),
    ('$.a[1]', ['$.a[0:2]']),
    ('$.a[0:2]', ['$.a[0:5]']),
    ('$.a[0:2]', ['$.a[*]']),
    ('$.a[1:]', ['$.a[*]']),
    ('$.a[::2]', ['$.a[*]']),
    ('$.a[1::2]', ['$.a[1:]']),
    ('$.a[0:10:2]', ['$.a[::2]']),
    ('$.a[0,2]', ['$.a[*]']),
    ('$.a[0,2]', ['$.a[0:3]']),
    ('$[2]', ['$[1:4]']),
    # descendants
    ('$.a.b', ['$..b']),
    ('$.a', ['$..a']),
    ('$..a', ['$..*']),
    ('$.a.b', ['$..*']),
    ('$[0]', ['$..[*]']),
    ('$..[0]', ['$..[*]']),
    ('$.a..b', ['$..b']),
    ('$.a[0]', ['$.a..[0]']),
    # unions
    ('($.a) | ($.b)', ['$.*']),
    ('$.a', ['($.a) | ($.b)']),
    ('($.a[0]) | ($.a[1])', ['$.a[0:2]']),
    ('$.a[0,1]', ['($.a[0]) | ($.a[1])']),
    ('$.a[0:2]', ['$.a[0]', '$.a[1]']),
    ('$.a', ['$.b', '$.a']),
    # empty denotation is contained in anything, even nothing
    ('$.a[2:1]', ['$.b']),
    ('$.a[2:1]', []),
    ('$.a[2:2]', ['$.*']),
    # a root reset behind an empty prefix can never fire
    ('$.a[2:1].$', ['$.b']),
    # root / this
    ('$', ['$']),
    ('`this`', ['$']),
    ('$.a.`this`', ['$.a']),
    ('$..$', ['$']),
    # multiple allowed selectors combine
    ('$.a[0:3]', ['$.a[0]', '$.a[1]', '$.a[2]']),
    ('$.a.b', ['$.a.b', '$.c']),
]

NOT_CONTAINED_CASES = [
    # (selector, allowed, expected out-of-bounds path)
    ('$.a', ['$.ab'], '$.a'),
    ('$.ab', ['$.a'], '$.ab'),
    ('$.*', ['$.a'], '$.b'),
    ('$.a.b', ['$.a'], '$.a.b'),
    ('$.a', ['$.a.b'], '$.a'),
    ('$.a[*]', ['$.a[0]'], '$.a[1]'),
    ('$.a[0:2]', ['$.a[1]'], '$.a[0]'),
    ('$.a[2]', ['$.a[0:2]'], '$.a[2]'),
    ('$.a[1:]', ['$.a[0]'], '$.a[1]'),
    ('$.a[1::2]', ['$.a[::2]'], '$.a[1]'),
    ('$.a[::2]', ['$.a[0:10:2]'], '$.a[10]'),
    ('$.a[*]', ['$.a[0:2]'], '$.a[2]'),
    ('$.a[0:3]', ['$.a[0]', '$.a[1]'], '$.a[2]'),
    ('$..b', ['$.a.b'], '$.b'),
    ('$..*', ['$..a'], '$.b'),
    ('$..[*]', ['$..*'], '$[0]'),
    ('$[0]', ['$..*'], '$[0]'),
    ('$.a', ['$..b'], '$.a'),
    ('$', ['$.a'], '$'),
    ('$.a', [], '$.a'),
    ('$.a.b', ['$.a.c'], '$.a.b'),
    ('$.a[0]', ['$.a[1]'], '$.a[0]'),
    ('$..a', ['$.a'], '$.a.a'),
    ('$.a[0].b', ['$.a[0].c'], '$.a[0].b'),
]

UNKNOWN_CASES = [
    '$.a[?(@.b)]',          # filters are not supported
    '$..[?(@.x > 1)]',
    '$.a[-1]',              # negative index
    '$.a[::-1]',            # negative step
    '$.a[::0]',             # zero step
    '$.a[-2:]',             # negative slice bound
    '$.a where $.b',        # where
    '$.a wherenot $.b',     # wherenot
    '$.a & $.b',            # intersection
    '$.a.`parent`',         # parent operator
    '$.a.$',                # guarded root reset
    '$.a..$',               # guarded root reset under descendants
    '($.a.$).b',            # guarded root reset in the middle
    'not a jsonpath',
    '',
    '$.a[1',
]


@pytest.mark.parametrize('selector,allowed', CONTAINED_CASES)
def test_contained(selector, allowed):
    result = check_relation(selector, allowed)
    assert result.relation is Relation.CONTAINED, result.reason
    assert result.contained
    assert bool(result)
    assert result.counterexample is None
    assert is_covered(selector, allowed)


@pytest.mark.parametrize('selector,allowed,path', NOT_CONTAINED_CASES)
def test_not_contained(selector, allowed, path):
    result = check_relation(selector, allowed)
    assert result.relation is Relation.NOT_CONTAINED, result.reason
    assert not result.contained
    assert not bool(result)
    assert not is_covered(selector, allowed)
    counterexample = result.counterexample
    assert counterexample is not None
    assert counterexample.path == path
    assert counterexample.verified
    # the witness document is a small JSON-serializable value
    json.dumps(counterexample.document)


@pytest.mark.parametrize('selector', UNKNOWN_CASES)
def test_unknown_selector(selector):
    result = check_relation(selector, ['$.*'])
    assert result.relation is Relation.UNKNOWN
    assert result.unknown
    assert not result.contained
    assert not bool(result)
    assert not is_covered(selector, ['$.*'])
    assert result.reason


def test_unknown_allowed_selector():
    result = check_relation('$.a', ['$.a', '$.b[?(@.c)]'])
    assert result.relation is Relation.UNKNOWN


def test_unparseable_allowed_selector():
    result = check_relation('$.a', ['not a jsonpath'])
    assert result.relation is Relation.UNKNOWN


def test_invalid_input_type():
    result = check_relation(42, ['$.a'])
    assert result.relation is Relation.UNKNOWN
    result = check_relation('$.a', [object()])
    assert result.relation is Relation.UNKNOWN


# ---------------------------------------------------------------------------
# Counterexamples replay with find()
# ---------------------------------------------------------------------------

def _full_paths(selector, document):
    return [datum.full_path for datum in parse(selector).find(document)]


@pytest.mark.parametrize('selector,allowed,path', NOT_CONTAINED_CASES)
def test_counterexample_replays_with_find(selector, allowed, path):
    result = check_relation(selector, allowed)
    counterexample = result.counterexample
    document = counterexample.document

    # the new selector matches the out-of-bounds path on the document ...
    assert counterexample.path_ast in _full_paths(selector, document)
    # ... and no allowed selector does
    for allowed_selector in allowed:
        assert counterexample.path_ast not in _full_paths(allowed_selector, document)


def test_counterexample_replay_helper():
    result = check_relation('$.a[*]', ['$.a[0]'])
    counterexample = result.counterexample
    assert counterexample.replay('$.a[*]', ['$.a[0]'])
    assert replay(counterexample, '$.a[*]', ['$.a[0]'])
    # replay fails when the selector no longer escapes the allowed set
    assert not replay(counterexample, '$.a[0]', ['$.a[0]'])


def test_counterexample_is_deterministic():
    first = check_relation('$.*', ['$.a']).counterexample
    second = check_relation('$.*', ['$.a']).counterexample
    assert first == second


def test_counterexample_for_empty_allowed_set():
    result = check_relation('$.a.b[2]', [])
    assert result.relation is Relation.NOT_CONTAINED
    assert result.counterexample.path == '$.a.b[2]'
    assert result.counterexample.document == {'a': {'b': [None, None, None]}}


def test_counterexample_path_quoting():
    result = check_relation("($['a b']) | ($.c)", ['$.c'])
    assert result.relation is Relation.NOT_CONTAINED
    assert result.counterexample.path == "$['a b']"
    assert result.counterexample.replay("($['a b']) | ($.c)", ['$.c'])


# ---------------------------------------------------------------------------
# Parsed ASTs are accepted as input
# ---------------------------------------------------------------------------

def test_accepts_parsed_expressions():
    result = check_relation(parse('$.a[0]'), [parse('$.a[*]')])
    assert result.relation is Relation.CONTAINED
    result = check_relation(parse('$.a[*]'), [parse('$.a[0]')])
    assert result.relation is Relation.NOT_CONTAINED
    # mixed strings and ASTs
    assert check_relation('$.a[0]', [parse('$.a[*]')]).contained


def test_allowed_set_as_tuple():
    assert check_relation('$.a', ('$.a', '$.b')).contained


def test_unparenthesized_union_follows_parser_precedence():
    # The parser binds '|' tighter than '.', so `$.a | $.b` parses as
    # `($.(a|$)).b` and matches $.a.b and $.b -- the checker follows the
    # actual find() semantics rather than the intended union.
    result = check_relation('$.a | $.b', ['$.*'])
    assert result.relation is Relation.NOT_CONTAINED
    assert result.counterexample.path == '$.a.b'
    assert check_relation('$.a | $.b', ['$.*', '$.a.b']).contained


# ---------------------------------------------------------------------------
# Cross-validation against an independent structural reference matcher
# ---------------------------------------------------------------------------
#
# The reference matcher below implements the documented semantics directly:
# selectors denote sets of structural paths over objects and arrays (no
# slice coercion quirks, no string indexing).  The checker must agree with
# brute-force enumeration over a pool of documents.

CROSS_CHECK_SELECTORS = [
    '$', '$.a', '$.b', '$.abc', '$.*', '$.a.b', '$.a.c', '$[0]', '$[1]', '$[2]',
    '$.a[0]', '$.a[1]', '$.a[3]', '$.a[*]', '$.a[0:1]', '$.a[0:2]', '$.a[1:2]',
    '$.a[1:]', '$.a[:2]', '$.a[::2]', '$.a[1::2]', '$.a[0:4:2]', '$.a[2:1]',
    '$.a[0,2]', '$..a', '$..b', '$..*', '$..[*]', '$..[0]', '$.a..b', '$.a..*',
    '($.a) | ($.b)', '($.a[0]) | ($.a[1])', '$.*.a', '$[*]', '$[*].a', '$.a.b[0]',
    '($.a[0:2]) | ($.b)', '$..a[0]', '$..*..b', '$.a.b.c', '$.a[0][1]', '$.a[*].b',
    '$.a[0].b', '$.a,b', '$.a..[0]', '$[0][0]', '$.a[0:10:3]',
]


def _reference_match(ast, node):
    """All (path, value) pairs matched by ``ast`` at ``node`` (structural)."""
    if isinstance(ast, (Root, This)):
        return [((), node)]
    if isinstance(ast, Fields):
        if not isinstance(node, dict):
            return []
        if '*' in ast.fields:
            names = list(node.keys())
        else:
            names = [name for name in ast.fields if name in node]
        return [((('f', name),), node[name]) for name in names]
    if isinstance(ast, Index):
        if not isinstance(node, list):
            return []
        return [
            ((('i', index),), node[index])
            for index in ast.indices
            if 0 <= index < len(node)
        ]
    if isinstance(ast, Slice):
        if not isinstance(node, list):
            return []
        step = ast.step if ast.step is not None else 1
        start = ast.start if ast.start is not None else 0
        return [
            ((('i', i),), node[i])
            for i in range(len(node))[start:ast.end:step]
        ]
    if isinstance(ast, Child):
        out = []
        for path1, value1 in _reference_match(ast.left, node):
            for path2, value2 in _reference_match(ast.right, value1):
                out.append((path1 + path2, value2))
        return out
    if isinstance(ast, Descendants):
        out = []
        for path1, value1 in _reference_match(ast.left, node):
            for path2, value2 in _reference_match(ast.right, value1):
                out.append((path1 + path2, value2))
            for sub_path, sub_value in _reference_descendants(value1):
                for path2, value2 in _reference_match(ast.right, sub_value):
                    out.append((path1 + sub_path + path2, value2))
        return out
    if isinstance(ast, Union):
        return _reference_match(ast.left, node) + _reference_match(ast.right, node)
    raise AssertionError('unexpected AST node: %r' % (ast,))


def _reference_descendants(node):
    """All (path, value) pairs of proper descendants of ``node``."""
    out = []
    if isinstance(node, dict):
        items = [(('f', key), value) for key, value in node.items()]
    elif isinstance(node, list):
        items = [(('i', index), value) for index, value in enumerate(node)]
    else:
        return out
    for step, value in items:
        out.append(((step,), value))
        for path, sub_value in _reference_descendants(value):
            out.append(((step,) + path, sub_value))
    return out


_REFERENCE_CACHE = {}


def _reference_paths(selector, document):
    key = (selector, json.dumps(document, sort_keys=True))
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = {
            path for path, _ in _reference_match(parse(selector), document)
        }
    return _REFERENCE_CACHE[key]


def _reference_violation(selector, allowed, document):
    """A path matched by ``selector`` but by no allowed selector, if any."""
    new_paths = _reference_paths(selector, document)
    covered = set()
    for allowed_selector in allowed:
        covered |= _reference_paths(allowed_selector, document)
    for path in sorted(new_paths - covered):
        return path
    return None


def _document_pool():
    docs = [None, {}, [], [None], [None, None], {'a': None}, {'b': None},
            {'a': None, 'b': None}, {'abc': None}]
    inner = [[None], [None, None], [None, None, None], {'a': None},
             {'b': None}, None, [], {}]
    for key in ('a', 'b', 'abc'):
        for value in inner:
            docs.append({key: value})
    docs.append({'a': {'b': {'a': None}}})
    docs.append({'a': [{'b': None}, {'a': [None]}]})
    docs.append([[None, None], {'a': None}])
    docs.append({'a': [[None, None], [None]]})
    docs.append([{'a': [None, None, None]}, [None]])
    docs.append({'a': {'b': [None, None]}, 'b': [None]})
    docs.append([None, [None, [None, None]]])
    return docs


def test_cross_validation_against_reference():
    documents = _document_pool()
    for selector in CROSS_CHECK_SELECTORS:
        for allowed in CROSS_CHECK_SELECTORS:
            result = check_relation(selector, [allowed])
            assert result.relation is not Relation.UNKNOWN, result.reason
            if result.relation is Relation.CONTAINED:
                for document in documents:
                    violation = _reference_violation(selector, [allowed], document)
                    assert violation is None, (
                        '%s vs %s: checker says contained but %r escapes on %r'
                        % (selector, allowed, violation, document)
                    )
            else:
                counterexample = result.counterexample
                assert counterexample is not None and counterexample.verified
                violation = _reference_violation(
                    selector, [allowed], counterexample.document
                )
                assert violation is not None


def test_cross_validation_allowed_combinations():
    documents = _document_pool()
    allowed_sets = [
        ['$.a', '$.b'],
        ['$.a[0]', '$.a[1]'],
        ['$..a', '$.b[0]'],
        ['$.a[*]', '$.b'],
        ['$.a[0:2]', '$.a[3]'],
        ['$.*', '$[*]'],
    ]
    for selector in CROSS_CHECK_SELECTORS:
        for allowed in allowed_sets:
            result = check_relation(selector, allowed)
            assert result.relation is not Relation.UNKNOWN, result.reason
            if result.relation is Relation.CONTAINED:
                for document in documents:
                    violation = _reference_violation(selector, allowed, document)
                    assert violation is None, (
                        '%s vs %s: checker says contained but %r escapes on %r'
                        % (selector, allowed, violation, document)
                    )
            else:
                assert result.counterexample is not None
                assert result.counterexample.verified


# ---------------------------------------------------------------------------
# Randomized agreement with the reference matcher (hypothesis)
# ---------------------------------------------------------------------------

hypothesis = pytest.importorskip('hypothesis')
from hypothesis import given, settings, strategies as st  # noqa: E402


def _json_documents():
    scalars = st.none()
    containers = st.recursive(
        scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=3),
            st.dictionaries(st.sampled_from(['a', 'b', 'abc']), children, max_size=3),
        ),
        max_leaves=8,
    )
    return containers


# A curated spread of (selector, allowed) pairs covering fields, wildcards,
# descendants, indices, slices and unions.
RANDOM_CHECK_PAIRS = [
    (selector, allowed)
    for selector in ['$.a', '$.a.b', '$.*', '$.a[0]', '$.a[1]', '$.a[*]',
                     '$.a[0:2]', '$.a[1:]', '$.a[::2]', '$..a', '$..*', '$..[*]',
                     '$.a[0].b', '$.a[*].b', '$', '$.a[0,2]']
    for allowed in ['$.a', '$.*', '$.a[*]', '$.a[0]', '$.a[0:2]', '$.a[1:]',
                    '$..a', '$..b', '$..*', '$..[*]', '$.a.b', '$..[0]']
]


@given(document=_json_documents())
@settings(derandomize=True, max_examples=40, deadline=None)
def test_random_documents_never_contradict_contained(document):
    for selector, allowed in RANDOM_CHECK_PAIRS:
        result = check_relation(selector, [allowed])
        if result.relation is Relation.CONTAINED:
            violation = _reference_violation(selector, [allowed], document)
            assert violation is None, (
                '%s vs %s: contained but %r escapes on %r'
                % (selector, allowed, violation, document)
            )
        elif result.relation is Relation.NOT_CONTAINED:
            counterexample = result.counterexample
            assert counterexample is not None and counterexample.verified
        else:
            raise AssertionError('unexpected relation: %s' % result.relation)
