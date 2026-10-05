import pytest

import jsonpath_ng
from jsonpath_ng import parse
from jsonpath_ng.jsonpath import Child, Fields, Index, Root
from jsonpath_ng.relation import (
    Relation,
    RelationResult,
    check_containment,
    check_relation,
    is_covered,
)


def check(candidate, allowed):
    return check_relation(candidate, allowed)


def assert_not_contained_replays(candidate, allowed, result):
    """A NOT_CONTAINED verdict must come with a find-replayable witness."""
    assert result.relation is Relation.NOT_CONTAINED
    assert result.witness_path is not None
    document = result.witness_document

    candidate_exprs = [parse(candidate)] if isinstance(candidate, str) else [parse(c) for c in candidate]
    allowed_exprs = [parse(a) for a in allowed]

    def reaches(expr):
        try:
            return any(d.full_path == result.witness_path for d in expr.find(document))
        except Exception:
            return False

    assert any(reaches(expr) for expr in candidate_exprs)
    assert not any(reaches(expr) for expr in allowed_exprs)


@pytest.mark.parametrize(
    "candidate, allowed",
    (
        # Identity and simple field nesting.
        ("$", ["$"]),
        ("$.a", ["$.a"]),
        ("$.a.b", ["$.a.b"]),
        ("$.a", ["$.*"]),
        ("$.a.b", ["$.a.*"]),
        ("$.a.b.c", ["$.a.*.c"]),
        ("$.a.b.c", ["$.*.*.c"]),
        ("$.'a b'", ["$.*"]),
        ("`this`.a", ["$.a"]),
        # Wildcards on both sides.
        ("$.a.*", ["$.a.*"]),
        ("$.*.*", ["$.*.*"]),
        # Indices and slices, including array-length boundaries.
        ("$[0]", ["$[*]"]),
        ("$[0]", ["$[0]"]),
        ("$[0,2]", ["$[*]"]),
        ("$[0,2]", ["$[0]", "$[2]"]),
        ("$.a[0]", ["$.a[*]"]),
        ("$.a[0:3]", ["$.a[*]"]),
        ("$.a[1:3]", ["$.a[0:5]"]),
        ("$.a[1:3]", ["$.a[1]", "$.a[2]"]),
        ("$.a[0:10:2]", ["$.a[::2]"]),
        ("$.a[::2]", ["$.a[*]"]),
        ("$.a[:]", ["$.a[*]"]),
        ("$[1:]", ["$[*]"]),
        ("$[3:9:3]", ["$[0:10]"]),
        # Empty selectors match nothing, hence are covered by anything.
        ("$.a[5:2]", []),
        ("$.a[2:2]", ["$.x"]),
        ("$.a[2:2]", ["$.a[?(@.b)]"]),  # even by the empty set of analyzable selectors
        # Descendants.
        ("$..b", ["$..b"]),
        ("$.a..b", ["$..b"]),
        ("$.a..b", ["$.a..b"]),
        ("$..b.c", ["$..b.c"]),
        ("$..*", ["$..*"]),
        ("$..[*]", ["$..[*]"]),
        ("$..a..b", ["$..a..b"]),
        ("$.a.b", ["$..b"]),
        # Unions on either side.
        ("($.a)|($.b)", ["$.a", "$.b"]),
        ("$.a", ["($.a)|($.b)"]),
        ("($.a[0])|($.a[1])", ["$.a[0:2]"]),
        ("$.a[0,1]", ["($.a[0])|($.a[1])"]),
        # Multiple allowed selectors combine as a union.
        ("$.a", ["$.b", "$.a", "$.c"]),
        ("$[3]", ["$[0:2]", "$[3]", "$[4:]"]),
        # Candidate lists are treated as a union.
        (["$.a", "$.b"], ["$.a", "$.b"]),
    ),
)
def test_contained(candidate, allowed):
    result = check(candidate, allowed)
    assert result.relation is Relation.CONTAINED
    assert result.contained
    assert bool(result)
    assert result.witness_path is None
    assert is_covered(candidate, allowed)
    assert check_containment(candidate, allowed).relation is Relation.CONTAINED


@pytest.mark.parametrize(
    "candidate, allowed",
    (
        # A textual prefix is not coverage: $.a does not cover $.ab.
        ("$.ab", ["$.a"]),
        ("$.a", ["$.ab"]),
        # Parent/child mismatch in both directions.
        ("$.a.b", ["$.a"]),
        ("$.a", ["$.a.b"]),
        ("$.a.b.c", ["$.a.*"]),
        ("$.a.*", ["$.a.b"]),
        ("$.a.*.d", ["$.a.b.*"]),
        ("$.*", ["$.a"]),
        ("$", ["$.a"]),
        ("$", ["$.*"]),
        ("$.*", ["$[*]"]),
        ("$[*]", ["$.*"]),
        # Array-length boundaries.
        ("$[5]", ["$[0:5]"]),
        ("$[*]", ["$[0:5]"]),
        ("$[0:5]", ["$[1:5]"]),
        ("$[1:]", ["$[0:100]"]),
        ("$[::2]", ["$[0:10:2]"]),
        ("$[1::2]", ["$[::2]"]),
        ("$[0:6:2]", ["$[0:6:3]"]),
        ("$[0]", ["$[1:]"]),
        ("$.a[0]", ["$.a[1]"]),
        # Descendants can always go deeper or sideways.
        ("$..b", ["$.a..b"]),
        ("$..*", ["$..a"]),
        ("$..b", ["$.b"]),
        ("$.a..b", ["$.a.b"]),
        ("$..b.c", ["$..b.d"]),
        ("$..[*]", ["$[0]"]),
        # Unions are only covered when every branch is.
        ("($.a)|($.c)", ["$.a", "$.b"]),
        ("($.a[0])|($.a[5])", ["$.a[0:5]"]),
        # Empty allowed set covers nothing that can match.
        ("$.a", []),
        ("$", []),
        ("$[*]", []),
        ("$..a", []),
    ),
)
def test_not_contained(candidate, allowed):
    result = check(candidate, allowed)
    assert_not_contained_replays(candidate, allowed, result)
    assert not result.contained
    assert not result
    assert not is_covered(candidate, allowed)


@pytest.mark.parametrize(
    "candidate, allowed",
    (
        # Filters and arbitrary predicates are out of scope.
        ("$.a[?(@.x > 1)]", ["$.a"]),
        ("$.a[?(@.x > 1)]", ["$..*"]),
        ("($.a) where ($.b)", ["$"]),
        # Intersections.
        ("($.a)&($.b)", ["$"]),
        # Named operators beyond `this`.
        ("$.a.`parent`", ["$"]),
        # Negative indices and non-positive or negative slice steps.
        ("$.a[-1]", ["$.a[*]"]),
        ("$.a[::-1]", ["$"]),
        ("$.a[::0]", ["$"]),
        ("$.a[-2:-1]", ["$"]),
        # `$` away from the start of a selector.
        ("$.a..$", ["$"]),
        # Unparseable input.
        ("not a path", ["$.a"]),
        ("", ["$.a"]),
        # An allowed selector that cannot be analyzed makes positive
        # coverage unprovable unless the supported ones already cover it.
        ("$.a", ["$.a[?(@.x)]"]),
        ("$.b", ["$.a", "$.b[?(@.x)]"]),
    ),
)
def test_unknown(candidate, allowed):
    result = check(candidate, allowed)
    assert result.relation is Relation.UNKNOWN
    assert not result.contained
    assert not result
    assert result.witness_path is None
    assert not is_covered(candidate, allowed)


def test_unknown_allowed_still_contained_via_supported():
    # Unsupported allowed selectors do not weaken a proof that only
    # relies on the supported ones.
    result = check("$.a", ["$.a", "$.b[?(@.x)]"])
    assert result.relation is Relation.CONTAINED


def test_witness_is_minimal_and_small():
    result = check("$[*]", ["$[0:3]"])
    assert result.relation is Relation.NOT_CONTAINED
    assert result.witness_document == [None, None, None, None]
    assert [d.value for d in parse("$[*]").find(result.witness_document)] == [None] * 4


def test_result_api():
    result = check("$.a", ["$.a"])
    assert isinstance(result, RelationResult)
    assert result.relation == Relation.CONTAINED
    assert result.reason

    result = check("$.a", ["$.b"])
    assert result.relation == Relation.NOT_CONTAINED
    assert result.witness_document == {"a": None}
    assert str(result.witness_path) == "a"
    # The witness path itself can be evaluated with find.
    assert [d.value for d in result.witness_path.find(result.witness_document)] == [None]


def test_accepts_parsed_expressions():
    assert check(parse("$.a"), [parse("$.a")]).relation is Relation.CONTAINED
    assert check(parse("$.a"), "$.a").relation is Relation.CONTAINED
    assert check("$.a", parse("$.a")).relation is Relation.CONTAINED


def test_root_only_selector_coverage():
    assert check("$", ["$"]).relation is Relation.CONTAINED
    result = check("$", ["$.*"])
    assert result.relation is Relation.NOT_CONTAINED
    assert result.witness_path == parse("$")


def test_descendant_witness_replays():
    result = check("$..b", ["$.a..b"])
    assert result.relation is Relation.NOT_CONTAINED
    matches = parse("$..b").find(result.witness_document)
    assert any(m.full_path == result.witness_path for m in matches)


def test_deeply_nested_descendants():
    assert check("$.a..b..c", ["$..c"]).relation is Relation.CONTAINED
    assert check("$..c", ["$.a..b..c"]).relation is Relation.NOT_CONTAINED


def test_package_level_exports():
    assert jsonpath_ng.check_relation is check_relation
    assert jsonpath_ng.check_containment is check_containment
    assert jsonpath_ng.is_covered is is_covered
    assert jsonpath_ng.Relation is Relation


# ---------------------------------------------------------------------------
# Property-based cross-validation against find()
# ---------------------------------------------------------------------------

import random

from hypothesis import given, settings, strategies as st

_FIELD_NAMES = st.sampled_from(["a", "b", "c", "ab"])

_STEPS = st.one_of(
    _FIELD_NAMES.map(lambda name: "." + name),
    st.just(".*"),
    st.integers(0, 4).map(lambda i: "[%d]" % i),
    st.builds(
        lambda s, e, t: "[{}{}{}{}]".format(
            "" if s is None else s,
            ":",
            "" if e is None else e,
            "" if t is None else ":%d" % t,
        ),
        st.one_of(st.none(), st.integers(0, 3)),
        st.one_of(st.none(), st.integers(0, 6)),
        st.one_of(st.none(), st.integers(1, 2)),
    ),
    _FIELD_NAMES.map(lambda name: ".." + name),
)

_SELECTORS = st.lists(_STEPS, max_size=3).map(lambda steps: "$" + "".join(steps))


def _canonical_document(path):
    """The smallest document providing every step of a concrete full path."""
    steps = []

    def walk(expr):
        if isinstance(expr, Child):
            walk(expr.left)
            walk(expr.right)
        elif isinstance(expr, Fields):
            steps.append(expr.fields[0])
        elif isinstance(expr, Index):
            steps.append(expr.indices[0])
        elif not isinstance(expr, Root):
            raise AssertionError("unexpected path element %r" % (expr,))

    walk(path)
    document = None
    for step in reversed(steps):
        document = {step: document} if isinstance(step, str) else [None] * step + [document]
    return document


def _random_document(rng, depth=0):
    if depth > 3 or rng.random() < 0.3:
        return rng.choice([None, 0, 1, True])
    if rng.random() < 0.65:
        return {name: _random_document(rng, depth + 1) for name in rng.sample(["a", "b", "c", "ab"], rng.randint(0, 3))}
    return [_random_document(rng, depth + 1) for _ in range(rng.randint(0, 4))]


@given(
    _SELECTORS,
    st.lists(_SELECTORS, max_size=3),
    st.integers(min_value=0),
)
@settings(max_examples=100)
def test_verdicts_against_find(candidate, allowed, seed):
    result = check_relation(candidate, allowed)
    candidate_expr = parse(candidate)
    allowed_exprs = [parse(a) for a in allowed]

    if result.relation is Relation.NOT_CONTAINED:
        assert_not_contained_replays(candidate, allowed, result)
    elif result.relation is Relation.CONTAINED:
        # Every path the candidate can produce must lie in the allowed
        # path set: some allowed selector must reproduce it on the
        # canonical document for that path.
        rng = random.Random(seed)
        seen = {}
        for _ in range(4):
            document = _random_document(rng)
            try:
                found = candidate_expr.find(document)
            except Exception:
                continue
            for datum in found:
                seen[str(datum.full_path)] = datum.full_path
        for path in seen.values():
            document = _canonical_document(path)
            covered = False
            for allowed_expr in allowed_exprs:
                try:
                    if any(d.full_path == path for d in allowed_expr.find(document)):
                        covered = True
                        break
                except Exception:
                    continue
            assert covered, "%r reached %s outside %r" % (candidate, path, allowed)
