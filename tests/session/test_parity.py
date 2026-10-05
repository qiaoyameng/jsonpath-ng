"""Property-style parity checks: incremental result == full find.

These scenarios are the ones explicitly requested: stepwise array insert and
delete, filter field mutation, compared against a complete ``find``.
"""

import copy
import random

import pytest
from jsonpath_ng.ext import parse as ext_parse

from jsonpath_ng.session import QuerySession
from jsonpath_ng.session import engine
from jsonpath_ng.session.patch import apply_batch


EXPRESSIONS = [
    "$.items[*].v",
    "$.items[?@.v >= 5].v",
    "$.items[?@.flag].v",
    "$.items[0].v",
    "$.items[0:2].v",
    "$.items[1:3].v",
    "$.items[::2].v",
    "$..v",
    "$.group.items[*].v",
]


def full_find(doc, expression):
    result = {}
    for datum in ext_parse(expression).find(doc):
        tokens = []
        engine._collect(datum.full_path, tokens)
        result[tuple(tokens)] = datum.value
    return result


def random_batch(rng, doc):
    ops = []
    work = copy.deepcopy(doc)
    for _ in range(rng.randint(1, 4)):
        n = len(work["items"])
        g = len(work["group"]["items"])
        choice = rng.random()
        if n > 0 and choice < 0.2:
            i = rng.randrange(n)
            ops.append({"op": "remove", "path": f"/items/{i}"})
            del work["items"][i]
        elif choice < 0.4:
            i = rng.randint(0, n)
            value = {"v": rng.randint(0, 9), "flag": rng.choice([True, False])}
            ops.append({"op": "add", "path": f"/items/{i}", "value": value})
            work["items"].insert(i, value)
        elif n > 0 and choice < 0.6:
            i = rng.randrange(n)
            field = rng.choice(["v", "flag"])
            if field == "v":
                value = rng.randint(0, 9)
            else:
                value = rng.choice([True, False])
            ops.append(
                {"op": "replace", "path": f"/items/{i}/{field}", "value": value}
            )
            work["items"][i][field] = value
        elif n > 1:
            i, j = rng.sample(range(n), 2)
            ops.append({"op": "move", "from": f"/items/{i}", "path": f"/items/{j}"})
            moved = work["items"].pop(i)
            work["items"].insert(j - (1 if j > i else 0), moved)
        elif n > 0 and g > 0:
            i = rng.randrange(n)
            ops.append(
                {"op": "move", "from": f"/items/{i}", "path": "/group/items/-"}
            )
            work["group"]["items"].append(work["items"].pop(i))
        elif g > 1:
            i, j = rng.sample(range(g), 2)
            ops.append(
                {"op": "move",
                 "from": f"/group/items/{i}",
                 "path": f"/group/items/{j}"}
            )
            moved = work["group"]["items"].pop(i)
            work["group"]["items"].insert(j - (1 if j > i else 0), moved)
        else:
            value = {"v": 0, "flag": False}
            ops.append({"op": "add", "path": "/items/-", "value": value})
            work["items"].append(value)
    return ops


def random_doc(rng):
    return {
        "items": [
            {"v": rng.randint(0, 9), "flag": rng.choice([True, False])}
            for _ in range(rng.randint(0, 6))
        ],
        "group": {
            "items": [
                {"v": rng.randint(0, 9), "flag": rng.choice([True, False])}
                for _ in range(rng.randint(0, 3))
            ]
        },
    }


@pytest.mark.parametrize("seed", [1, 7, 42, 2025, 999, 31337, 55555])
def test_incremental_matches_full_find(seed):
    rng = random.Random(seed)
    for _ in range(25):
        base = random_doc(rng)
        session = QuerySession(copy.deepcopy(base), *EXPRESSIONS)
        doc = copy.deepcopy(base)
        for _ in range(5):
            batch = random_batch(rng, doc)
            reference = copy.deepcopy(doc)
            apply_batch(reference, copy.deepcopy(batch))
            session.apply_patches(copy.deepcopy(batch))
            doc = reference
            for expression in EXPRESSIONS:
                expected = full_find(doc, expression)
                query = next(
                    q for q in session.queries if q.expression == expression
                )
                assert query.matches == expected, (
                    expression,
                    base,
                    batch,
                    query.matches,
                    expected,
                )


def test_stepwise_insert_and_delete_scenario():
    """The explicit showcase: gradual inserts/deletes vs complete find."""
    doc = {"items": [{"v": 1}, {"v": 9}, {"v": 2}, {"v": 8}, {"v": 3}]}
    session = QuerySession(copy.deepcopy(doc), "$.items[?@.v >= 5].v", "$.items[*].v")
    operations = [
        {"op": "add", "path": "/items/0", "value": {"v": 7}},
        {"op": "remove", "path": "/items/3"},
        {"op": "replace", "path": "/items/0/v", "value": 1},
        {"op": "add", "path": "/items/-", "value": {"v": 10}},
    ]
    for op in operations:
        apply_batch(doc, [op])
        session.apply_patches([copy.deepcopy(op)])
        for expression in ("$.items[?@.v >= 5].v", "$.items[*].v"):
            query = next(q for q in session.queries if q.expression == expression)
            assert query.matches == full_find(doc, expression)
