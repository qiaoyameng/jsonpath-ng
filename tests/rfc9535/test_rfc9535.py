import pytest

from jsonpath_ng.exceptions import JSONPathError
from jsonpath_ng.ext import EXTENDED, RFC9535, parse as parse_ext
from jsonpath_ng.rfc9535 import Nothing, parse as parse_rfc


DATA = {
    "store": {
        "book": [
            {"category": "reference", "author": "Nigel Rees", "price": 8.95},
            {"category": "fiction", "author": "Evelyn Waugh", "price": 12.99},
            {"category": "fiction", "author": "Herman Melville", "price": 8.99},
            {"category": "fiction", "author": "J. R. R. Tolkien", "price": 22.99},
        ],
        "bicycle": {"color": "red", "price": 19.95},
    },
    "values": [
        {"n": 1, "flag": True, "label": "one"},
        {"n": 1.0, "flag": False, "label": None},
        {"n": True, "label": "bool"},
        {"n": "1", "label": "string"},
        {},
    ],
    "matrix": [[1, 2], [3]],
    "store.book": "dotted",
    "a'b\n": "escaped",
    "café": "naïve",
    "1abc": 1,
    "-x": 2,
}


@pytest.mark.parametrize(
    ("expression", "expected_values", "expected_paths"),
    [
        (
            '$["store"]["book"][0].author',
            ["Nigel Rees"],
            ["$['store']['book'][0]['author']"],
        ),
        ('$["store"]["book"][-1].price', [22.99], ["$['store']['book'][3]['price']"]),
        ('$["store"]["book"][4].price', [], []),
        ('$["store"]["book"][1:3]', DATA["store"]["book"][1:3], [
            "$['store']['book'][1]",
            "$['store']['book'][2]",
        ]),
        ('$["store"]["book"][-2:]', DATA["store"]["book"][-2:], [
            "$['store']['book'][2]",
            "$['store']['book'][3]",
        ]),
        ('$["store"]["book"][::-1]', list(reversed(DATA["store"]["book"])), [
            "$['store']['book'][3]",
            "$['store']['book'][2]",
            "$['store']['book'][1]",
            "$['store']['book'][0]",
        ]),
        ('$["store"]["book"][0:3:0]', [], []),
        ('$["store"]["bicycle"].price', [19.95], ["$['store']['bicycle']['price']"]),
        (
            "$..price",
            [8.95, 12.99, 8.99, 22.99, 19.95],
            [
                "$['store']['book'][0]['price']",
                "$['store']['book'][1]['price']",
                "$['store']['book'][2]['price']",
                "$['store']['book'][3]['price']",
                "$['store']['bicycle']['price']",
            ],
        ),
        (
            '$["store"]["book"][0]["author","author"]',
            ["Nigel Rees", "Nigel Rees"],
            [
                "$['store']['book'][0]['author']",
                "$['store']['book'][0]['author']",
            ],
        ),
        ('$["matrix"][0,0][1]', [2, 2], [
            "$['matrix'][0][1]",
            "$['matrix'][0][1]",
        ]),
        ('$["store.book"]', ["dotted"], ["$['store.book']"]),
        ('$["a\'b\\n"]', ["escaped"], ["$['a\\'b\\n']"]),
        ("$.store.book", ["dotted"], ["$['store.book']"]),
        ("$.café", ["naïve"], ["$['café']"]),
        ("$.1abc", [1], ["$['1abc']"]),
        ("$.-x", [2], ["$['-x']"]),
        ('$["missing"][*]', [], []),
        ('$["store"]["bicycle"][*]', ["red", 19.95], [
            "$['store']['bicycle']['color']",
            "$['store']['bicycle']['price']",
        ]),
        ("$..book[?@.price]", DATA["store"]["book"], [
            f"$['store']['book'][{i}]" for i in range(4)
        ]),
        ('$["matrix"][*][*]', [1, 2, 3], [
            "$['matrix'][0][0]",
            "$['matrix'][0][1]",
            "$['matrix'][1][0]",
        ]),
    ],
)
def test_standard_selectors(expression, expected_values, expected_paths):
    matches = parse_rfc(expression).find(DATA)
    assert [match.value for match in matches] == expected_values
    assert [str(match.full_path) for match in matches] == expected_paths


@pytest.mark.parametrize(
    ("expression", "expected_indices"),
    [
        ("$.values[?@.n == 1]", [0, 1]),
        ("$.values[?@.n == 1.0]", [0, 1]),
        ("$.values[?@.n == true]", [2]),
        ('$.values[?@.label == "one"]', [0]),
        ("$.values[?@.label == null]", [1]),
        ("$.values[?@.missing == null]", []),
        ("$.values[?@.missing != null]", [0, 1, 2, 3, 4]),
        ("$.values[?@.n < 1.1]", [0, 1]),
        ("$.values[?@.flag == true]", [0]),
        ("$.values[?@.flag == false]", [1]),
        ("$.values[?@.n]", [0, 1, 2, 3]),
        ("$.values[?!@.n]", [4]),
        ('$.values[?@.n == 1 && @.label == "one"]', [0]),
        ("$.values[?@.n == 1.0 || @.flag == true]", [0, 1]),
        ("$.values[?(@.n == 1 || @.n == 1.0) && @.label]", [0, 1]),
        ("$..book[?@.price < 10].price", [8.95, 8.99]),
    ],
)
def test_standard_filter_expressions(expression, expected_indices):
    if expression.startswith("$.values"):
        expected = [DATA["values"][index] for index in expected_indices]
    else:
        expected = [8.95, 8.99]
    assert [match.value for match in parse_rfc(expression).find(DATA)] == expected


def test_absolute_filter_query_uses_root():
    data = {"selected": 1, "items": [{"id": 1}, {"id": 2}]}
    matches = parse_rfc("$.items[?@.id == $.selected]").find(data)
    assert [match.value for match in matches] == [{"id": 1}]


@pytest.mark.parametrize(
    "expression",
    [
        "store.book[0]",
        "store.book[*]",
        "$.store.book[?(@.price = 10)]",
        "$.store.book[?(@.price =~ /10/)]",
        "$.store.book[?(@.price + 1)]",
        "$.store.book[?(@.price & @.author)]",
        "$.values[?@[*] == 1]",
        "$.values[?1 == @[*]]",
        "$.values[?@..n == 1]",
        "$.store.book['author']",
        "$.store.book[01]",
        "$.store.book[1.0]",
        "$.store.book[1e2]",
        '$["a"].["b"]',
        "$.\"a\"",
        '$[?@.a = 1]',
        "$[?true]",
        "$[?false]",
        "$[?null]",
        "$[?42]",
        '$[?"x"]',
        "$[?!(@.n)]",
    ],
)
def test_invalid_standard_expressions_are_rejected(expression):
    with pytest.raises(JSONPathError):
        parse_rfc(expression)


def test_explicit_mode_keeps_extended_entry_separate():
    standard = parse_ext("$.values[?@.n == 1]", mode=RFC9535)
    assert [match.value for match in standard.find(DATA)] == DATA["values"][:2]

    extended = parse_ext("$.values[?@.n == 1]", mode=EXTENDED)
    assert [match.value for match in extended.find(DATA)] == [
        DATA["values"][0],
        DATA["values"][1],
        DATA["values"][2],
        DATA["values"][3],
    ]


def test_nothing_is_distinct_from_null_and_nodelist():
    assert Nothing is not None
    assert bool(Nothing) is True
    assert parse_rfc("$.missing").find(DATA) == []
    assert parse_rfc("$.values[1].label").find(DATA)[0].value is None
