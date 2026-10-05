import pytest

from jsonpath_ng import parse_rfc9535
from jsonpath_ng.exceptions import JsonPathParserError
from jsonpath_ng.ext import parse as parse_extended


DATA = {
    "store": {
        "book": [
            {"category": "reference", "title": "Sayings", "price": 8.95, "available": True},
            {"category": "fiction", "title": "Sword", "price": 12.99, "available": False},
            {"category": "fiction", "title": "Moby", "isbn": "0-553", "price": 8.99, "available": None},
        ],
        "bicycle": {"color": "red", "price": 19.95, "count": 1},
    },
    "true-name": {"true": 1, "false": 0},
}


def matches(expression, data=DATA):
    return [
        (match.value, str(match.full_path))
        for match in parse_rfc9535(expression).find(data)
    ]


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("$.store.book[0].title", [("Sayings", "$['store']['book'][0]['title']")]),
        ("$['store']['book'][-1]['title']", [("Moby", "$['store']['book'][2]['title']")]),
        (
            "$.store.book[0,2].category",
            [
                ("reference", "$['store']['book'][0]['category']"),
                ("fiction", "$['store']['book'][2]['category']"),
            ],
        ),
        (
            "$.store.book[0:3:2].price",
            [
                (8.95, "$['store']['book'][0]['price']"),
                (8.99, "$['store']['book'][2]['price']"),
            ],
        ),
        (
            "$.store.bicycle[*]",
            [
                ("red", "$['store']['bicycle']['color']"),
                (19.95, "$['store']['bicycle']['price']"),
                (1, "$['store']['bicycle']['count']"),
            ],
        ),
        (
            "$.store..price",
            [
                (8.95, "$['store']['book'][0]['price']"),
                (12.99, "$['store']['book'][1]['price']"),
                (8.99, "$['store']['book'][2]['price']"),
                (19.95, "$['store']['bicycle']['price']"),
            ],
        ),
        ("$..[?@.isbn].isbn", [("0-553", "$['store']['book'][2]['isbn']")]),
        (
            "$.store.book[?@.price < 10].title",
            [
                ("Sayings", "$['store']['book'][0]['title']"),
                ("Moby", "$['store']['book'][2]['title']"),
            ],
        ),
        (
            "$.store.book[?@.available == true].title",
            [("Sayings", "$['store']['book'][0]['title']")],
        ),
        (
            "$.store.book[?@.available == false].title",
            [("Sword", "$['store']['book'][1]['title']")],
        ),
        (
            "$.store.book[?@.available == null].title",
            [("Moby", "$['store']['book'][2]['title']")],
        ),
        (
            "$.store.book[?@.available].title",
            [
                ("Sayings", "$['store']['book'][0]['title']"),
                ("Sword", "$['store']['book'][1]['title']"),
                ("Moby", "$['store']['book'][2]['title']"),
            ],
        ),
        ("$.store.book[?@.isbn != null].title", [("Moby", "$['store']['book'][2]['title']")]),
        (
            "$.store.book[?@.category == 'fiction' && @.price > 10].title",
            [("Sword", "$['store']['book'][1]['title']")],
        ),
        (
            "$.store.book[?@.price < 9 || @.isbn].title",
            [
                ("Sayings", "$['store']['book'][0]['title']"),
                ("Moby", "$['store']['book'][2]['title']"),
            ],
        ),
        (
            "$.store.book[?!@.isbn].title",
            [
                ("Sayings", "$['store']['book'][0]['title']"),
                ("Sword", "$['store']['book'][1]['title']"),
            ],
        ),
        ("$.missing", []),
        ("$.store.book[99].title", []),
        ("$.store.bicycle[0]", []),
        ("$['true-name']['true']", [(1, "$['true-name']['true']")]),
    ),
)
def test_rfc9535_positive_examples(expression, expected):
    assert matches(expression) == expected


def test_filter_distinguishes_boolean_from_number():
    data = {
        "items": [
            {"v": True, "name": "bool"},
            {"v": 1, "name": "number"},
            {"v": False, "name": "false-bool"},
            {"v": 0, "name": "zero"},
        ]
    }
    assert matches("$..[?@.v == true].name", data) == [
        ("bool", "$['items'][0]['name']")
    ]
    assert matches("$..[?@.v == 1].name", data) == [
        ("number", "$['items'][1]['name']")
    ]
    assert matches("$..[?@.v == false].name", data) == [
        ("false-bool", "$['items'][2]['name']")
    ]
    assert matches("$..[?@.v == 0].name", data) == [
        ("zero", "$['items'][3]['name']")
    ]


def test_nothing_is_distinct_from_null_and_false():
    data = {
        "items": [
            {"name": "missing"},
            {"name": "null", "v": None},
            {"name": "false", "v": False},
        ]
    }
    assert matches("$..[?@.v == null].name", data) == [
        ("null", "$['items'][1]['name']")
    ]
    assert matches("$..[?@.v != null].name", data) == [
        ("false", "$['items'][2]['name']")
    ]
    assert matches("$..[?@.v].name", data) == [
        ("null", "$['items'][1]['name']"),
        ("false", "$['items'][2]['name']"),
    ]


def test_duplicate_selectors_keep_duplicate_nodes_in_order():
    data = {"a": 1, "b": 2, "items": ["x", "y"]}
    assert matches("$['a','b','a']", data) == [
        (1, "$['a']"),
        (2, "$['b']"),
        (1, "$['a']"),
    ]
    assert matches("$.items[1,0,1]", data) == [
        ("y", "$['items'][1]"),
        ("x", "$['items'][0]"),
        ("y", "$['items'][1]"),
    ]


def test_descendant_order_is_document_order():
    data = {"a": [{"x": 1}, {"x": 2}], "x": 3, "b": {"x": 4}}
    assert matches("$..x", data) == [
        (3, "$['x']"),
        (1, "$['a'][0]['x']"),
        (2, "$['a'][1]['x']"),
        (4, "$['b']['x']"),
    ]


@pytest.mark.parametrize(
    "expression",
    (
        "store.book[0].title",
        "$.store.book[*] where @.isbn",
        "$.store.book[?@.price > 1 & @.isbn]",
        "$.store.book[?@.price =~ /x/]",
        "$.store.book[?@[*] == 'x']",
        "$.store.book[?@.author[*] == 'x']",
        "$.store.book[?(@.author) == 'x']",
        "$.store.book[0::0]",
        "$.store.book[01]",
        "$.store.book[?null]",
        "$.store.book[?@.price > 01]",
        "$.store.book[?@.price > 1.]",
        "$.store.book[?@.price > 1.2.3]",
    ),
)
def test_rfc9535_rejects_non_standard_or_invalid_forms(expression):
    with pytest.raises(JsonPathParserError):
        parse_rfc9535(expression)


def test_extended_entry_retains_legacy_semantics():
    from jsonpath_ng.ext import parse as parse_extended

    data = {"items": [{"v": True}, {"v": 1}]}
    legacy = parse_extended("$.items[?(@.v==1)]")
    assert [match.value for match in legacy.find(data)] == [
        {"v": True},
        {"v": 1},
    ]
    standard = parse_rfc9535("$.items[?@.v == 1]")
    assert [match.value for match in standard.find(data)] == [{"v": 1}]


def test_root_query_in_filter_evaluates_against_root():
    data = {"limit": 10, "items": [{"price": 8}, {"price": 12}]}
    assert matches("$.items[?@.price < $.limit]", data) == [
        ({"price": 8}, "$['items'][0]")
    ]


def test_normalized_paths_handle_special_names_and_negative_indices():
    data = {"a\nb": {"x": 1}, "items": ["last"]}
    assert matches("$['a\\nb'].x", data) == [
        (1, "$['a\\nb']['x']")
    ]
    assert matches("$.items[-1]", data) == [
        ("last", "$['items'][0]")
    ]
