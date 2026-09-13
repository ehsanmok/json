# Tests for JSONPath (RFC 9535).
#
# The suite is organised around the specification rather than around
# the implementation: the worked examples of section 1.5 and the
# per-selector examples of section 2 are transcribed as written, so a
# failure names the paragraph it disagrees with. After those come the
# regressions, one per defect the pre-RFC implementation carried.

from std.testing import assert_equal, assert_true, TestSuite

from json import loads, jsonpath_query, jsonpath_one
from json.jsonpath import JSONPath
from json.value import Value


def _bookstore() raises -> Value:
    """The document of RFC 9535 section 1.5."""
    return loads(
        '{"store":{"book":['
        '{"category":"reference","author":"Nigel Rees",'
        '"title":"Sayings of the Century","price":8.95},'
        '{"category":"fiction","author":"Evelyn Waugh",'
        '"title":"Sword of Honour","price":12.99},'
        '{"category":"fiction","author":"Herman Melville",'
        '"title":"Moby Dick","isbn":"0-553-21311-3","price":8.99},'
        '{"category":"fiction","author":"J. R. R. Tolkien",'
        '"title":"The Lord of the Rings","isbn":"0-395-19395-8",'
        '"price":22.99}],'
        '"bicycle":{"color":"red","price":399}}}'
    )


def _filter_doc() raises -> Value:
    """The document of RFC 9535 section 2.3.5.3."""
    return loads(
        '{"a":[3,5,1,2,4,6,{"b":"j"},{"b":"k"},{"b":{}},{"b":"kilo"}],'
        '"o":{"p":1,"q":2,"r":3,"s":5,"t":{"u":6}},'
        '"e":"f"}'
    )


def _assert_raises(document: Value, path: String, why: String) raises:
    var caught = False
    try:
        _ = jsonpath_query(document, path)
    except:
        caught = True
    assert_true(caught, why + ": " + path)


def _strings(values: List[Value]) raises -> List[String]:
    var out = List[String]()
    for i in range(len(values)):
        out.append(values[i].string_value())
    return out^


def _ints(values: List[Value]) raises -> List[String]:
    """Numbers rendered as text, so a list can be compared in one go."""
    var out = List[String]()
    for i in range(len(values)):
        out.append(values[i].raw_json())
    return out^


def _joined(parts: List[String]) -> String:
    var out = String()
    for i in range(len(parts)):
        if i > 0:
            out += ","
        out += parts[i]
    return out^


# -- section 1.5, the worked examples ------------------------------


def test_rfc_1_5_store_book_authors() raises:
    var doc = _bookstore()
    var got = _strings(jsonpath_query(doc, "$.store.book[*].author"))
    assert_equal(len(got), 4)
    assert_equal(got[0], "Nigel Rees")
    assert_equal(got[3], "J. R. R. Tolkien")


def test_rfc_1_5_all_authors() raises:
    var doc = _bookstore()
    var got = _strings(jsonpath_query(doc, "$..author"))
    assert_equal(len(got), 4)
    assert_equal(got[2], "Herman Melville")


def test_rfc_1_5_store_wildcard() raises:
    """`$.store.*` selects the book array and the bicycle object."""
    var doc = _bookstore()
    var got = jsonpath_query(doc, "$.store.*")
    assert_equal(len(got), 2)
    assert_true(got[0].is_array())
    assert_true(got[1].is_object())


def test_rfc_1_5_store_descendant_price() raises:
    """Four book prices and the bicycle's."""
    var doc = _bookstore()
    assert_equal(len(jsonpath_query(doc, "$.store..price")), 5)


def test_rfc_1_5_book_index_two() raises:
    var doc = _bookstore()
    var got = jsonpath_query(doc, "$..book[2]")
    assert_equal(len(got), 1)
    assert_equal(got[0]["title"].string_value(), "Moby Dick")
    assert_equal(
        jsonpath_one(doc, "$..book[2].author").string_value(),
        "Herman Melville",
    )
    assert_equal(len(jsonpath_query(doc, "$..book[2].publisher")), 0)


def test_rfc_1_5_book_last() raises:
    var doc = _bookstore()
    var got = jsonpath_query(doc, "$..book[-1]")
    assert_equal(len(got), 1)
    assert_equal(got[0]["title"].string_value(), "The Lord of the Rings")


def test_rfc_1_5_book_first_two() raises:
    """`$..book[0,1]` and `$..book[:2]` name the same two books."""
    var doc = _bookstore()
    var union = jsonpath_query(doc, "$..book[0,1]")
    var slice = jsonpath_query(doc, "$..book[:2]")
    assert_equal(len(union), 2)
    assert_equal(len(slice), 2)
    assert_equal(union[0]["title"].string_value(), "Sayings of the Century")
    assert_equal(union[1]["title"].string_value(), "Sword of Honour")
    assert_equal(slice[1]["title"].string_value(), "Sword of Honour")


def test_rfc_1_5_book_with_isbn() raises:
    var doc = _bookstore()
    assert_equal(len(jsonpath_query(doc, "$..book[?@.isbn]")), 2)


def test_rfc_1_5_book_cheaper_than_ten() raises:
    var doc = _bookstore()
    var got = jsonpath_query(doc, "$..book[?@.price<10]")
    assert_equal(len(got), 2)
    assert_equal(got[0]["title"].string_value(), "Sayings of the Century")
    assert_equal(got[1]["title"].string_value(), "Moby Dick")


def test_rfc_1_5_all_descendants() raises:
    """`$..*` gives all 27 member values and array elements."""
    var doc = _bookstore()
    assert_equal(len(jsonpath_query(doc, "$..*")), 27)


# -- section 2.3.1, the name selector ------------------------------


def test_name_selector_examples() raises:
    var doc = loads('{"o": {"j j": {"k.k": 3}}, "\'": "@"}')
    assert_equal(jsonpath_one(doc, "$.o['j j']['k.k']").raw_json(), String("3"))
    assert_equal(jsonpath_one(doc, '$.o["j j"]["k.k"]').raw_json(), String("3"))
    assert_equal(jsonpath_one(doc, '$["\'"]').string_value(), "@")
    assert_equal(jsonpath_one(doc, "$['\\'']").string_value(), "@")


def test_name_selector_escapes() raises:
    """Defect 10: both quote flavours take the whole JSON escape set."""
    var doc = loads('{"a\\"b": 1, "a\'b": 2, "A": 3, "x\\\\y": 4, "t\\tb": 5}')
    assert_equal(jsonpath_one(doc, '$["a\\"b"]').raw_json(), String("1"))
    assert_equal(jsonpath_one(doc, "$['a\\'b']").raw_json(), String("2"))
    assert_equal(jsonpath_one(doc, '$["\\u0041"]').raw_json(), String("3"))
    assert_equal(jsonpath_one(doc, '$["x\\\\y"]').raw_json(), String("4"))
    assert_equal(jsonpath_one(doc, '$["t\\tb"]').raw_json(), String("5"))


def test_name_selector_surrogate_pair() raises:
    var doc = loads('{"\\ud834\\udd1e": 7}')
    assert_equal(
        jsonpath_one(doc, '$["\\ud834\\udd1e"]').raw_json(), String("7")
    )


def test_name_selector_unescaped_other_quote() raises:
    """A double quote stands for itself inside a single-quoted name."""
    var doc = loads('{"a\\"b": 1}')
    assert_equal(jsonpath_one(doc, "$['a\"b']").raw_json(), String("1"))


# -- section 2.3.2, the wildcard selector --------------------------


def test_wildcard_selector_examples() raises:
    var doc = loads('{"o": {"j": 1, "k": 2}, "a": [5, 3]}')
    assert_equal(len(jsonpath_query(doc, "$[*]")), 2)
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.o[*]"))), "1,2")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.o[*, *]"))), "1,2,1,2")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.a[*]"))), "5,3")


# -- section 2.3.3, the index selector -----------------------------


def test_index_selector_examples() raises:
    var doc = loads('["a","b"]')
    assert_equal(jsonpath_one(doc, "$[1]").string_value(), "b")
    assert_equal(jsonpath_one(doc, "$[-2]").string_value(), "a")
    assert_equal(len(jsonpath_query(doc, "$[2]")), 0)
    assert_equal(len(jsonpath_query(doc, "$[-3]")), 0)


def test_negative_index_resolved_per_node() raises:
    """Defect 4: each array resolves `-1` against its own length.

    The old code hoisted the normalised index out of the loop over
    nodes, so every array after the first reused the first one's
    answer and either missed or read the wrong element.
    """
    var doc = loads('{"x":[[1,2,3],[4],[5,6]]}')
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.x[*][-1]"))), "3,4,6")


# -- section 2.3.4, the slice selector -----------------------------


def test_slice_selector_examples() raises:
    var doc = loads('["a","b","c","d","e","f","g"]')
    assert_equal(_joined(_strings(jsonpath_query(doc, "$[1:3]"))), "b,c")
    assert_equal(_joined(_strings(jsonpath_query(doc, "$[5:]"))), "f,g")
    assert_equal(_joined(_strings(jsonpath_query(doc, "$[1:5:2]"))), "b,d")
    assert_equal(_joined(_strings(jsonpath_query(doc, "$[5:1:-2]"))), "f,d")
    assert_equal(
        _joined(_strings(jsonpath_query(doc, "$[::-1]"))), "g,f,e,d,c,b,a"
    )


def test_slice_negative_step_terminates() raises:
    """Defect 1: a negative step used to read out of bounds forever."""
    var doc = loads("[1,2,3,4,5]")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[::-1]"))), "5,4,3,2,1")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[::-2]"))), "5,3,1")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[-1:0:-1]"))), "5,4,3,2")
    assert_equal(len(jsonpath_query(doc, "$[1:1:-1]")), 0)


def test_slice_zero_step_is_empty() raises:
    """Defect 1: section 2.3.4.2.2 makes a zero step select nothing."""
    var doc = loads("[1,2,3,4,5]")
    assert_equal(len(jsonpath_query(doc, "$[::0]")), 0)
    assert_equal(len(jsonpath_query(doc, "$[1:4:0]")), 0)


def test_slice_open_ended_keeps_last_element() raises:
    """Defect 3: an absent bound is not the bound -1.

    Conflating the two made `end` come out as `count - 1`, so every
    open-ended slice quietly dropped the last element.
    """
    var doc = loads("[1,2,3,4,5]")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[1:]"))), "2,3,4,5")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[:]"))), "1,2,3,4,5")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[::2]"))), "1,3,5")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[:3]"))), "1,2,3")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[-2:]"))), "4,5")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[:-2]"))), "1,2,3")


def test_slice_clamps_out_of_range_bounds() raises:
    var doc = loads("[1,2,3]")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$[-99:99]"))), "1,2,3")
    assert_equal(len(jsonpath_query(doc, "$[99:]")), 0)
    assert_equal(len(jsonpath_query(doc, "$[3:1]")), 0)


def test_slice_on_empty_array() raises:
    var doc = loads("[]")
    assert_equal(len(jsonpath_query(doc, "$[:]")), 0)
    assert_equal(len(jsonpath_query(doc, "$[::-1]")), 0)


# -- section 2.5.1, unions -----------------------------------------


def test_union_of_indices() raises:
    """Defect 5: `$[0,2]` used to raise out of `atol`."""
    var doc = loads('["a","b","c","d"]')
    assert_equal(_joined(_strings(jsonpath_query(doc, "$[0,2]"))), "a,c")
    assert_equal(_joined(_strings(jsonpath_query(doc, "$[2,0]"))), "c,a")
    assert_equal(_joined(_strings(jsonpath_query(doc, "$[0,0]"))), "a,a")


def test_union_of_names() raises:
    """Defect 5: `$['a','b']` used to return only the first."""
    var doc = loads('{"a":1,"b":2,"c":3}')
    assert_equal(_joined(_ints(jsonpath_query(doc, "$['a','b']"))), "1,2")
    assert_equal(_joined(_ints(jsonpath_query(doc, '$["c","a"]'))), "3,1")


def test_union_mixes_selector_kinds() raises:
    """Section 2.5.1: one segment may mix selectors of any kind."""
    var doc = loads('{"a":[10,20,30],"b":"x"}')
    assert_equal(_joined(_ints(jsonpath_query(doc, "$['a'][0,2]"))), "10,30")
    assert_equal(
        _joined(_ints(jsonpath_query(doc, "$.a[0, 1:3, -1]"))),
        "10,20,30,30",
    )
    var mixed = jsonpath_query(doc, "$['b', 'a']")
    assert_equal(len(mixed), 2)
    assert_true(mixed[0].is_string())
    assert_true(mixed[1].is_array())


def test_union_of_filters() raises:
    """RFC 9535 section 2.3.5.3: `$.o[?@<3, ?@<3]`."""
    var doc = _filter_doc()
    assert_equal(
        _joined(_ints(jsonpath_query(doc, "$.o[?@<3, ?@<3]"))), "1,2,1,2"
    )


# -- section 2.3.5, filters ----------------------------------------


def test_filter_examples_from_section_2_3_5_3() raises:
    var doc = _filter_doc()
    var kilo = jsonpath_query(doc, "$.a[?@.b == 'kilo']")
    assert_equal(len(kilo), 1)
    assert_equal(kilo[0]["b"].string_value(), "kilo")

    # Defect 6: a parenthesised filter used to return nothing at all.
    var parens = jsonpath_query(doc, "$.a[?(@.b == 'kilo')]")
    assert_equal(len(parens), 1)

    assert_equal(_joined(_ints(jsonpath_query(doc, "$.a[?@>3.5]"))), "5,4,6")
    assert_equal(len(jsonpath_query(doc, "$.a[?@.b]")), 4)
    assert_equal(len(jsonpath_query(doc, "$[?@.*]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@[?@.b]]")), 1)
    assert_equal(len(jsonpath_query(doc, '$.a[?@<2 || @.b == "k"]')), 2)
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.o[?@>1 && @<4]"))), "2,3")
    assert_equal(len(jsonpath_query(doc, "$.o[?@.u || @.x]")), 1)
    assert_equal(len(jsonpath_query(doc, "$.a[?@.b == $.x]")), 6)
    assert_equal(len(jsonpath_query(doc, "$.a[?@ == @]")), 10)


def test_filter_existence_test() raises:
    """Defect 6: `$[?@.isbn]` has no comparison to lean on."""
    var doc = loads('[{"isbn":"x"},{"isbn":null},{"other":1}]')
    # A member whose value is null still exists, so the test holds.
    assert_equal(len(jsonpath_query(doc, "$[?@.isbn]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@.other]")), 1)


def test_filter_logical_operators() raises:
    """Defect 6: `&&`, `||`, `!` and parenthesised groups."""
    var doc = loads('[{"a":1,"b":1},{"a":1},{"b":1},{"c":1}]')
    assert_equal(len(jsonpath_query(doc, "$[?@.a && @.b]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@.a || @.b]")), 3)
    assert_equal(len(jsonpath_query(doc, "$[?!@.a]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?!(@.a || @.b)]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?(@.a || @.b) && !@.c]")), 3)
    # `&&` binds tighter than `||`, so this is `a || (b && c)`.
    assert_equal(len(jsonpath_query(doc, "$[?@.a || @.b && @.c]")), 2)


def test_filter_root_reference() raises:
    """Defect 6: `$` inside a filter names the document, not the node."""
    var doc = loads('{"limit":10,"items":[{"v":5},{"v":10},{"v":15}]}')
    assert_equal(len(jsonpath_query(doc, "$.items[?@.v < $.limit]")), 1)
    assert_equal(len(jsonpath_query(doc, "$.items[?@.v <= $.limit]")), 2)


def test_filter_nested_paths_and_bracket_selectors() raises:
    """Defect 6: `@.a.b`, `@['a']` and `@[0]` are all singular queries."""
    var doc = loads('[{"a":{"b":1}},{"a":{"b":2}},{"a":[9]}]')
    assert_equal(len(jsonpath_query(doc, "$[?@.a.b == 1]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@['a']['b'] == 2]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@.a[0] == 9]")), 1)


def test_filter_literal_on_the_left() raises:
    """Defect 6: a comparison may put the literal first."""
    var doc = loads('[{"a":1},{"a":2}]')
    assert_equal(len(jsonpath_query(doc, "$[?1 == @.a]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?2 > @.a]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?'x' == 'x']")), 2)


def test_filter_single_quoted_literals() raises:
    """Defect 6: single-quoted string literals in comparisons."""
    var doc = loads('[{"a":"x"},{"a":"y"}]')
    assert_equal(len(jsonpath_query(doc, "$[?@.a == 'x']")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@.a != 'x']")), 1)


def test_filter_bracket_inside_string_literal() raises:
    """Defect 10: the old extent scan ended the filter inside a string."""
    var doc = loads('[{"a":"]"},{"a":"["},{"a":"]["},{"a":"z"}]')
    assert_equal(len(jsonpath_query(doc, '$[?@.a=="]"]')), 1)
    assert_equal(len(jsonpath_query(doc, '$[?@.a=="["]')), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@.a == '][']")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@.a == ']]]']")), 0)


def test_filter_applies_to_object_members() raises:
    """Defect 9: section 2.3.5.2 filters object members too."""
    var doc = loads('{"o":{"p":{"v":1},"q":{"v":2},"r":{"w":3}}}')
    assert_equal(len(jsonpath_query(doc, "$.o[?@.v]")), 2)
    assert_equal(len(jsonpath_query(doc, "$.o[?@.v > 1]")), 1)


def test_filter_ignores_non_collections() raises:
    var doc = loads('{"a":1,"b":"x"}')
    assert_equal(len(jsonpath_query(doc, "$.a[?@]")), 0)
    assert_equal(len(jsonpath_query(doc, "$.b[?@]")), 0)


# -- section 2.3.5.2.2, comparison type rules ----------------------


def test_ordering_needs_matching_types() raises:
    """Defect 7: `<`, `<=`, `>` and `>=` are false across types.

    The old code called incomparable operands equal, so `@.name <= 5`
    held for every object that had a name at all.
    """
    var doc = loads('[{"name":"Alice"},{"name":"Bob"},{"name":3}]')
    assert_equal(len(jsonpath_query(doc, "$[?@.name <= 5]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@.name < 5]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@.name > 5]")), 0)
    assert_equal(len(jsonpath_query(doc, "$[?@.name >= 5]")), 0)
    # Equality across types is simply false, never an error.
    assert_equal(len(jsonpath_query(doc, "$[?@.name == 5]")), 0)


def test_ordering_of_booleans_and_null_is_false() raises:
    var doc = loads("[true,false,null]")
    assert_equal(len(jsonpath_query(doc, "$[?@ < true]")), 0)
    assert_equal(len(jsonpath_query(doc, "$[?@ <= true]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@ == null]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@ >= null]")), 1)


def test_nothing_semantics() raises:
    """Defect 7: a missing member is Nothing, which is not null.

    Nothing equals only Nothing, so `@.missing != 1` is true and
    `@.missing == @.also_missing` is true as well.
    """
    var doc = loads('[{"a":1},{"b":2}]')
    assert_equal(len(jsonpath_query(doc, "$[?@.missing != 1]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@.missing == 1]")), 0)
    assert_equal(len(jsonpath_query(doc, "$[?@.missing == @.other]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@.a != 1]")), 1)
    # Nothing is not null: a member holding null is present.
    var nulls = loads('[{"a":null},{"b":1}]')
    assert_equal(len(jsonpath_query(nulls, "$[?@.a == null]")), 1)


def test_numbers_compare_across_representations() raises:
    var doc = loads("[1, 1.0, 2, 2.5]")
    assert_equal(len(jsonpath_query(doc, "$[?@ == 1]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@ == 1.0]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@ > 2]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?@ >= 2]")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@ == 2.5e0]")), 1)


def test_strings_compare_by_code_point() raises:
    var doc = loads('["a","b","Z"]')
    assert_equal(len(jsonpath_query(doc, "$[?@ < 'b']")), 2)
    assert_equal(len(jsonpath_query(doc, "$[?@ > 'a']")), 1)


def test_deep_equality_of_arrays_and_objects() raises:
    """Defect 8: structural equality, with member order ignored."""
    var doc = loads(
        '[{"x":[1,2],"y":[1,2]},'
        '{"x":[1,2],"y":[2,1]},'
        '{"x":{"a":1,"b":2},"y":{"b":2,"a":1}},'
        '{"x":{"a":1},"y":{"a":1,"b":2}},'
        '{"x":[],"y":[]},'
        '{"x":[1],"y":{"0":1}}]'
    )
    assert_equal(len(jsonpath_query(doc, "$[?@.x == @.y]")), 3)
    assert_equal(len(jsonpath_query(doc, "$[?@.x != @.y]")), 3)


# -- section 2.4, function extensions ------------------------------


def test_function_length() raises:
    var doc = loads(
        '[{"a":"hello"},{"a":[1,2,3]},{"a":{"x":1}},{"a":7},{"b":1}]'
    )
    assert_equal(len(jsonpath_query(doc, "$[?length(@.a) == 5]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?length(@.a) == 3]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?length(@.a) == 1]")), 1)
    # A number has no length, and neither has an absent member, so
    # both compare as Nothing.
    assert_equal(len(jsonpath_query(doc, "$[?length(@.a) >= 0]")), 3)


def test_function_length_counts_code_points() raises:
    """Section 2.4.4 counts scalar values, not the bytes encoding them."""
    var doc = loads('[{"a":"\\u65e5\\u672c\\u8a9e"},{"a":"abc"}]')
    assert_equal(len(jsonpath_query(doc, "$[?length(@.a) == 3]")), 2)


def test_function_count() raises:
    var doc = loads('[{"a":[1,2,3]},{"a":[1]},{"a":{"p":1,"q":2}}]')
    assert_equal(len(jsonpath_query(doc, "$[?count(@.a[*]) == 3]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?count(@.a.*) == 2]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?count(@.nope[*]) == 0]")), 3)


def test_function_match_and_search() raises:
    """`match` anchors to the whole string, `search` does not."""
    var doc = _filter_doc()
    assert_equal(len(jsonpath_query(doc, '$.a[?match(@.b, "[jk]")]')), 2)
    assert_equal(len(jsonpath_query(doc, '$.a[?search(@.b, "[jk]")]')), 3)

    var tz = loads('[{"z":"Europe/Berlin"},{"z":"America/Lima"},{"z":7}]')
    assert_equal(len(jsonpath_query(tz, "$[?match(@.z, 'Europe/.*')]")), 1)
    assert_equal(len(jsonpath_query(tz, "$[?search(@.z, 'eri')]")), 1)
    # A non-string subject is LogicalFalse rather than an error.
    assert_equal(len(jsonpath_query(tz, "$[?match(@.z, '.*')]")), 2)


def test_function_match_pattern_from_the_document() raises:
    """A pattern that is not a literal is compiled while evaluating."""
    var doc = loads('{"p":"a+","v":["a","aa","b"]}')
    assert_equal(len(jsonpath_query(doc, "$.v[?match(@, $.p)]")), 2)


def test_function_value() raises:
    var doc = loads(
        '[{"c":{"color":"red"}},{"c":{"color":"blue"}},'
        '{"c":{"a":{"color":"red"},"b":{"color":"red"}}}]'
    )
    # A nodelist of more than one node converts to Nothing, so the
    # third entry does not match even though both colours are red.
    assert_equal(len(jsonpath_query(doc, '$[?value(@..color) == "red"]')), 1)


def test_function_well_typedness_is_checked() raises:
    """Section 2.4.3: a call that does not match its declaration fails."""
    var doc = loads('[{"a":1}]')
    # `length` returns a value, so it is not a test expression.
    _assert_raises(doc, "$[?length(@.a)]", "value type used as a test")
    # `match` returns a logical value, so it cannot be compared.
    _assert_raises(doc, "$[?match(@.a, 'x') == true]", "logical compared")
    # `count` wants a nodelist, not a literal.
    _assert_raises(doc, "$[?count(1) == 1]", "literal given to count")
    # `length` wants a value, so a non-singular query is out.
    _assert_raises(doc, "$[?length(@.*) == 1]", "non-singular given to length")
    _assert_raises(doc, "$[?match(@.a)]", "wrong arity")
    _assert_raises(doc, "$[?length(@.a, @.a) == 1]", "wrong arity")
    _assert_raises(doc, "$[?nosuch(@.a)]", "unknown function")
    _assert_raises(doc, "$[?Length(@.a) == 1]", "function names are lowercase")


def test_function_count_accepts_non_singular_queries() raises:
    """The counterpart: a NodesType parameter wants a query, any query."""
    var doc = loads('{"a":{"b":[1,2,3]}}')
    assert_equal(len(jsonpath_query(doc, "$[?count(@..*) >= 2]")), 1)


# -- section 2.3.5.2.1, the singular query rule --------------------


def test_non_singular_query_in_comparison_is_a_parse_error() raises:
    var doc = loads('[{"a":1}]')
    _assert_raises(doc, "$[?@.* == 1]", "wildcard is not singular")
    _assert_raises(doc, "$[?@..a == 1]", "descendant is not singular")
    _assert_raises(doc, "$[?@[0,1] == 1]", "a union is not singular")
    _assert_raises(doc, "$[?@[0:1] == 1]", "a slice is not singular")
    _assert_raises(doc, "$[?1 == $.a[*]]", "not singular on the right")
    _assert_raises(doc, "$[?@[?@.b] == 1]", "a filter is not singular")


def test_singular_queries_are_accepted_in_comparisons() raises:
    var doc = loads('{"r":2,"items":[{"a":{"b":2}},{"a":{"b":3}}]}')
    assert_equal(len(jsonpath_query(doc, "$.items[?@.a.b == $.r]")), 1)
    assert_equal(len(jsonpath_query(doc, "$.items[?@['a']['b'] == 2]")), 1)


# -- section 2.5.2, descendant segments ----------------------------


def test_descendant_examples() raises:
    var doc = loads(
        '{"o": {"j": 1, "k": 2}, "a": [5, 3, [{"j": 4}, {"k": 6}]]}'
    )
    assert_equal(_joined(_ints(jsonpath_query(doc, "$..j"))), "1,4")
    assert_equal(len(jsonpath_query(doc, "$..[0]")), 2)
    assert_equal(
        len(jsonpath_query(doc, "$..[*]")), len(jsonpath_query(doc, "$..*"))
    )
    assert_equal(len(jsonpath_query(doc, "$..o")), 1)
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.o..[*, *]"))), "1,2,1,2")


def test_descendant_does_not_overflow_on_deep_documents() raises:
    """Defect 2: the old collector recursed once per level.

    Building a document a few thousand levels deep and walking it with
    a descendant segment is what used to exhaust the stack.
    """
    var depth = 1000
    var text = String()
    for _ in range(depth):
        text += '{"a":'
    text += "1"
    for _ in range(depth):
        text += "}"
    var doc = loads(text)
    assert_equal(len(jsonpath_query(doc, "$..a")), depth)


# -- section 2.7, normalized paths ---------------------------------


def test_normalized_paths() raises:
    var doc = _bookstore()
    var compiled = JSONPath.compile("$.store.book[0].title")
    var located = compiled.paths(doc)
    assert_equal(len(located), 1)
    assert_equal(located[0], "$['store']['book'][0]['title']")

    var authors = JSONPath.compile("$..author").paths(doc)
    assert_equal(len(authors), 4)
    assert_equal(authors[0], "$['store']['book'][0]['author']")
    assert_equal(authors[3], "$['store']['book'][3]['author']")


def test_normalized_path_escaping() raises:
    """Section 2.7 escapes only `'`, `\\` and the control characters."""
    var doc = loads(
        '{"a\'b":1,"c\\\\d":2,"e\\tf":3,"g\\u0001h":4,"i/j":5,"k\\"l":6}'
    )
    var located = JSONPath.compile("$[*]").paths(doc)
    assert_equal(len(located), 6)
    assert_equal(located[0], "$['a\\'b']")
    assert_equal(located[1], "$['c\\\\d']")
    assert_equal(located[2], "$['e\\tf']")
    assert_equal(located[3], "$['g\\u0001h']")
    # A solidus and a quotation mark stand for themselves here, unlike
    # in a JSON string.
    assert_equal(located[4], "$['i/j']")
    assert_equal(located[5], "$['k\"l']")


def test_query_with_paths() raises:
    var doc = loads('{"a":[10,20]}')
    var pairs = JSONPath.compile("$.a[*]").query_with_paths(doc)
    assert_equal(len(pairs), 2)
    assert_equal(pairs[0][0], "$['a'][0]")
    assert_equal(pairs[0][1].raw_json(), String("10"))
    assert_equal(pairs[1][0], "$['a'][1]")
    assert_equal(pairs[1][1].raw_json(), String("20"))


def test_normalized_path_round_trips() raises:
    """A normalized path is itself a query naming the same one node."""
    var doc = _bookstore()
    var located = JSONPath.compile("$..price").paths(doc)
    assert_equal(len(located), 5)
    for i in range(len(located)):
        var again = jsonpath_query(doc, located[i])
        assert_equal(len(again), 1)


def test_normalized_path_inside_a_filter() raises:
    """A filter's results carry the path of the node, not of the test."""
    var doc = loads('{"a":[{"v":1},{"v":2}]}')
    var located = JSONPath.compile("$.a[?@.v > 1]").paths(doc)
    assert_equal(len(located), 1)
    assert_equal(located[0], "$['a'][1]")


# -- compiled reuse and the two free functions ---------------------


def test_compile_once_run_many() raises:
    var compiled = JSONPath.compile("$.items[?@.n > 1].n")
    var first = loads('{"items":[{"n":1},{"n":2},{"n":3}]}')
    var second = loads('{"items":[{"n":9}]}')
    assert_equal(len(compiled.query(first)), 2)
    assert_equal(len(compiled.query(second)), 1)
    assert_equal(len(compiled.query(first)), 2)


def test_jsonpath_root() raises:
    var doc = loads('{"a":1}')
    var results = jsonpath_query(doc, "$")
    assert_equal(len(results), 1)
    assert_true(results[0].is_object())
    assert_equal(JSONPath.compile("$").paths(doc)[0], "$")


def test_jsonpath_one() raises:
    var doc = loads('{"name":"Alice"}')
    assert_equal(jsonpath_one(doc, "$.name").string_value(), "Alice")


def test_jsonpath_one_no_match() raises:
    var doc = loads('{"name":"Alice"}')
    var caught = False
    try:
        _ = jsonpath_one(doc, "$.nonexistent")
    except:
        caught = True
    assert_true(caught)


def test_shorthand_and_bracket_notation_agree() raises:
    var doc = loads('{"a":{"b":{"c":42}}}')
    assert_equal(jsonpath_one(doc, "$.a.b.c").raw_json(), String("42"))
    assert_equal(jsonpath_one(doc, "$['a']['b']['c']").raw_json(), String("42"))
    assert_equal(jsonpath_one(doc, '$["a"].b["c"]').raw_json(), String("42"))


def test_selectors_miss_quietly_on_the_wrong_kind() raises:
    """A selector that does not apply yields nothing, and never raises."""
    var doc = loads('{"a":1,"b":[1,2],"c":"xy"}')
    assert_equal(len(jsonpath_query(doc, "$.a[0]")), 0)
    assert_equal(len(jsonpath_query(doc, "$.a[*]")), 0)
    assert_equal(len(jsonpath_query(doc, "$.b.name")), 0)
    assert_equal(len(jsonpath_query(doc, "$.c[0]")), 0)
    assert_equal(len(jsonpath_query(doc, "$.c[:]")), 0)


# -- section 2.1, whitespace ---------------------------------------


def test_whitespace_is_allowed_where_the_abnf_allows_it() raises:
    """Defect 13: around selectors, commas, operators and segments."""
    var doc = loads('{"a":[1,2,3],"b":2}')
    assert_equal(len(jsonpath_query(doc, "$[ 'a' ]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[ 'a' , 'b' ]")), 2)
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.a[ 0 , 2 ]"))), "1,3")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.a[ 1 : 3 ]"))), "2,3")
    assert_equal(_joined(_ints(jsonpath_query(doc, "$.a[ : : 2 ]"))), "1,3")
    assert_equal(len(jsonpath_query(doc, "$.a[ ?  @  >  1 ]")), 2)
    assert_equal(len(jsonpath_query(doc, "$.a[?@>1  &&  @<3]")), 1)
    assert_equal(len(jsonpath_query(doc, "$.a[? ! (@ > 1) ]")), 1)
    assert_equal(len(jsonpath_query(doc, "$\t.a\n[0]")), 1)


def test_leading_and_trailing_whitespace_are_rejected() raises:
    """The ABNF has no `S` outside `segments`, so neither is allowed."""
    var doc = loads('{"a":1}')
    _assert_raises(doc, " $.a", "leading whitespace")
    _assert_raises(doc, "$.a ", "trailing whitespace")
    _assert_raises(doc, "$ ", "trailing whitespace after the root")


# -- section 2.1, invalid queries ----------------------------------


def test_invalid_queries_are_rejected() raises:
    """Defect 11: the old catch-all stepped over anything it disliked."""
    var doc = loads('{"a":1}')
    _assert_raises(doc, "", "an empty query")
    _assert_raises(doc, "a", "no root identifier")
    _assert_raises(doc, "$.a~", "a stray byte")
    _assert_raises(doc, "$a", "a name glued to the root")
    _assert_raises(doc, "$..", "a descendant segment with no selector")
    _assert_raises(doc, "$...a", "three dots")
    _assert_raises(doc, "$.", "a trailing dot")
    _assert_raises(doc, "$.a.", "a trailing dot")
    _assert_raises(doc, "$[", "an unclosed bracket")
    _assert_raises(doc, "$[]", "an empty bracketed selection")
    _assert_raises(doc, "$[,]", "an empty selector")
    _assert_raises(doc, "$[0,]", "a trailing comma")
    _assert_raises(doc, "$['a'", "an unclosed bracket")
    _assert_raises(doc, "$['a]", "an unterminated string literal")
    _assert_raises(doc, "$[1 2]", "two integers")
    _assert_raises(doc, "$[*.a]", "a wildcard with a suffix")
    _assert_raises(doc, "$..[]", "an empty descendant selection")


def test_invalid_member_name_shorthand() raises:
    """Defect 12: the shorthand follows `member-name-shorthand`."""
    var doc = loads('{"a":1}')
    _assert_raises(doc, "$.1abc", "a name may not begin with a digit")
    _assert_raises(doc, "$.foo bar", "a name may not contain a space")
    _assert_raises(doc, "$.a-b", "a name may not contain a hyphen")
    _assert_raises(doc, "$.-a", "a name may not begin with a hyphen")
    _assert_raises(doc, "$.'a'", "a quoted name needs brackets")
    _assert_raises(doc, "$.*a", "the wildcard takes no suffix")
    # These are the shapes the ABNF does allow.
    var named = loads('{"_x":1,"A9":2,"\\u00e9t\\u00e9":3}')
    assert_equal(len(jsonpath_query(named, "$._x")), 1)
    assert_equal(len(jsonpath_query(named, "$.A9")), 1)
    assert_equal(len(jsonpath_query(named, "$.été")), 1)


def test_invalid_integers() raises:
    var doc = loads("[1,2,3]")
    _assert_raises(doc, "$[-0]", "'-0' is not an index")
    _assert_raises(doc, "$[01]", "a leading zero")
    _assert_raises(doc, "$[+1]", "an explicit plus")
    _assert_raises(doc, "$[1.0]", "a fractional index")
    _assert_raises(doc, "$[9007199254740992]", "outside the safe range")
    _assert_raises(doc, "$[0:01]", "a leading zero in a slice")
    _assert_raises(doc, "$[:::]", "a fourth slice field")


def test_invalid_filters() raises:
    var doc = loads('{"a":1}')
    _assert_raises(doc, "$[?]", "an empty filter")
    _assert_raises(doc, "$[?@.a ==]", "a missing right operand")
    _assert_raises(doc, "$[?== 1]", "a missing left operand")
    _assert_raises(doc, "$[?@.a == 1 &&]", "a dangling conjunction")
    _assert_raises(doc, "$[?(@.a == 1]", "an unclosed group")
    _assert_raises(doc, "$[?@.a == 1)]", "a stray close paren")
    _assert_raises(doc, "$[?!@.a == 1]", "'!' does not take a comparison")
    _assert_raises(doc, "$[?@.a < @.b < @.c]", "comparisons do not chain")
    _assert_raises(doc, "$[?1]", "a literal is not a test")
    _assert_raises(doc, "$[?'x']", "a literal is not a test")
    _assert_raises(doc, "$[?@.a = 1]", "'=' is not an operator")
    _assert_raises(doc, "$[?@.a === 1]", "'===' is not an operator")
    _assert_raises(doc, "$[?@.a == 'x\"]", "mismatched quotes")
    _assert_raises(doc, "$[?@.a == 1e]", "an exponent with no digits")
    _assert_raises(doc, "$[?@.a == 1.]", "a fraction with no digits")
    _assert_raises(doc, "$[?@.a == 01]", "a leading zero")
    _assert_raises(doc, "$[?@.a == truer]", "a keyword with a suffix")
    _assert_raises(doc, "$[?@.a & @.b]", "'&' is not an operator")


def test_invalid_string_literal_escapes() raises:
    var doc = loads('{"a":1}')
    _assert_raises(doc, '$["\\q"]', "an unknown escape")
    _assert_raises(doc, '$["\\u00"]', "a short '\\u' escape")
    _assert_raises(doc, '$["\\ud834"]', "a lone high surrogate")
    _assert_raises(doc, '$["\\udd1e"]', "a lone low surrogate")
    _assert_raises(doc, "$['\\\"']", "the other quote may not be escaped")
    _assert_raises(doc, '$["\\\'"]', "the other quote may not be escaped")


def test_filter_negates_a_function_call() raises:
    var doc = loads('[{"z":"Europe/Berlin"},{"z":"America/Lima"}]')
    assert_equal(len(jsonpath_query(doc, "$[?!match(@.z, 'Europe/.*')]")), 1)
    assert_equal(len(jsonpath_query(doc, "$[?!search(@.z, 'eri')]")), 1)


def test_logical_not_applies_once() raises:
    """`logical-not-op` is optional, not repeatable."""
    var doc = loads('[{"a":1}]')
    _assert_raises(doc, "$[?!!@.a]", "'!' does not stack")
    _assert_raises(doc, "$[?!]", "'!' with nothing to negate")


def test_descendant_segment_carries_a_filter() raises:
    var doc = _filter_doc()
    var got = jsonpath_query(doc, "$..[?@.b == 'kilo']")
    assert_equal(len(got), 1)
    assert_equal(got[0]["b"].string_value(), "kilo")


def test_empty_member_name() raises:
    """The empty string is a member name like any other."""
    var doc = loads('{"":1,"a":2}')
    assert_equal(jsonpath_one(doc, "$['']").raw_json(), String("1"))
    assert_equal(JSONPath.compile("$['']").paths(doc)[0], "$['']")


def test_query_nesting_is_capped() raises:
    """A query may not nest deeply enough to exhaust the parser."""
    var doc = loads('{"a":1}')
    var deep = String("$")
    for _ in range(70):
        deep += "[?@"
    _assert_raises(doc, deep, "a query that nests too deeply")


def test_exponent_marker_is_lowercase() raises:
    """Section 2.3.5.1 spells `exp` with a lowercase `e` only."""
    var doc = loads("[100]")
    assert_equal(len(jsonpath_query(doc, "$[?@ == 1e2]")), 1)
    _assert_raises(doc, "$[?@ == 1E2]", "an uppercase exponent marker")


def test_jsonpath_unclosed_filter_rejected() raises:
    """Regression: an unclosed ``[?`` underflowed the old extent scan
    and crashed the tokenizer (found by fuzz_jsonpath). It must raise a
    regular ``Error``."""
    var doc = loads('{"x":1}')
    var paths = [
        String("$[?"),
        String("$[?@.price<"),
        String("$..book[?@.price<"),
        String("$.store[?"),
    ]
    for i in range(len(paths)):
        var caught = False
        try:
            _ = jsonpath_query(doc, paths[i].copy())
        except:
            caught = True
        assert_true(caught, "expected unclosed filter to raise: " + paths[i])


def main() raises:
    print("=" * 60)
    print("test_jsonpath.mojo - JSONPath (RFC 9535) Tests")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
