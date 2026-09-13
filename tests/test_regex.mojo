# Tests for json/regex.mojo -- the I-Regexp (RFC 9485) engine.
#
# Three things need pinning here. First the accepted language, since
# JSONPath `match()`/`search()` and JSON Schema `pattern` both answer
# to RFC 9485 and a quietly wrong quantifier or class is a silently
# wrong validation result. Second the rejections: I-Regexp is defined
# as much by what it refuses (backreferences, lazy quantifiers,
# unbalanced brackets) as by what it accepts, and a parser that
# shrugs at a bad pattern hides the caller's typo. Third that matching
# counts scalar values rather than bytes, which only shows up on
# non-ASCII input and is the kind of thing an ASCII-only test suite
# will never notice.

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from json.regex import Regex, regex_full_match, regex_search


def _assert_invalid(pattern: String) raises:
    """Assert that `pattern` is rejected at compile time."""
    var rejected = False
    try:
        _ = Regex.compile(pattern)
    except:
        rejected = True
    assert_true(rejected, "expected compile to reject: " + pattern)


def test_literals() raises:
    assert_true(regex_full_match("abc", "abc"))
    assert_false(regex_full_match("abc", "ab"))
    assert_false(regex_full_match("abc", "abcd"))
    assert_false(regex_full_match("abc", "ABC"))

    # The empty pattern is a valid I-Regexp and matches only the empty
    # string.
    assert_true(regex_full_match("", ""))
    assert_false(regex_full_match("", "a"))


def test_literal_metacharacters() raises:
    """`^` and `$` are ordinary characters, not anchors."""
    assert_true(regex_full_match("^a", "^a"))
    assert_false(regex_full_match("^a", "a"))
    assert_true(regex_full_match("a$", "a$"))
    assert_false(regex_full_match("a$", "a"))

    # `\$` is accepted on top of the RFC's escape list.
    assert_true(regex_full_match("a\\$", "a$"))
    assert_true(regex_full_match("\\^\\-\\.", "^-."))


def test_alternation() raises:
    assert_true(regex_full_match("a|b", "a"))
    assert_true(regex_full_match("a|b", "b"))
    assert_false(regex_full_match("a|b", "c"))

    assert_true(regex_full_match("cat|dog|bird", "dog"))
    assert_true(regex_full_match("cat|dog|bird", "bird"))
    assert_false(regex_full_match("cat|dog|bird", "cato"))

    # An empty branch is legal, so this matches the empty string too.
    assert_true(regex_full_match("a|", ""))
    assert_true(regex_full_match("|a", "a"))

    # Alternation spans the whole branch, not just the adjacent atom.
    assert_true(regex_full_match("ab|cd", "cd"))
    assert_false(regex_full_match("ab|cd", "acd"))


def test_quantifier_question() raises:
    assert_true(regex_full_match("ab?c", "ac"))
    assert_true(regex_full_match("ab?c", "abc"))
    assert_false(regex_full_match("ab?c", "abbc"))


def test_quantifier_star() raises:
    assert_true(regex_full_match("ab*c", "ac"))
    assert_true(regex_full_match("ab*c", "abc"))
    assert_true(regex_full_match("ab*c", "abbbbbbc"))
    assert_false(regex_full_match("ab*c", "abd"))

    assert_true(regex_full_match("a*", ""))
    assert_true(regex_full_match("a*", "aaaaa"))


def test_quantifier_plus() raises:
    assert_false(regex_full_match("ab+c", "ac"))
    assert_true(regex_full_match("ab+c", "abc"))
    assert_true(regex_full_match("ab+c", "abbbc"))

    assert_false(regex_full_match("a+", ""))
    assert_true(regex_full_match("a+", "a"))


def test_quantifier_exact() raises:
    assert_false(regex_full_match("a{3}", "aa"))
    assert_true(regex_full_match("a{3}", "aaa"))
    assert_false(regex_full_match("a{3}", "aaaa"))

    # `{0}` erases the atom rather than being rejected.
    assert_true(regex_full_match("a{0}", ""))
    assert_false(regex_full_match("a{0}", "a"))
    assert_true(regex_full_match("ba{0}c", "bc"))


def test_quantifier_open_range() raises:
    assert_false(regex_full_match("a{2,}", "a"))
    assert_true(regex_full_match("a{2,}", "aa"))
    assert_true(regex_full_match("a{2,}", "aaaaaaaa"))

    # `{0,}` is `*`, including on the empty subject.
    assert_true(regex_full_match("a{0,}", ""))
    assert_true(regex_full_match("a{0,}", "aaa"))

    # The mandatory prefix and the loop must not be double counted.
    assert_false(regex_full_match("ab{3,}c", "abbc"))
    assert_true(regex_full_match("ab{3,}c", "abbbc"))
    assert_true(regex_full_match("ab{3,}c", "abbbbbc"))


def test_quantifier_closed_range() raises:
    assert_false(regex_full_match("a{2,4}", "a"))
    assert_true(regex_full_match("a{2,4}", "aa"))
    assert_true(regex_full_match("a{2,4}", "aaa"))
    assert_true(regex_full_match("a{2,4}", "aaaa"))
    assert_false(regex_full_match("a{2,4}", "aaaaa"))

    assert_true(regex_full_match("a{0,2}", ""))
    assert_true(regex_full_match("a{0,2}", "aa"))
    assert_false(regex_full_match("a{0,2}", "aaa"))

    # A range whose bounds are equal behaves like `{n}`.
    assert_true(regex_full_match("a{3,3}", "aaa"))
    assert_false(regex_full_match("a{3,3}", "aaaa"))

    # The expansion cap is 1000, so the boundary itself must compile.
    var at_cap = Regex.compile("a{1000}")
    assert_true(at_cap.full_match("a" * 1000))
    assert_false(at_cap.full_match("a" * 999))


def test_dot() raises:
    assert_true(regex_full_match(".", "a"))
    assert_true(regex_full_match(".", " "))
    assert_true(regex_full_match(".", "\t"))
    assert_true(regex_full_match("a.c", "a.c"))
    assert_true(regex_full_match("a.c", "abc"))

    # RFC 9485 section 3 maps an unescaped dot to `[^\n\r]`, so it
    # matches neither line feed nor carriage return.
    assert_false(regex_full_match(".", "\n"))
    assert_false(regex_full_match(".", "\r"))
    assert_false(regex_full_match("a.c", "a\nc"))

    # The escaped form is the literal period and nothing else.
    assert_true(regex_full_match("a\\.c", "a.c"))
    assert_false(regex_full_match("a\\.c", "abc"))

    assert_true(regex_full_match(".*", "anything at all"))


def test_char_class_basics() raises:
    assert_true(regex_full_match("[abc]", "b"))
    assert_false(regex_full_match("[abc]", "d"))
    assert_true(regex_full_match("[abc]+", "cabba"))
    assert_false(regex_full_match("[abc]+", "cabxba"))


def test_char_class_ranges() raises:
    assert_true(regex_full_match("[a-z]+", "hello"))
    assert_false(regex_full_match("[a-z]+", "Hello"))
    assert_true(regex_full_match("[a-zA-Z0-9]+", "Ab9"))

    # Endpoints are inclusive on both sides.
    assert_true(regex_full_match("[a-c]", "a"))
    assert_true(regex_full_match("[a-c]", "c"))
    assert_false(regex_full_match("[a-c]", "d"))

    # A dash at either end of the class is a literal, not a range.
    assert_true(regex_full_match("[a-]", "-"))
    assert_true(regex_full_match("[a-]", "a"))
    assert_true(regex_full_match("[-a]", "-"))
    assert_true(regex_full_match("[0-9-]+", "1-2"))


def test_char_class_negation() raises:
    assert_true(regex_full_match("[^abc]", "d"))
    assert_false(regex_full_match("[^abc]", "a"))
    assert_true(regex_full_match("[^a-z]+", "XYZ"))

    # A negated class is not a dot: it has no newline exemption.
    assert_true(regex_full_match("[^a]", "\n"))
    assert_true(regex_full_match("[^a]", "\r"))

    # `^` after the first position is a literal member.
    assert_true(regex_full_match("[a^]", "^"))
    assert_false(regex_full_match("[^^]", "^"))


def test_char_class_escapes() raises:
    """Escapes inside a class, including the ones the ABNF requires."""
    assert_true(regex_full_match("[\\]]", "]"))
    assert_true(regex_full_match("[\\\\]", "\\"))
    assert_true(regex_full_match("[\\-]", "-"))
    assert_true(regex_full_match("[\\n\\r\\t]+", "\n\r\t"))
    assert_true(regex_full_match("[a\\-c]", "-"))
    assert_false(regex_full_match("[a\\-c]", "b"))

    # An escaped character can still be a range endpoint.
    assert_true(regex_full_match("[\\(-\\+]+", "()*+"))
    assert_false(regex_full_match("[\\(-\\+]+", ","))


def test_multi_char_escapes() raises:
    assert_true(regex_full_match("\\d+", "12345"))
    assert_false(regex_full_match("\\d+", "12a45"))
    assert_true(regex_full_match("\\D+", "abc"))
    assert_false(regex_full_match("\\D+", "ab3"))

    # XSD defines `\s` as exactly tab, line feed, carriage return and
    # space.
    assert_true(regex_full_match("\\s+", " \t\n\r"))
    assert_false(regex_full_match("\\s", "a"))
    assert_true(regex_full_match("\\S+", "abc"))
    assert_false(regex_full_match("\\S+", "ab c"))

    assert_true(regex_full_match("\\w+", "a_Z9"))
    assert_false(regex_full_match("\\w+", "a-Z"))
    assert_true(regex_full_match("\\W+", "-+="))
    assert_false(regex_full_match("\\W+", "-a"))

    # The complements cover the whole scalar range, not just ASCII.
    assert_true(regex_full_match("\\D", "α"))
    assert_true(regex_full_match("\\W", "α"))


def test_multi_char_escapes_in_class() raises:
    assert_true(regex_full_match("[\\d]+", "42"))
    assert_true(regex_full_match("[\\dx]+", "4x2"))
    assert_false(regex_full_match("[\\dx]+", "4y2"))

    # A complement escape inside a class unions in its own ranges, so
    # it stays correct under an outer negation.
    assert_true(regex_full_match("[\\D]+", "abc"))
    assert_false(regex_full_match("[\\D]+", "ab1"))
    assert_true(regex_full_match("[^\\d]+", "abc"))
    assert_false(regex_full_match("[^\\d]+", "ab1"))
    assert_true(regex_full_match("[^\\D]+", "123"))
    assert_false(regex_full_match("[^\\D]+", "12c"))

    assert_true(regex_full_match("[\\s\\d]+", "1 2\t3"))


def test_groups() raises:
    assert_true(regex_full_match("(abc)", "abc"))
    assert_true(regex_full_match("(a|b)c", "ac"))
    assert_true(regex_full_match("(a|b)c", "bc"))
    assert_false(regex_full_match("(a|b)c", "cc"))


def test_groups_with_quantifiers() raises:
    assert_true(regex_full_match("(ab)+", "ababab"))
    assert_false(regex_full_match("(ab)+", "ababa"))
    assert_true(regex_full_match("(ab)*", ""))
    assert_true(regex_full_match("(ab)?c", "c"))
    assert_true(regex_full_match("(ab)?c", "abc"))

    assert_true(regex_full_match("(ab){2}", "abab"))
    assert_false(regex_full_match("(ab){2}", "ababab"))
    assert_true(regex_full_match("(a|b){2,3}", "aba"))
    assert_false(regex_full_match("(a|b){2,3}", "abab"))

    # Nested groups, so the fragment combinators have to compose.
    assert_true(regex_full_match("((a|b)c)+", "acbc"))
    assert_false(regex_full_match("((a|b)c)+", "acb"))

    # A group that can match empty must not loop forever.
    assert_true(regex_full_match("(a?)*", ""))
    assert_true(regex_full_match("(a?)*", "aaa"))
    assert_true(regex_full_match("()*", ""))


def test_realistic_patterns() raises:
    var date = Regex.compile("\\d{4}-\\d{2}-\\d{2}")
    assert_true(date.full_match("2024-01-31"))
    assert_false(date.full_match("2024-1-31"))
    assert_false(date.full_match("2024-01-311"))

    var email = Regex.compile("[a-z0-9._]+@[a-z0-9]+(\\.[a-z]{2,4})+")
    assert_true(email.full_match("user.name@example.com"))
    assert_true(email.full_match("a@b.co.uk"))
    assert_false(email.full_match("user.name@example"))
    assert_false(email.full_match("@example.com"))


def test_non_ascii_literals() raises:
    assert_true(regex_full_match("héllo", "héllo"))
    assert_false(regex_full_match("héllo", "hello"))
    assert_true(regex_full_match("日本語", "日本語"))


def test_non_ascii_classes() raises:
    assert_true(regex_full_match("[α-ω]+", "αβγω"))
    assert_false(regex_full_match("[α-ω]+", "Δ"))
    assert_true(regex_full_match("[^α-ω]+", "ABΔ"))

    # A range over astral characters, where each endpoint is four
    # bytes of UTF-8.
    assert_true(regex_full_match("[😀-😇]+", "😀😃😇"))
    assert_false(regex_full_match("[😀-😇]+", "😈"))


def test_non_ascii_dot_counts_scalars() raises:
    """A dot consumes one scalar value, however many bytes it spans."""
    assert_true(regex_full_match(".", "😀"))
    assert_true(regex_full_match(".", "é"))

    # Three characters spanning six bytes: a byte-oriented matcher
    # would want `.{6}` here.
    assert_true(regex_full_match(".{3}", "a😀b"))
    assert_false(regex_full_match(".{6}", "a😀b"))

    assert_true(regex_full_match("a.b", "a😀b"))


def test_search_versus_full_match() raises:
    assert_true(regex_search("b", "abc"))
    assert_false(regex_full_match("b", "abc"))

    assert_true(regex_search("abc", "xxabcxx"))
    assert_false(regex_full_match("abc", "xxabcxx"))

    # A match at either end counts, and so does one at every offset.
    assert_true(regex_search("^a", "^abc"))
    assert_true(regex_search("c$", "abc$"))
    assert_false(regex_search("q", "abc"))

    # An empty match is found in any subject, including an empty one.
    assert_true(regex_search("a*", ""))
    assert_true(regex_search("a*", "bbb"))
    assert_false(regex_search("a", ""))

    # Searching has to try every start, not just the first that can
    # begin a match.
    assert_true(regex_search("ab+c", "aab abbc"))
    assert_true(regex_search("[0-9]{3}", "xx12 345 yy"))
    assert_false(regex_search("[0-9]{3}", "xx12 34 yy"))

    # Multi-byte subjects must not desynchronise the scan.
    assert_true(regex_search("b", "αβγb"))
    assert_true(regex_search("β", "aαβγ"))


def test_reuse_across_subjects() raises:
    """Compiling once and matching many times must be stateless."""
    var re = Regex.compile("(foo|bar)[0-9]+")
    for _ in range(3):
        assert_true(re.full_match("foo12"))
        assert_true(re.full_match("bar7"))
        assert_false(re.full_match("baz1"))
        assert_false(re.full_match("foo"))
        assert_true(re.search("xx bar42 yy"))
        assert_false(re.search("xx baz yy"))


def test_no_catastrophic_backtracking() raises:
    """Patterns that defeat a backtracking engine must still return.

    A backtracking matcher takes exponential time on these; the
    Thompson simulation is linear in the subject for a fixed program,
    so a failing run on a long subject is unremarkable.
    """
    var nested = Regex.compile("(a+)+b")
    assert_false(nested.full_match("a" * 200))
    assert_true(nested.full_match("a" * 200 + "b"))

    var overlapping = Regex.compile("(a|aa)+c")
    assert_false(overlapping.full_match("a" * 200))
    assert_true(overlapping.full_match("a" * 200 + "c"))

    assert_false(regex_search("(x+x+)+y", "x" * 200))


def test_invalid_unbalanced_group() raises:
    _assert_invalid("(abc")
    _assert_invalid("abc)")
    _assert_invalid("(a|b")
    _assert_invalid("((a)")
    _assert_invalid("a)b")

    # Groups are parsed by recursion, so nesting past the depth cap is
    # refused rather than allowed to run the parser off its stack.
    _assert_invalid("(" * 100 + "a" + ")" * 100)
    assert_true(Regex.compile("(" * 60 + "a" + ")" * 60).full_match("a"))


def test_invalid_unbalanced_class() raises:
    _assert_invalid("[abc")
    _assert_invalid("[a-")
    _assert_invalid("[^")
    _assert_invalid("abc]")

    # A class needs at least one member.
    _assert_invalid("[]")
    _assert_invalid("[^]")

    # A range with the endpoints the wrong way round is a typo, not a
    # match against nothing.
    _assert_invalid("[z-a]")
    _assert_invalid("[9-0]")

    # A multi-character escape has no single code point to anchor a
    # range on.
    _assert_invalid("[a-\\d]")


def test_invalid_escapes() raises:
    _assert_invalid("abc\\")
    _assert_invalid("[abc\\")
    _assert_invalid("\\q")
    _assert_invalid("\\8")
    _assert_invalid("[\\q]")

    # Backreferences and the category classes of full XSD are not part
    # of what this engine accepts.
    _assert_invalid("(a)\\1")
    _assert_invalid("\\p{Lu}")


def test_invalid_quantifiers() raises:
    # A quantifier needs an atom in front of it.
    _assert_invalid("*a")
    _assert_invalid("+a")
    _assert_invalid("?a")
    _assert_invalid("{2}")
    _assert_invalid("a|*b")

    # A piece takes one quantifier, which is what rules out the lazy
    # and possessive forms.
    _assert_invalid("a**")
    _assert_invalid("a*?")
    _assert_invalid("a+?")
    _assert_invalid("a{2}{3}")

    # Braces are metacharacters and must be escaped to match literally.
    _assert_invalid("a}")
    _assert_invalid("a{")
    _assert_invalid("a{2")
    _assert_invalid("a{,3}")
    _assert_invalid("a{}")
    _assert_invalid("a{x}")

    # A backwards range.
    _assert_invalid("a{3,1}")


def test_invalid_quantifier_expansion_cap() raises:
    """Bounded repetition is expanded, so the bound has to be capped."""
    _assert_invalid("a{1001}")
    _assert_invalid("a{0,1001}")
    _assert_invalid("a{1001,}")
    _assert_invalid("a{99999999999999999999}")

    # Nesting multiplies, so the bound alone is not a sufficient guard
    # and the total program size is checked as well.
    _assert_invalid("(a{1000}){1000}")
    _assert_invalid("((ab){500}){500}")


def test_escaped_braces_are_literal() raises:
    """The escaped forms of the metacharacters must still work."""
    assert_true(regex_full_match("a\\{2\\}", "a{2}"))
    assert_true(regex_full_match("\\(\\)", "()"))
    assert_true(regex_full_match("\\[\\]", "[]"))
    assert_true(regex_full_match("\\*\\+\\?", "*+?"))
    assert_true(regex_full_match("a\\|b", "a|b"))
    assert_true(regex_full_match("a\\\\b", "a\\b"))


def test_compiled_and_free_function_agree() raises:
    """The module-level helpers are the one-shot form of the struct."""
    var patterns = [
        String("a+b"),
        String("[0-9]{2,4}"),
        String("(x|y)*z"),
        String("."),
    ]
    var subjects = [String("aab"), String("123"), String("xyz"), String("q")]

    for p in patterns:
        var re = Regex.compile(p)
        for s in subjects:
            assert_equal(re.full_match(s), regex_full_match(p, s))
            assert_equal(re.search(s), regex_search(p, s))


def main() raises:
    print("=" * 60)
    print("test_regex.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
