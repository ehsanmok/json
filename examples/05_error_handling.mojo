# Example 05: Error Handling
#
# Demonstrates: Handling JSON parse errors using try/except and assert_raises

from testing import assert_raises
from src import loads, dumps, Value


fn parse_safely(json_str: String) -> String:
    """Attempt to parse JSON and return result or error message."""
    try:
        var result = loads(json_str)
        return "OK: " + dumps(result)
    except e:
        return "Error: " + String(e)


fn main() raises:
    print("JSON Error Handling Examples")
    print("=" * 40)
    print()

    # Valid JSON examples
    print("Valid JSON:")
    print("  '42' ->", parse_safely("42"))
    print("  'true' ->", parse_safely("true"))
    print("  '\"hello\"' ->", parse_safely('"hello"'))
    print("  '[1,2,3]' ->", parse_safely("[1,2,3]"))
    print("  '{\"a\":1}' ->", parse_safely('{"a":1}'))
    print()

    # Invalid JSON - using assert_raises to verify errors are raised
    print("Invalid JSON (verified with assert_raises):")
    
    with assert_raises():
        _ = loads("")  # empty
    print("  '' (empty) -> correctly raises error")
    
    with assert_raises():
        _ = loads("{")  # unclosed
    print("  '{' (unclosed) -> correctly raises error")
    
    with assert_raises():
        _ = loads("[1,2,")  # incomplete
    print("  '[1,2,' (incomplete) -> correctly raises error")
    
    with assert_raises():
        _ = loads('{"key":}')  # missing value
    print("  '{\"key\":}' (missing value) -> correctly raises error")
    
    with assert_raises():
        _ = loads("undefined")  # not JSON
    print("  'undefined' (not JSON) -> correctly raises error")
    
    with assert_raises():
        _ = loads("{key: 1}")  # unquoted key
    print("  '{key: 1}' (unquoted key) -> correctly raises error")
    
    print()

    # Practical error handling pattern
    print("Practical usage pattern:")

    var user_input = '{"name": "Alice", "age": 30}'

    try:
        var data = loads(user_input)
        print("  Successfully parsed user data")
        print("  Data:", dumps(data))
    except e:
        print("  Failed to parse user data:", e)
        print("  Using default values instead...")

    print()
    print("All error handling tests passed!")
