# Example 05: Error Handling
#
# Demonstrates: Handling JSON parse errors using try/except

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

    # Invalid JSON examples
    print("Invalid JSON:")
    print("  '' (empty) ->", parse_safely(""))
    print("  '{' (unclosed) ->", parse_safely("{"))
    print("  '[1,2,' (incomplete) ->", parse_safely("[1,2,"))
    print("  '{\"key\":}' (missing value) ->", parse_safely('{"key":}'))
    print("  'undefined' (not JSON) ->", parse_safely("undefined"))
    print("  '{key: 1}' (unquoted key) ->", parse_safely("{key: 1}"))
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

    # Processing a list of JSON strings
    print("Batch processing with error recovery:")
    var json_inputs = List[String]()
    json_inputs.append('{"id": 1, "value": "a"}')
    json_inputs.append('{"id": 2, "value": "b"}')
    json_inputs.append('{"id": 3, broken}')  # Invalid
    json_inputs.append('{"id": 4, "value": "d"}')

    var successful = 0
    var failed = 0

    for i in range(len(json_inputs)):
        try:
            var parsed = loads(json_inputs[i])
            successful += 1
            print("  [", i, "] OK:", dumps(parsed))
        except:
            failed += 1
            print("  [", i, "] FAILED: Invalid JSON")

    print()
    print("  Summary:", successful, "succeeded,", failed, "failed")
