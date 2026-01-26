# mojson - JSONPath query language
#
# JSONPath is a query language for JSON, similar to XPath for XML.
# Syntax examples:
#   $.store.book[0].title    - Get first book's title
#   $.store.book[*].author   - Get all authors
#   $..author                - Recursive descent, get all authors
#   $.store.book[?@.price<10] - Filter books by price

from collections import List
from .value import Value, Null
from .parser import loads
from .serialize import dumps


fn jsonpath_query(document: Value, path: String) raises -> List[Value]:
    """Query a JSON document using JSONPath syntax.
    
    Supported syntax:
    - $ : Root element
    - .key or ['key'] : Child element
    - [n] : Array index (0-based)
    - [*] : All elements
    - .. : Recursive descent
    - [start:end] : Array slice
    - [?expr] : Filter expression (basic support)
    
    Args:
        document: The JSON document to query
        path: JSONPath expression
    
    Returns:
        List of matching values
    
    Example:
        var doc = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
        var names = jsonpath(doc, "$.users[*].name")
        # [Value("Alice"), Value("Bob")]
    """
    if not path.startswith("$"):
        raise Error("JSONPath must start with $")
    
    var results = List[Value]()
    results.append(document.copy())
    
    var tokens = _tokenize_jsonpath(path)
    
    for i in range(len(tokens)):
        results = _apply_jsonpath_token(results^, tokens[i])
    
    return results^


fn jsonpath_one(document: Value, path: String) raises -> Value:
    """Query and return a single value (first match).
    
    Args:
        document: The JSON document to query
        path: JSONPath expression
    
    Returns:
        First matching value
    
    Raises:
        Error if no matches found
    """
    var results = jsonpath_query(document, path)
    if len(results) == 0:
        raise Error("No match found for JSONPath: " + path)
    return results[0].copy()


struct JSONPathToken(Copyable, Movable):
    """Represents a parsed JSONPath token."""
    var type: Int  # 0=root, 1=child, 2=index, 3=wildcard, 4=recursive, 5=slice, 6=filter
    var value: String
    var start: Int
    var end: Int
    var step: Int
    
    fn __init__(out self, type: Int, value: String = ""):
        self.type = type
        self.value = value
        self.start = 0
        self.end = -1
        self.step = 1
    
    fn __copyinit__(out self, existing: Self):
        self.type = existing.type
        self.value = existing.value
        self.start = existing.start
        self.end = existing.end
        self.step = existing.step
    
    fn __moveinit__(out self, deinit existing: Self):
        self.type = existing.type
        self.value = existing.value^
        self.start = existing.start
        self.end = existing.end
        self.step = existing.step


fn _tokenize_jsonpath(path: String) raises -> List[JSONPathToken]:
    """Parse JSONPath into tokens."""
    var tokens = List[JSONPathToken]()
    var path_bytes = path.as_bytes()
    var n = len(path_bytes)
    var i = 0
    
    # Skip $
    if i < n and path_bytes[i] == ord("$"):
        tokens.append(JSONPathToken(0))  # root
        i += 1
    
    while i < n:
        var c = path_bytes[i]
        
        if c == ord("."):
            i += 1
            if i < n and path_bytes[i] == ord("."):
                # Recursive descent
                tokens.append(JSONPathToken(4))  # recursive
                i += 1
            
            if i < n and path_bytes[i] == ord("*"):
                tokens.append(JSONPathToken(3))  # wildcard
                i += 1
            elif i < n and path_bytes[i] != ord("["):
                # Property name
                var start = i
                while i < n and path_bytes[i] != ord(".") and path_bytes[i] != ord("["):
                    i += 1
                var name = String(path[start:i])
                tokens.append(JSONPathToken(1, name))  # child
        
        elif c == ord("["):
            i += 1
            
            # Skip whitespace
            while i < n and path_bytes[i] == ord(" "):
                i += 1
            
            if i < n and path_bytes[i] == ord("*"):
                tokens.append(JSONPathToken(3))  # wildcard
                i += 1
            elif i < n and path_bytes[i] == ord("?"):
                # Filter expression
                i += 1
                var depth = 1
                var start = i
                while i < n and depth > 0:
                    if path_bytes[i] == ord("["):
                        depth += 1
                    elif path_bytes[i] == ord("]"):
                        depth -= 1
                    i += 1
                var expr = String(path[start:i - 1])
                tokens.append(JSONPathToken(6, expr))  # filter
                continue
            elif i < n and path_bytes[i] == ord("'"):
                # Quoted property ['key']
                i += 1
                var start = i
                while i < n and path_bytes[i] != ord("'"):
                    i += 1
                var name = String(path[start:i])
                tokens.append(JSONPathToken(1, name))  # child
                i += 1
            elif i < n and path_bytes[i] == ord('"'):
                # Double quoted property ["key"]
                i += 1
                var start = i
                while i < n and path_bytes[i] != ord('"'):
                    i += 1
                var name = String(path[start:i])
                tokens.append(JSONPathToken(1, name))  # child
                i += 1
            elif i < n:
                # Index, slice, or negative index
                var start = i
                while i < n and path_bytes[i] != ord("]"):
                    i += 1
                var content = String(path[start:i])
                
                if content.find(":") >= 0:
                    # Slice
                    var token = _parse_slice(content)
                    tokens.append(token^)
                else:
                    # Index
                    tokens.append(JSONPathToken(2, content))  # index
            
            # Skip closing ]
            while i < n and path_bytes[i] != ord("]"):
                i += 1
            if i < n:
                i += 1
        
        else:
            i += 1
    
    return tokens^


fn _parse_slice(content: String) raises -> JSONPathToken:
    """Parse a slice expression like 0:5 or 1:10:2."""
    var token = JSONPathToken(5)  # slice
    var parts = List[String]()
    var content_bytes = content.as_bytes()
    var start = 0
    
    for i in range(len(content_bytes) + 1):
        if i == len(content_bytes) or content_bytes[i] == ord(":"):
            parts.append(String(content[start:i]))
            start = i + 1
    
    if len(parts) >= 1 and len(parts[0]) > 0:
        token.start = atol(parts[0])
    
    if len(parts) >= 2 and len(parts[1]) > 0:
        token.end = atol(parts[1])
    
    if len(parts) >= 3 and len(parts[2]) > 0:
        token.step = atol(parts[2])
    
    return token^


fn _apply_jsonpath_token(var values: List[Value], ref token: JSONPathToken) raises -> List[Value]:
    """Apply a single JSONPath token to a list of values."""
    var results = List[Value]()
    
    if token.type == 0:  # root
        return values^  # Root is already in values
    
    elif token.type == 1:  # child
        for i in range(len(values)):
            var v = values[i].copy()
            if v.is_object():
                try:
                    results.append(v[token.value].copy())
                except:
                    pass  # Key not found, skip
    
    elif token.type == 2:  # index
        var idx = atol(token.value)
        for i in range(len(values)):
            var v = values[i].copy()
            if v.is_array():
                var count = v.array_count()
                # Handle negative index
                if idx < 0:
                    idx = count + idx
                if idx >= 0 and idx < count:
                    results.append(v[idx].copy())
    
    elif token.type == 3:  # wildcard
        for i in range(len(values)):
            var v = values[i].copy()
            if v.is_array():
                var items = v.array_items()
                for j in range(len(items)):
                    results.append(items[j].copy())
            elif v.is_object():
                var items = v.object_items()
                for j in range(len(items)):
                    results.append(items[j][1].copy())
    
    elif token.type == 4:  # recursive descent
        for i in range(len(values)):
            _recursive_collect(values[i], results)
    
    elif token.type == 5:  # slice
        for i in range(len(values)):
            var v = values[i].copy()
            if v.is_array():
                var items = v.array_items()
                var count = len(items)
                var start = token.start
                var end = token.end
                var step = token.step
                
                if start < 0:
                    start = count + start
                if end < 0:
                    end = count + end
                elif end == -1:
                    end = count
                
                start = max(0, min(start, count))
                end = max(0, min(end, count))
                
                var j = start
                while j < end:
                    results.append(items[j].copy())
                    j += step
    
    elif token.type == 6:  # filter
        for i in range(len(values)):
            var v = values[i].copy()
            if v.is_array():
                var items = v.array_items()
                for j in range(len(items)):
                    if _evaluate_filter(items[j], token.value):
                        results.append(items[j].copy())
    
    return results^


fn _recursive_collect(value: Value, mut results: List[Value]):
    """Recursively collect all values (for ..)."""
    results.append(value.copy())
    
    if value.is_array():
        try:
            var items = value.array_items()
            for i in range(len(items)):
                _recursive_collect(items[i], results)
        except:
            pass
    elif value.is_object():
        try:
            var items = value.object_items()
            for i in range(len(items)):
                _recursive_collect(items[i][1], results)
        except:
            pass


fn _evaluate_filter(value: Value, expr: String) -> Bool:
    """Evaluate a basic filter expression.
    
    Supports:
    - @.field == value
    - @.field != value
    - @.field < value
    - @.field > value
    - @.field <= value
    - @.field >= value
    """
    var expr_bytes = expr.as_bytes()
    var n = len(expr_bytes)
    
    # Find operator
    var op_start = -1
    var op_end = -1
    var op = String()
    
    for i in range(n - 1):
        if expr_bytes[i] == ord("=") and expr_bytes[i + 1] == ord("="):
            op = "=="
            op_start = i
            op_end = i + 2
            break
        elif expr_bytes[i] == ord("!") and expr_bytes[i + 1] == ord("="):
            op = "!="
            op_start = i
            op_end = i + 2
            break
        elif expr_bytes[i] == ord("<") and expr_bytes[i + 1] == ord("="):
            op = "<="
            op_start = i
            op_end = i + 2
            break
        elif expr_bytes[i] == ord(">") and expr_bytes[i + 1] == ord("="):
            op = ">="
            op_start = i
            op_end = i + 2
            break
        elif expr_bytes[i] == ord("<"):
            op = "<"
            op_start = i
            op_end = i + 1
            break
        elif expr_bytes[i] == ord(">"):
            op = ">"
            op_start = i
            op_end = i + 1
            break
    
    if op_start < 0:
        return False
    
    # Extract field path (after @)
    var left = String(expr[:op_start])
    var right = String(expr[op_end:])
    
    # Trim whitespace
    left = _trim(left)
    right = _trim(right)
    
    # Extract field name from @.field
    if not left.startswith("@."):
        return False
    var field = String(left[2:])
    
    # Get field value
    var field_value: Value
    try:
        field_value = value[field].copy()
    except:
        return False
    
    # Parse right side as a literal
    var compare_value: Value
    try:
        compare_value = loads[target="cpu"](right)
    except:
        # Try as raw string
        compare_value = Value(right)
    
    # Compare
    if op == "==":
        return _values_equal_basic(field_value, compare_value)
    elif op == "!=":
        return not _values_equal_basic(field_value, compare_value)
    elif op == "<":
        return _compare_values(field_value, compare_value) < 0
    elif op == ">":
        return _compare_values(field_value, compare_value) > 0
    elif op == "<=":
        return _compare_values(field_value, compare_value) <= 0
    elif op == ">=":
        return _compare_values(field_value, compare_value) >= 0
    
    return False


fn _values_equal_basic(a: Value, b: Value) -> Bool:
    """Basic value equality check."""
    if a.is_null() and b.is_null():
        return True
    if a.is_bool() and b.is_bool():
        return a.bool_value() == b.bool_value()
    if a.is_int() and b.is_int():
        return a.int_value() == b.int_value()
    if a.is_float() and b.is_float():
        return a.float_value() == b.float_value()
    if a.is_string() and b.is_string():
        return a.string_value() == b.string_value()
    # Compare numbers across types
    if (a.is_int() or a.is_float()) and (b.is_int() or b.is_float()):
        var af = a.float_value() if a.is_float() else Float64(a.int_value())
        var bf = b.float_value() if b.is_float() else Float64(b.int_value())
        return af == bf
    return False


fn _compare_values(a: Value, b: Value) -> Int:
    """Compare two values. Returns -1, 0, or 1."""
    if (a.is_int() or a.is_float()) and (b.is_int() or b.is_float()):
        var af = a.float_value() if a.is_float() else Float64(a.int_value())
        var bf = b.float_value() if b.is_float() else Float64(b.int_value())
        if af < bf:
            return -1
        elif af > bf:
            return 1
        return 0
    if a.is_string() and b.is_string():
        var as_ = a.string_value()
        var bs = b.string_value()
        if as_ < bs:
            return -1
        elif as_ > bs:
            return 1
        return 0
    return 0


fn _trim(s: String) -> String:
    """Trim whitespace from both ends."""
    var s_bytes = s.as_bytes()
    var start = 0
    var end = len(s_bytes)
    
    while start < end and (s_bytes[start] == ord(" ") or s_bytes[start] == ord("\t")):
        start += 1
    while end > start and (s_bytes[end - 1] == ord(" ") or s_bytes[end - 1] == ord("\t")):
        end -= 1
    
    return String(s[start:end])
