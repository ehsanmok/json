# mojson - Value type for JSON values

from collections import List


struct Null(Stringable, Writable):
    """Represents JSON null."""

    fn __init__(out self):
        pass

    fn __str__(self) -> String:
        return "null"

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("null")


struct Value(Copyable, Movable, Stringable, Writable):
    """A JSON value that can hold null, bool, int, float, string, array, or object.
    """

    var _type: Int  # 0=null, 1=bool, 2=int, 3=float, 4=string, 5=array, 6=object
    var _bool: Bool
    var _int: Int64
    var _float: Float64
    var _string: String
    var _raw: String  # Raw JSON for arrays/objects
    var _keys: List[String]  # Object keys
    var _count: Int  # Array/object element count

    fn __init__(out self, null: Null):
        self._type = 0
        self._bool = False
        self._int = 0
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0

    fn __init__(out self, none: NoneType):
        self._type = 0
        self._bool = False
        self._int = 0
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0

    fn __init__(out self, b: Bool):
        self._type = 1
        self._bool = b
        self._int = 0
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0

    fn __init__(out self, i: Int):
        self._type = 2
        self._bool = False
        self._int = Int64(i)
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0

    fn __init__(out self, i: Int64):
        self._type = 2
        self._bool = False
        self._int = i
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0

    fn __init__(out self, f: Float64):
        self._type = 3
        self._bool = False
        self._int = 0
        self._float = f
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0

    fn __init__(out self, s: String):
        self._type = 4
        self._bool = False
        self._int = 0
        self._float = 0.0
        self._string = s
        self._raw = String()
        self._keys = List[String]()
        self._count = 0

    # Type checking
    fn is_null(self) -> Bool:
        return self._type == 0

    fn is_bool(self) -> Bool:
        return self._type == 1

    fn is_int(self) -> Bool:
        return self._type == 2

    fn is_float(self) -> Bool:
        return self._type == 3

    fn is_string(self) -> Bool:
        return self._type == 4

    fn is_array(self) -> Bool:
        return self._type == 5

    fn is_object(self) -> Bool:
        return self._type == 6

    fn is_number(self) -> Bool:
        return self._type == 2 or self._type == 3

    # Value extraction
    fn bool_value(self) -> Bool:
        return self._bool

    fn int_value(self) -> Int64:
        return self._int

    fn float_value(self) -> Float64:
        return self._float

    fn string_value(self) -> String:
        return self._string

    fn raw_json(self) -> String:
        return self._raw

    fn array_count(self) -> Int:
        return self._count

    fn object_keys(self) -> List[String]:
        return self._keys.copy()

    fn object_count(self) -> Int:
        return self._count

    # Stringable
    fn __str__(self) -> String:
        if self._type == 0:
            return "null"
        elif self._type == 1:
            return "true" if self._bool else "false"
        elif self._type == 2:
            return String(self._int)
        elif self._type == 3:
            return String(self._float)
        elif self._type == 4:
            return '"' + self._string + '"'
        elif self._type == 5 or self._type == 6:
            return self._raw
        return "unknown"

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    # Equality
    fn __eq__(self, other: Value) -> Bool:
        if self._type != other._type:
            return False
        if self._type == 0:
            return True
        elif self._type == 1:
            return self._bool == other._bool
        elif self._type == 2:
            return self._int == other._int
        elif self._type == 3:
            return self._float == other._float
        elif self._type == 4:
            return self._string == other._string
        elif self._type == 5 or self._type == 6:
            return self._raw == other._raw
        return False

    fn __ne__(self, other: Value) -> Bool:
        return not self.__eq__(other)


fn make_array_value(raw: String, count: Int) -> Value:
    """Create an array Value from raw JSON."""
    var v = Value(Null())
    v._type = 5
    v._raw = raw
    v._count = count
    return v^


fn make_object_value(raw: String, var keys: List[String]) -> Value:
    """Create an object Value from raw JSON and keys."""
    var v = Value(Null())
    v._type = 6
    v._raw = raw
    v._count = len(keys)
    v._keys = keys^
    return v^
