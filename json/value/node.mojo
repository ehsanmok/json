# json - The owned JSON node.
#
# `OwnedValue` is the structured, *mutable* tree representation of a JSON
# value: an array keeps its children in `array_val`, an object keeps
# parallel `object_keys` / `object_values`. Mutating it is a direct
# in-place edit, so growing a tree costs O(1) amortized per node.
#
# It lives in its own module because both directions need it:
#   - `value.mojo` embeds one in `Value` as the write representation,
#   - `owned.mojo` converts between it and the tape-backed `Document`.
# Keeping it here is what breaks the value <-> owned import cycle. This
# module therefore depends on nothing else in the package.
#
# A default `OwnedValue` performs **no heap allocation**: `kind` is a
# scalar, `str_val` is an empty `String`, and the three `List` fields are
# empty (null data pointer, zero capacity). That is what makes scalar and
# empty-container `Value`s allocation-free.

from std.collections import List


# Node kind tags. These mirror the tape tags in `document.mojo` but are
# deliberately a separate, dense 0..6 space: the owned tree has no
# STRING vs STRING_OWNED distinction and no KEY entries.
comptime OWNED_NULL = 0
comptime OWNED_BOOL = 1
comptime OWNED_INT = 2
comptime OWNED_FLOAT = 3
comptime OWNED_STRING = 4
comptime OWNED_ARRAY = 5
comptime OWNED_OBJECT = 6


struct OwnedValue(Copyable, Deinitable, Movable):
    """Structured tree representation of a JSON value.

    Unlike `Value`, an `OwnedValue` for an array stores its children in
    `array_val: List[OwnedValue]` rather than as a raw JSON substring,
    so a mutation at any nesting level can be applied in place and a
    fresh JSON serialization can be produced after the fact.
    """

    # One of the OWNED_* constants above.
    var kind: Int
    var bool_val: Bool
    var int_val: Int64
    var float_val: Float64
    var str_val: String
    var array_val: List[OwnedValue]
    var object_keys: List[String]
    var object_values: List[OwnedValue]

    def __init__(out self):
        self.kind = OWNED_NULL
        self.bool_val = False
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = String()
        self.array_val = List[OwnedValue]()
        self.object_keys = List[String]()
        self.object_values = List[OwnedValue]()

    def __deinit__(deinit self):
        # Explicit (no-op) deinit breaks the self-referential
        # `List[OwnedValue]` field's `Deinitable` completeness check;
        # fields are still destroyed automatically after this runs.
        pass

    def copy(self) -> Self:
        var out = Self()
        out.kind = self.kind
        out.bool_val = self.bool_val
        out.int_val = self.int_val
        out.float_val = self.float_val
        out.str_val = self.str_val
        out.array_val = self.array_val.copy()
        out.object_keys = self.object_keys.copy()
        out.object_values = self.object_values.copy()
        return out^

    @staticmethod
    def make_null() -> Self:
        return Self()

    @staticmethod
    def make_bool(b: Bool) -> Self:
        var v = Self()
        v.kind = OWNED_BOOL
        v.bool_val = b
        return v^

    @staticmethod
    def make_int(i: Int64) -> Self:
        var v = Self()
        v.kind = OWNED_INT
        v.int_val = i
        return v^

    @staticmethod
    def make_float(f: Float64) -> Self:
        var v = Self()
        v.kind = OWNED_FLOAT
        v.float_val = f
        return v^

    @staticmethod
    def make_string(var s: String) -> Self:
        var v = Self()
        v.kind = OWNED_STRING
        v.str_val = s^
        return v^

    @staticmethod
    def make_array(var items: List[OwnedValue]) -> Self:
        var v = Self()
        v.kind = OWNED_ARRAY
        v.array_val = items^
        return v^

    @staticmethod
    def make_object(
        var keys: List[String], var values: List[OwnedValue]
    ) -> Self:
        var v = Self()
        v.kind = OWNED_OBJECT
        v.object_keys = keys^
        v.object_values = values^
        return v^

    def find_key(self, key: String) -> Int:
        """Index of `key` in `object_keys`, or -1.

        A linear scan, deliberately. Objects in practice hold a handful of
        keys, where scanning a contiguous `List` beats hashing; and giving
        every node a `Dict` would put a heap allocation back into the
        default-constructed node, which is exactly what keeps scalars and
        empty containers allocation-free.
        """
        for i in range(len(self.object_keys)):
            if self.object_keys[i] == key:
                return i
        return -1

    def set_key(mut self, var key: String, var value: OwnedValue):
        """Insert or overwrite `key`. O(1) amortized for a new key."""
        var pos = self.find_key(key)
        if pos >= 0:
            self.object_values[pos] = value^
        else:
            self.object_keys.append(key^)
            self.object_values.append(value^)

    def push(mut self, var value: OwnedValue):
        """Append to an array node. O(1) amortized."""
        self.array_val.append(value^)
