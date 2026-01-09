# cuJSON Types - Data structures for JSON parsing
from collections import List


struct JSONInput(Movable):
    """Input structure for JSON data."""

    var data: List[UInt8]

    fn __init__(out self, var data: List[UInt8]):
        self.data = data^

    fn __copyinit__(out self, other: Self):
        self.data = other.data.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.data = other.data^


struct JSONResult(Movable):
    """Result structure containing parsed JSON information.

    - structural: Array containing byte positions of all JSON structural characters
    - pair_pos: Array storing the closing structural character's index for each opening
    - depth: Maximum nesting depth encountered in the parsed JSON
    """

    var structural: List[Int32]
    var pair_pos: List[Int32]
    var depth: Int
    var file_size: Int

    fn __init__(out self):
        self.structural = List[Int32]()
        self.pair_pos = List[Int32]()
        self.depth = 0
        self.file_size = 0

    fn __copyinit__(out self, other: Self):
        self.structural = other.structural.copy()
        self.pair_pos = other.pair_pos.copy()
        self.depth = other.depth
        self.file_size = other.file_size

    fn __moveinit__(out self, deinit other: Self):
        self.structural = other.structural^
        self.pair_pos = other.pair_pos^
        self.depth = other.depth
        self.file_size = other.file_size

    fn total_result_size(self) -> Int:
        return len(self.structural)


# Structural character constants
comptime CHAR_OPEN_BRACE: UInt8 = 0x7B  # '{'
comptime CHAR_CLOSE_BRACE: UInt8 = 0x7D  # '}'
comptime CHAR_OPEN_BRACKET: UInt8 = 0x5B  # '['
comptime CHAR_CLOSE_BRACKET: UInt8 = 0x5D  # ']'
comptime CHAR_COLON: UInt8 = 0x3A  # ':'
comptime CHAR_COMMA: UInt8 = 0x2C  # ','
comptime CHAR_QUOTE: UInt8 = 0x22  # '"'
comptime CHAR_BACKSLASH: UInt8 = 0x5C  # '\'
comptime CHAR_NEWLINE: UInt8 = 0x0A  # '\n'


fn is_structural_char(c: UInt8) -> Bool:
    """Check if a character is a JSON structural character."""
    return (
        c == CHAR_OPEN_BRACE
        or c == CHAR_CLOSE_BRACE
        or c == CHAR_OPEN_BRACKET
        or c == CHAR_CLOSE_BRACKET
        or c == CHAR_COLON
        or c == CHAR_COMMA
    )


fn is_open_bracket(c: UInt8) -> Bool:
    """Check if character is an opening bracket."""
    return c == CHAR_OPEN_BRACE or c == CHAR_OPEN_BRACKET


fn is_close_bracket(c: UInt8) -> Bool:
    """Check if character is a closing bracket."""
    return c == CHAR_CLOSE_BRACE or c == CHAR_CLOSE_BRACKET
