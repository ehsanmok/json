# mojson - Streaming JSON parsing for large files
#
# Process JSON files larger than memory by reading in chunks.
# Best suited for NDJSON (newline-delimited JSON) files.

from collections import List
from .value import Value
from .parser import loads


struct StreamingParser:
    """Streaming JSON parser for processing large files.
    
    Reads files in chunks and yields complete JSON values
    as they become available. Ideal for NDJSON files where
    each line is independent.
    
    Example:
        var parser = StreamingParser("large_file.ndjson")
        while parser.has_next():
            var value = parser.next()
            process(value)
        parser.close()
    """
    var _file: FileHandle
    var _buffer: String
    var _chunk_size: Int
    var _eof: Bool
    var _closed: Bool
    
    fn __init__(out self, path: String, chunk_size: Int = 65536) raises:
        """Open a file for streaming parsing.
        
        Args:
            path: Path to the JSON file
            chunk_size: Size of chunks to read (default 64KB)
        """
        self._file = open(path, "r")
        self._buffer = String()
        self._chunk_size = chunk_size
        self._eof = False
        self._closed = False
    
    fn close(mut self) raises:
        """Close the file handle."""
        if not self._closed:
            self._file.close()
            self._closed = True
    
    fn has_next(mut self) -> Bool:
        """Check if there are more JSON values to read.
        
        Returns:
            True if more values are available
        """
        if self._closed:
            return False
        
        # Check if buffer has a complete line
        if self._has_complete_line():
            return True
        
        # Try to read more data
        if not self._eof:
            try:
                self._read_chunk()
            except:
                return False
        
        return self._has_complete_line()
    
    fn next(mut self) raises -> Value:
        """Read and parse the next JSON value.
        
        Returns:
            The next parsed Value
        
        Raises:
            Error if no more values or parse error
        """
        if self._closed:
            raise Error("StreamingParser is closed")
        
        # Ensure we have data
        while not self._has_complete_line() and not self._eof:
            self._read_chunk()
        
        # Extract and parse the next line
        var line = self._extract_line()
        
        # Skip empty lines
        while _is_empty_line(line) and (self._has_complete_line() or not self._eof):
            if not self._has_complete_line() and not self._eof:
                self._read_chunk()
            if self._has_complete_line():
                line = self._extract_line()
            else:
                break
        
        if _is_empty_line(line):
            raise Error("No more JSON values")
        
        return loads[target="cpu"](line)
    
    fn _read_chunk(mut self) raises:
        """Read a chunk of data from the file."""
        var chunk = self._file.read(self._chunk_size)
        if len(chunk) == 0:
            self._eof = True
        else:
            self._buffer += chunk
    
    fn _has_complete_line(self) -> Bool:
        """Check if buffer contains a complete line."""
        var buffer_bytes = self._buffer.as_bytes()
        for i in range(len(buffer_bytes)):
            if buffer_bytes[i] == ord("\n"):
                return True
        # If EOF and buffer has content, treat it as complete
        if self._eof and len(self._buffer) > 0:
            return True
        return False
    
    fn _extract_line(mut self) raises -> String:
        """Extract the next line from the buffer."""
        var buffer_bytes = self._buffer.as_bytes()
        var n = len(buffer_bytes)
        var line_end = -1
        
        for i in range(n):
            if buffer_bytes[i] == ord("\n"):
                line_end = i
                break
        
        var line: String
        if line_end >= 0:
            # Extract line (handle \r\n)
            var end = line_end
            if end > 0 and buffer_bytes[end - 1] == ord("\r"):
                end -= 1
            line = String(self._buffer[:end])
            self._buffer = String(self._buffer[line_end + 1:])
        else:
            # No newline found, return entire buffer (EOF case)
            line = self._buffer
            self._buffer = String()
        
        return line^


fn stream_ndjson(
    path: String,
    chunk_size: Int = 65536,
) raises -> StreamingParser:
    """Create a streaming parser for an NDJSON file.
    
    Args:
        path: Path to the NDJSON file
        chunk_size: Size of chunks to read (default 64KB)
    
    Returns:
        StreamingParser for iterating over values
    
    Example:
        var parser = stream_ndjson("logs.ndjson")
        while parser.has_next():
            var entry = parser.next()
            if entry["level"].string_value() == "error":
                print(entry)
        parser.close()
    """
    return StreamingParser(path, chunk_size)


fn stream_json_array(
    path: String,
    chunk_size: Int = 65536,
) raises -> ArrayStreamingParser:
    """Create a streaming parser for a JSON array file.
    
    Parses files containing a single JSON array, yielding
    each element one at a time.
    
    Args:
        path: Path to the JSON file containing an array
        chunk_size: Size of chunks to read (default 64KB)
    
    Returns:
        ArrayStreamingParser for iterating over elements
    
    Example:
        var parser = stream_json_array("users.json")  # [{"name":"Alice"},...]
        while parser.has_next():
            var user = parser.next()
            print(user["name"].string_value())
        parser.close()
    """
    return ArrayStreamingParser(path, chunk_size)


struct ArrayStreamingParser:
    """Streaming parser for JSON array files.
    
    Parses a file containing a single JSON array and yields
    elements one at a time without loading the entire array.
    """
    var _file: FileHandle
    var _buffer: String
    var _chunk_size: Int
    var _eof: Bool
    var _closed: Bool
    var _started: Bool
    var _depth: Int
    
    fn __init__(out self, path: String, chunk_size: Int = 65536) raises:
        """Open a JSON array file for streaming.
        
        Args:
            path: Path to the JSON file
            chunk_size: Size of chunks to read
        """
        self._file = open(path, "r")
        self._buffer = String()
        self._chunk_size = chunk_size
        self._eof = False
        self._closed = False
        self._started = False
        self._depth = 0
    
    fn close(mut self) raises:
        """Close the file handle."""
        if not self._closed:
            self._file.close()
            self._closed = True
    
    fn has_next(mut self) -> Bool:
        """Check if there are more array elements."""
        if self._closed:
            return False
        
        # Skip to array start if not started
        if not self._started:
            try:
                self._skip_to_array_start()
            except:
                return False
        
        # Check if we can find a complete element
        return self._has_complete_element()
    
    fn next(mut self) raises -> Value:
        """Read and parse the next array element."""
        if self._closed:
            raise Error("ArrayStreamingParser is closed")
        
        if not self._started:
            self._skip_to_array_start()
        
        # Read until we have a complete element
        while not self._has_complete_element() and not self._eof:
            self._read_chunk()
        
        return self._extract_element()
    
    fn _read_chunk(mut self) raises:
        """Read a chunk of data."""
        var chunk = self._file.read(self._chunk_size)
        if len(chunk) == 0:
            self._eof = True
        else:
            self._buffer += chunk
    
    fn _skip_to_array_start(mut self) raises:
        """Skip whitespace and find the opening bracket."""
        while True:
            var buffer_bytes = self._buffer.as_bytes()
            for i in range(len(buffer_bytes)):
                var c = buffer_bytes[i]
                if c == ord("["):
                    self._buffer = String(self._buffer[i + 1:])
                    self._started = True
                    self._depth = 1
                    return
                elif c != ord(" ") and c != ord("\t") and c != ord("\n") and c != ord("\r"):
                    raise Error("Expected JSON array")
            
            if self._eof:
                raise Error("Expected JSON array, got EOF")
            
            self._read_chunk()
    
    fn _has_complete_element(mut self) -> Bool:
        """Check if buffer contains a complete element."""
        var buffer_bytes = self._buffer.as_bytes()
        var n = len(buffer_bytes)
        var depth = 0
        var in_string = False
        var escaped = False
        var found_start = False
        
        for i in range(n):
            var c = buffer_bytes[i]
            
            if escaped:
                escaped = False
                continue
            
            if c == ord("\\") and in_string:
                escaped = True
                continue
            
            if c == ord('"'):
                in_string = not in_string
                if not found_start:
                    found_start = True
                continue
            
            if in_string:
                continue
            
            # Skip leading whitespace
            if not found_start:
                if c == ord(" ") or c == ord("\t") or c == ord("\n") or c == ord("\r") or c == ord(","):
                    continue
                if c == ord("]"):
                    return False  # End of array
                found_start = True
            
            if c == ord("{") or c == ord("["):
                depth += 1
            elif c == ord("}") or c == ord("]"):
                if depth > 0:
                    depth -= 1
                if depth == 0 and found_start:
                    return True
            elif c == ord(",") and depth == 0:
                return True
        
        # For primitives without brackets
        if found_start and depth == 0 and (self._eof or self._buffer.find(",") >= 0 or self._buffer.find("]") >= 0):
            return True
        
        return False
    
    fn _extract_element(mut self) raises -> Value:
        """Extract and parse the next element."""
        var buffer_bytes = self._buffer.as_bytes()
        var n = len(buffer_bytes)
        var depth = 0
        var in_string = False
        var escaped = False
        var start = -1
        var end = -1
        
        for i in range(n):
            var c = buffer_bytes[i]
            
            if escaped:
                escaped = False
                continue
            
            if c == ord("\\") and in_string:
                escaped = True
                continue
            
            if c == ord('"'):
                in_string = not in_string
                if start < 0:
                    start = i
                continue
            
            if in_string:
                continue
            
            # Skip leading whitespace/commas
            if start < 0:
                if c == ord(" ") or c == ord("\t") or c == ord("\n") or c == ord("\r") or c == ord(","):
                    continue
                if c == ord("]"):
                    raise Error("No more array elements")
                start = i
            
            if c == ord("{") or c == ord("["):
                depth += 1
            elif c == ord("}"):
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            elif c == ord("]"):
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                else:
                    # End of outer array, element ends before ]
                    end = i
                    break
            elif c == ord(",") and depth == 0:
                end = i
                break
        
        if start < 0:
            raise Error("No more array elements")
        
        if end < 0:
            end = n
        
        var element_str = String(self._buffer[start:end])
        self._buffer = String(self._buffer[end:])
        
        return loads[target="cpu"](element_str)


fn _is_empty_line(s: String) -> Bool:
    """Check if string is empty or whitespace-only."""
    var s_bytes = s.as_bytes()
    for i in range(len(s_bytes)):
        var c = s_bytes[i]
        if c != ord(" ") and c != ord("\t") and c != ord("\r") and c != ord("\n"):
            return False
    return True
