# Example 07: NDJSON (Newline-Delimited JSON)
#
# NDJSON is a format where each line is a separate JSON value.
# Common for log files, data streams, and large datasets.

from src import loads, dumps, load, Value


fn main() raises:
    print("NDJSON Examples")
    print("=" * 50)
    print()
    
    # ==========================================================
    # 1. Parse NDJSON string
    # ==========================================================
    print("1. Parse NDJSON string:")
    
    var ndjson_str = '{"id":1,"name":"Alice"}\n{"id":2,"name":"Bob"}\n{"id":3,"name":"Charlie"}'
    
    # Use format="ndjson" for strings (can't auto-detect without filename)
    var values = loads[format="ndjson"](ndjson_str)
    print("   Parsed", len(values), "records")
    
    for i in range(len(values)):
        print("   Record", i + 1, ":", values[i]["name"].string_value())
    print()
    
    # ==========================================================
    # 2. Load NDJSON file (auto-detected from .ndjson extension)
    # ==========================================================
    print("2. Load NDJSON file:")
    
    # Create a test file
    var f = open("example_data.ndjson", "w")
    f.write('{"event":"login","user":"alice"}\n')
    f.write('{"event":"purchase","user":"bob","amount":99.99}\n')
    f.write('{"event":"logout","user":"alice"}\n')
    f.close()
    
    # Auto-detects .ndjson and returns array Value
    var events = load("example_data.ndjson")
    print("   Loaded", events.array_count(), "events")
    print("   Is array?", events.is_array())
    
    var items = events.array_items()
    for i in range(len(items)):
        print("   Event:", items[i]["event"].string_value(), "by", items[i]["user"].string_value())
    print()
    
    # ==========================================================
    # 3. GPU-accelerated NDJSON parsing
    # ==========================================================
    print("3. GPU-accelerated NDJSON:")
    
    # For strings
    var gpu_values = loads[target="gpu", format="ndjson"](ndjson_str)
    print("   GPU parsed", len(gpu_values), "records from string")
    
    # For files
    var gpu_events = load[target="gpu"]("example_data.ndjson")
    print("   GPU parsed", gpu_events.array_count(), "records from file")
    print()
    
    # ==========================================================
    # 4. Serialize to NDJSON
    # ==========================================================
    print("4. Serialize to NDJSON:")
    
    var records = List[Value]()
    records.append(loads('{"type":"A","value":100}'))
    records.append(loads('{"type":"B","value":200}'))
    records.append(loads('{"type":"C","value":300}'))
    
    var output = dumps[format="ndjson"](records)
    print("   Output:")
    print("   ", output.replace("\n", "\n    "))
    print()
    
    # ==========================================================
    # 5. Streaming for large files (CPU only, memory efficient)
    # ==========================================================
    print("5. Streaming large files:")
    
    var parser = load[streaming=True]("example_data.ndjson")
    var count = 0
    while parser.has_next():
        var item = parser.next()
        count += 1
        # Process each item without loading entire file
    parser.close()
    print("   Streamed", count, "records (memory efficient)")
    print()
    
    # Cleanup
    import os
    os.remove("example_data.ndjson")
    
    print("Done!")
