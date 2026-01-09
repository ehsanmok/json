# Example 02: File Operations
#
# Demonstrates: load() and dump() for file-based JSON operations

from src import load, dump, loads, dumps, Value


fn main() raises:
    # Create a sample JSON file
    var sample_data = loads(
        '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "total": 2}'
    )

    # Write JSON to file using dump()
    print("Writing JSON to 'output.json'...")
    with open("output.json", "w") as f:
        dump(sample_data, f)
    print("  Done!")
    print()

    # Read JSON from file using load()
    print("Reading JSON from 'output.json'...")
    with open("output.json", "r") as f:
        var loaded_data = load(f)
        print("  Loaded:", dumps(loaded_data))
    print()

    # Process a larger JSON structure
    var config = loads(
        '{"app": {"name": "MyApp", "version": "1.0.0", "debug": false}, "database": {"host": "localhost", "port": 5432, "name": "mydb"}, "features": ["auth", "logging", "cache"]}'
    )

    print("Writing config to 'config.json'...")
    with open("config.json", "w") as f:
        dump(config, f)

    with open("config.json", "r") as f:
        var loaded_config = load(f)
        print("  Loaded config:", dumps(loaded_config))
    print()

    # Roundtrip: load -> modify -> dump
    print("File roundtrip demonstration:")
    var data = loads('{"counter": 0, "name": "test"}')

    # Write initial data
    with open("counter.json", "w") as f:
        dump(data, f)
    print("  Wrote initial data")

    # Read it back
    with open("counter.json", "r") as f:
        var read_data = load(f)
        print("  Read back:", dumps(read_data))
