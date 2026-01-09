# Example 04: GPU-Accelerated Parsing
#
# Demonstrates: loads[target="gpu"]() and load[target="gpu"]() for GPU parsing
#
# Note: GPU parsing is optimized for large JSON documents. For small inputs,
# CPU parsing may be faster due to GPU kernel launch overhead.

from src import loads, load, dumps, Value


fn main() raises:
    print("GPU-Accelerated JSON Parsing")
    print("=" * 40)
    print()

    # Parse JSON string using GPU
    print("1. Basic GPU parsing:")
    var json_str = '{"message": "Hello from GPU!", "count": 42}'
    var data = loads[target="gpu"](json_str)
    print("   Input:", json_str)
    print("   Parsed:", dumps(data))
    print()

    # GPU parsing with larger data
    print("2. Parsing nested structures:")
    var nested_json = """{
        "users": [
            {"id": 1, "name": "Alice", "scores": [95, 87, 92]},
            {"id": 2, "name": "Bob", "scores": [88, 91, 85]},
            {"id": 3, "name": "Charlie", "scores": [90, 93, 89]}
        ],
        "metadata": {
            "total_users": 3,
            "generated_at": "2024-01-01T00:00:00Z"
        }
    }"""
    var nested_data = loads[target="gpu"](nested_json)
    print("   Parsed successfully!")
    print("   Result:", dumps(nested_data))
    print()

    # GPU file loading
    print("3. GPU parsing from file:")
    # First create a test file
    with open("gpu_test.json", "w") as f:
        _ = f.write(nested_json)

    with open("gpu_test.json", "r") as f:
        var file_data = load[target="gpu"](f)
        print("   Loaded from file successfully!")
        print("   Object keys:", file_data.object_keys().__str__())
    print()

    # Comparing CPU vs GPU (for demonstration)
    print("4. CPU vs GPU comparison:")
    var test_json = '{"x": 1, "y": 2, "z": 3}'

    var cpu_result = loads(test_json)  # CPU is default
    var gpu_result = loads[target="gpu"](test_json)

    print("   CPU result:", dumps(cpu_result))
    print("   GPU result:", dumps(gpu_result))
    print()

    print("Note: GPU parsing excels with large JSON documents (MB+ sized).")
    print("For small inputs, CPU parsing is typically faster due to")
    print("GPU kernel launch overhead and data transfer costs.")
