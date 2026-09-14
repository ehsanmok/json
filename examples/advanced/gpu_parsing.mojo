# GPU-accelerated parsing
#
# Demonstrates ``loads_gpu`` and ``load_gpu`` for
# GPU parsing on NVIDIA (CUDA), AMD (ROCm), and Apple Metal hosts.
#
# Performance
# -----------
# GPU parsing wins on large documents (MB+ sized). For small inputs,
# CPU parsing is typically faster because of kernel launch overhead
# and host <-> device data transfer.

from std.sys import has_accelerator

from json import dumps, Value
from json.gpu import loads, load


def _demo() raises:
    print("1. Basic GPU parsing:")
    var json_str = '{"message": "Hello from GPU!", "count": 42}'
    var data = loads[target="gpu"](json_str)
    print("   Input:", json_str)
    print("   Parsed:", dumps(data))
    print()

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

    print("3. GPU parsing from file:")
    with open("gpu_test.json", "w") as f:
        _ = f.write(nested_json)

    with open("gpu_test.json", "r") as f:
        var file_data = load[target="gpu"](f)
        print("   Loaded from file successfully!")
        var keys = file_data.object_keys()
        print("   Object keys:", ", ".join(keys))
    print()

    print("4. CPU vs GPU comparison:")
    var test_json = '{"x": 1, "y": 2, "z": 3}'
    var cpu_result = loads(test_json)
    var gpu_result = loads[target="gpu"](test_json)
    print("   CPU result:", dumps(cpu_result))
    print("   GPU result:", dumps(gpu_result))
    print()

    print("Note: GPU parsing excels with large JSON documents (MB+ sized).")
    print("For small inputs, CPU parsing is typically faster due to")
    print("GPU kernel launch overhead and data transfer costs.")


def main() raises:
    print("GPU-Accelerated JSON Parsing")
    print("=" * 40)
    print()

    comptime if not has_accelerator():
        print(
            "No GPU detected. Build on a host with a CUDA, ROCm, or Apple"
            " Metal accelerator to run this example."
        )
        return

    _demo()
