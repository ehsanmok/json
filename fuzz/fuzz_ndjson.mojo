"""Fuzz harness: ``loads[format='ndjson']`` line splitter.

NDJSON parsing splits the input on ``\n`` and feeds each non-empty
line into the regular JSON parser. The line splitter has its own
edge cases (``\r\n``, trailing newline, blank lines, embedded
newlines inside strings) that compound with the parser's own.

Run:
    pixi run -e fuzz fuzz-ndjson
"""

from mozz import fuzz, FuzzConfig

from json import loads


def target(data: List[UInt8]) raises:
    var s = String()
    for i in range(len(data)):
        s += chr(Int(data[i]))

    try:
        _ = loads[format="ndjson"](s)
    except:
        pass


def _bytes(s: StringLiteral) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        out.append(b[i])
    return out^


def main() raises:
    print("[mozz] fuzzing loads[format='ndjson']...")

    var seeds = List[List[UInt8]]()

    seeds.append(_bytes('{"a":1}\n{"b":2}\n{"c":3}'))
    seeds.append(_bytes('{"a":1}\r\n{"b":2}\r\n'))
    seeds.append(_bytes('{"a":1}\n\n{"b":2}\n'))
    seeds.append(_bytes('{"k":"line\\nwith\\nescapes"}'))
    seeds.append(_bytes("\n\n\n"))
    seeds.append(_bytes(""))
    seeds.append(_bytes('{"a":1'))
    seeds.append(_bytes('{}\n[}\n{"c":3}'))
    seeds.append(_bytes('"hello"\n"world"\n42\n'))

    fuzz(
        target,
        FuzzConfig(
            max_runs=200_000,
            seed=0,
            verbose=True,
            crash_dir=".mozz_crashes/ndjson",
            corpus_dir="fuzz/corpus/ndjson",
            max_input_len=4096,
        ),
        seeds,
    )
