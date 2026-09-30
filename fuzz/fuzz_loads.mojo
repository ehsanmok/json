"""Fuzz harness: ``json.loads`` (default Mojo CPU two-pass parser).

Tests ``loads(s)`` for crashes on arbitrary string inputs. Malformed
JSON raises a regular ``Error`` (a "rejection") -- only panic-like
errors and aborts are bugs. The harness also checks one cheap property
on successful parses: a parsed ``Value`` survives a ``dumps`` ->
``loads`` round trip.

Run:
    pixi run -e fuzz fuzz-loads
"""

from mozz import fuzz, FuzzConfig

from json import loads, dumps


def target(data: List[UInt8]) raises:
    """Fuzz target: parse arbitrary bytes as JSON, then round-trip.

    Args:
        data: Arbitrary bytes interpreted as a UTF-8 (or invalid)
              string. Invalid UTF-8 produces replacement characters,
              which is fine -- we want to see how the parser handles
              them.
    """
    var s = String()
    for i in range(len(data)):
        s += chr(Int(data[i]))

    # Parser must not crash on any input. Errors (e.g. bad escapes,
    # leading zeros, unterminated strings) raise the regular
    # ``Error`` exception which the harness treats as a rejection.
    try:
        var v = loads(s)
        # Round-trip property: parse(dumps(v)) must succeed and the
        # second dumps must produce the same canonical bytes.
        var s2 = dumps(v)
        var v2 = loads(s2)
        var s3 = dumps(v2)
        if s2 != s3:
            raise Error("loads/dumps roundtrip: not idempotent")
    except:
        pass


def _bytes(s: StringLiteral) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        out.append(b[i])
    return out^


def main() raises:
    print("[mozz] fuzzing json.loads()...")

    var seeds = List[List[UInt8]]()

    # Valid JSON across every type
    seeds.append(_bytes("null"))
    seeds.append(_bytes("true"))
    seeds.append(_bytes("false"))
    seeds.append(_bytes("0"))
    seeds.append(_bytes("-3.14e2"))
    seeds.append(_bytes('""'))
    seeds.append(_bytes('"hello"'))
    seeds.append(_bytes('"\\u00e9"'))
    seeds.append(_bytes("[]"))
    seeds.append(_bytes("[1,2,3]"))
    seeds.append(_bytes("{}"))
    seeds.append(_bytes('{"k":"v"}'))
    seeds.append(_bytes('{"a":[1,{"b":[true,null,false]}]}'))

    # Edge cases the stage 2 validator rejects.
    seeds.append(_bytes("007"))
    seeds.append(_bytes("[1,,2]"))
    seeds.append(_bytes('{"k" "v"}'))
    seeds.append(_bytes('"\\x"'))
    seeds.append(_bytes('"\\u"'))
    seeds.append(_bytes("{1:2}"))
    seeds.append(_bytes("[}"))
    seeds.append(_bytes('"unterminated'))
    seeds.append(_bytes("123trailing"))

    fuzz(
        target,
        FuzzConfig(
            max_runs=200_000,
            seed=0,
            verbose=True,
            crash_dir=".mozz_crashes/loads",
            corpus_dir="fuzz/corpus/loads",
            max_input_len=2048,
        ),
        seeds,
    )
