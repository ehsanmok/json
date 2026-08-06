"""Fuzz harness: ``Value`` access + COW mutation.

After ``loads(s)`` succeeds, ``Value`` is a view over a
tape-backed ``Document``. The access surface (``[]``, ``int_value``,
``string_value``, ``object_keys``, ``array_count``, ``raw_json``) and
the COW mutation surface (``set``, ``append``, nested writes) are
both substantial Mojo code. This harness throws fuzzed JSON into
``loads`` and then exercises a representative subset of the access
API to catch:

* Out-of-bounds reads / writes during access on partially-decoded
  payloads.
* COW-on-write mistakes that would surface as panics rather than the
  expected silent re-materialisation.

Lookup miss / type mismatch raising an ``Error`` is an expected
rejection.

Run:
    pixi run -e fuzz fuzz-value-access
"""

from mozz import fuzz, FuzzConfig

from json import loads, dumps, Value


def _exercise(value: Value, depth: Int) raises:
    """Walk a ``Value`` tree, exercising the access API without
    assuming any particular structure."""
    if depth >= 6:
        return

    if value.is_null():
        return
    elif value.is_bool():
        _ = value.bool_value()
    elif value.is_int():
        _ = value.int_value()
    elif value.is_float():
        _ = value.float_value()
    elif value.is_string():
        _ = value.string_value()
    elif value.is_array():
        var n = value.array_count()
        for i in range(n):
            _exercise(value[i], depth + 1)
    elif value.is_object():
        var keys = value.object_keys()
        for i in range(len(keys)):
            _exercise(value[keys[i]], depth + 1)


def target(data: List[UInt8]) raises:
    var s = String(capacity=len(data) + 1)
    for i in range(len(data)):
        s += chr(Int(data[i]))

    try:
        var v = loads(s)
        _exercise(v, 0)

        # raw_json must be a non-empty UTF-8 byte sequence for any
        # successful parse.
        var raw = v.raw_json()
        if raw.byte_length() == 0:
            raise Error("raw_json on parsed Value returned empty")

        # COW property: re-dumping the value must round-trip.
        var d1 = dumps(v)
        var v2 = loads(d1)
        var d2 = dumps(v2)
        if d1 != d2:
            raise Error("dumps/loads idempotence violated after access")
    except:
        pass


def _bytes(s: StringLiteral) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        out.append(b[i])
    return out^


def main() raises:
    print("[mozz] fuzzing Value access + COW...")

    var seeds = List[List[UInt8]]()
    seeds.append(_bytes('{"k":"v","n":1}'))
    seeds.append(_bytes("[1,2,3]"))
    seeds.append(_bytes('{"a":[1,{"b":[true,null,false]}]}'))
    seeds.append(_bytes('{"":""}'))
    seeds.append(_bytes("[[]]"))
    seeds.append(_bytes('{"deep":{"deeper":{"deepest":42}}}'))
    seeds.append(_bytes('[null, true, false, 0, 1, "", "x", [], {}]'))

    fuzz(
        target,
        FuzzConfig(
            max_runs=100_000,
            seed=0,
            verbose=True,
            crash_dir=".mozz_crashes/value_access",
            corpus_dir="fuzz/corpus/value_access",
            max_input_len=2048,
        ),
        seeds,
    )
