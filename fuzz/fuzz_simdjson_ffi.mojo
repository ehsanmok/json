"""Fuzz harness: simdjson FFI boundary.

Drives the ``loads[target='cpu-simdjson']`` path, which crosses an
``OwnedDLHandle`` + ``external_call`` boundary into the simdjson C++
wrapper for every byte of input. Catches:

* Pointer / lifetime mistakes in the FFI shim.
* Inputs simdjson rejects but our shim mishandles (use-after-free,
  double-free, leaks under sanitizer).
* Differential drift between the simdjson FFI backend and the default
  Mojo two-pass parser: when both succeed, their canonical
  ``dumps`` output must match.

Malformed JSON raising a regular ``Error`` is an expected rejection.
Only panic-like errors and aborts are bugs.

Run:
    pixi run -e fuzz fuzz-simdjson
"""

from mozz import fuzz, FuzzConfig

from json import loads, dumps


def target(data: List[UInt8]) raises:
    """Fuzz target: drive both backends with the same bytes.

    Args:
        data: Arbitrary bytes interpreted as a UTF-8 (or invalid)
              string. The simdjson backend rejects non-UTF-8 with
              ``SIMDJSON_ERROR_UTF8``; our default backend tolerates
              replacement characters.
    """
    var s = String()
    for i in range(len(data)):
        s += chr(Int(data[i]))

    var simd_ok = False
    var simd_dump = String("")
    try:
        var v = loads[target="cpu-simdjson"](s)
        simd_dump = dumps(v)
        simd_ok = True
    except:
        pass

    var mojo_ok = False
    var mojo_dump = String("")
    try:
        var v2 = loads(s)
        mojo_dump = dumps(v2)
        mojo_ok = True
    except:
        pass

    # Differential property: when both succeed, their canonical dumps
    # must match. Disagreement is a real bug -- one of the parsers is
    # accepting a value the other doesn't.
    if simd_ok and mojo_ok:
        if simd_dump != mojo_dump:
            raise Error(
                "simdjson FFI / Mojo native dumps disagree: simd="
                + simd_dump
                + " mojo="
                + mojo_dump
            )


def _bytes(s: StringLiteral) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        out.append(b[i])
    return out^


def main() raises:
    print("[mozz] fuzzing simdjson FFI boundary...")

    var seeds = List[List[UInt8]]()

    seeds.append(_bytes("null"))
    seeds.append(_bytes("true"))
    seeds.append(_bytes("0"))
    seeds.append(_bytes('"hello"'))
    seeds.append(_bytes('"\\u00e9"'))
    seeds.append(_bytes("[1,2,3]"))
    seeds.append(_bytes('{"k":"v","n":42,"a":[true,null,false]}'))
    seeds.append(_bytes('{"big":1234567890123456789}'))
    seeds.append(_bytes('{"f":1e308}'))

    # Inputs the simdjson capacity / UTF-8 paths might react to
    seeds.append(_bytes(""))
    seeds.append(_bytes(" "))
    seeds.append(_bytes("[\xff]"))
    seeds.append(_bytes('"\xc3\x28"'))
    seeds.append(_bytes("{"))
    seeds.append(_bytes("[1, 2, 3"))
    seeds.append(_bytes('{"k":"v"} extra'))

    fuzz(
        target,
        FuzzConfig(
            max_runs=200_000,
            seed=0,
            verbose=True,
            crash_dir=".mozz_crashes/simdjson",
            corpus_dir="fuzz/corpus/simdjson",
            max_input_len=4096,
        ),
        seeds,
    )
