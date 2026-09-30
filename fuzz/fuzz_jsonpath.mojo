"""Fuzz harness: ``json.jsonpath_query`` (RFC 9535).

Tests ``jsonpath_query()`` for crashes on arbitrary path strings
against a fixed document. Malformed paths raise a regular ``Error``
(an "expected rejection"); only panic-like errors and aborts are bugs.

Why a fixed document? The interesting failure mode for a JSONPath
engine is the path expression -- recursive descent, slices, filters,
union expressions. Fuzzing the path against a stable, structurally
varied document gives the engine a wide search space without
conflating parser bugs with engine bugs.

Run:
    pixi run -e fuzz fuzz-jsonpath
"""

from mozz import fuzz, FuzzConfig

from json import loads
from json.jsonpath import jsonpath_query, jsonpath_one


comptime DOC: StaticString = """{
    "store": {
        "book": [
            {"category": "ref",     "author": "Nigel", "price": 8.95},
            {"category": "fiction", "author": "Evelyn","price": 12.99},
            {"category": "fiction", "author": "Tolkien","price": 22.99}
        ],
        "bicycle": {"color": "red", "price": 19.95}
    },
    "expensive": 10
}"""


def target(data: List[UInt8]) raises:
    """Fuzz target: run an arbitrary path expression against ``DOC``.
    """
    var path = String()
    for i in range(len(data)):
        path += chr(Int(data[i]))

    var v = loads(DOC)

    try:
        _ = jsonpath_query(v, path)
    except:
        pass

    try:
        _ = jsonpath_one(v, path)
    except:
        pass


def _bytes(s: StringLiteral) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8](capacity=len(b))
    for i in range(len(b)):
        out.append(b[i])
    return out^


def main() raises:
    print("[mozz] fuzzing jsonpath_query() / jsonpath_one()...")

    var seeds = List[List[UInt8]]()

    # Spec-correct paths
    seeds.append(_bytes("$"))
    seeds.append(_bytes("$.store"))
    seeds.append(_bytes("$.store.book[0]"))
    seeds.append(_bytes("$.store.book[*]"))
    seeds.append(_bytes("$.store.book[*].author"))
    seeds.append(_bytes("$..book"))
    seeds.append(_bytes("$..*"))
    seeds.append(_bytes("$..book[?@.price<10]"))
    seeds.append(_bytes("$..book[?@.price>=$.expensive]"))
    seeds.append(_bytes("$.store.book[0:2]"))
    seeds.append(_bytes("$.store.book[-1]"))

    # Malformed paths the engine should reject without crashing
    seeds.append(_bytes("$["))
    seeds.append(_bytes("$.."))
    seeds.append(_bytes("$.[*]"))
    seeds.append(_bytes("$..book[?]"))
    seeds.append(_bytes("$.store..book[*]..author"))
    seeds.append(_bytes("$['store']['book'][?(@.price<10)]"))
    seeds.append(_bytes(""))
    seeds.append(_bytes(".store"))
    seeds.append(_bytes("foo"))

    fuzz(
        target,
        FuzzConfig(
            max_runs=200_000,
            seed=0,
            verbose=True,
            crash_dir=".mozz_crashes/jsonpath",
            corpus_dir="fuzz/corpus/jsonpath",
            max_input_len=512,
        ),
        seeds,
    )
