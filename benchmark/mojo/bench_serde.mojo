# Typed serde benchmark: `serialize_json` / `deserialize_json` against the
# `Value`-tree round trip, over five record shapes at two batch sizes.
#
# Why this exists: `bench_cpu` measures the parser and `bench_build`
# measures `Value` construction, so the two entry points most consumers
# actually call -- typed struct in, JSON out and back -- had no
# reproducible harness in this repository at all. Published numbers for
# them came from an external suite running a vendored older release,
# which is not something a change here can be checked against.
#
# Shapes. Five records, generated here from a seeded xorshift64* stream
# so a run is reproducible and no fixture files are needed:
#
#   message    flat scalars (bool, int32, int64, float64, two strings)
#   document   nested struct + a list of structs
#   telemetry  two homogeneous lists (strings, floats)
#   strings    one long string list -- escape/copy dominated
#   event      string-heavy record + a list of key/value structs
#
# Each is wrapped in a `{"items": [...]}` batch so one code path serves
# both n=1 and n=100; at n=1 the payload is a few hundred bytes, which
# is where per-call fixed cost shows up.
#
# Lanes, per shape and n:
#
#   serialize    serialize_json(batch)                     -> String
#   deserialize  deserialize_json[Batch](payload)          -> struct
#   loads+walk   loads(payload) then read fields off Value -> struct
#   dumps        dumps(value_tree)  (tree built outside the timer)
#
# The `loads+walk` lane is what a consumer has to write when typed
# deserialization does not cover the shape, so the gap between it and
# the `deserialize` lane is the value of the typed path.
#
# Timing: 5 warmup calls, then 7 runs of a calibrated iteration count
# (~20 ms per run, capped), reporting the median of the per-run means
# and the best run. Calibration keeps a slow lane from taking minutes
# while still giving a fast lane enough iterations to be stable.
#
# Note on comparability: an external suite that times the same shapes
# also copies its record list into the batch wrapper inside the timed
# region. This harness times the library call only, so its numbers are
# the lower, more useful figure for spotting regressions here.
#
# Run: pixi run -e dev bench-serde        (add --json for machine output)

from std.collections import List
from std.sys import argv
from std.time import perf_counter_ns

from json import Value, dumps, loads
from json.reflection import deserialize_json, serialize_json


comptime WARMUP = 5
comptime RUNS = 7
comptime TARGET_NS = 20_000_000
comptime MAX_ITERS = 2000
comptime SEED: UInt64 = 0x5DEECE66D


# ---------------------------------------------------------------------------
# Deterministic generator
# ---------------------------------------------------------------------------

comptime ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-_"


struct Rng(Copyable):
    """A xorshift64* stream -- same sequence on every host and every run."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed if seed != 0 else 0x9E3779B97F4A7C15

    def next_u64(mut self) -> UInt64:
        var x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * 0x2545F4914F6CDD1D

    def below(mut self, n: Int) -> Int:
        return Int(self.next_u64() % UInt64(n))

    def word(mut self, lo: Int, hi: Int) -> String:
        var alphabet = ALPHABET.as_bytes()
        var n = lo + self.below(hi - lo + 1)
        var out = List[UInt8](capacity=n)
        for _ in range(n):
            out.append(alphabet[self.below(len(alphabet))])
        return String(unsafe_from_utf8=out^)

    def unit_float(mut self) -> Float64:
        """A float with a non-trivial decimal expansion."""
        return Float64(self.below(1000000000)) / 1000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


trait Fixture(Copyable, Defaultable, Deinitable):
    @staticmethod
    def label() -> String:
        ...

    @staticmethod
    def build(n: Int) -> Self:
        ...

    def to_value(self) raises -> Value:
        ...

    @staticmethod
    def from_value(v: Value) raises -> Self:
        ...

    def equals(self, other: Self) -> Bool:
        ...

    def checksum(self) -> Int:
        """A cheap value derived from the record, to defeat elision."""
        ...


def _float_eq(a: Float64, b: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    var scale = a if a > 0.0 else -a
    if scale < 1.0:
        scale = 1.0
    return d <= 1e-9 * scale


# --- message ---------------------------------------------------------------


@fieldwise_init
struct Message(Copyable, Defaultable):
    var f_bool: Bool
    var f_int32: Int32
    var f_int64: Int64
    var f_float64: Float64
    var f_string: String
    var f_bool_2: Bool
    var f_int32_2: Int32
    var f_string_2: String

    def __init__(out self):
        self.f_bool = False
        self.f_int32 = 0
        self.f_int64 = 0
        self.f_float64 = 0.0
        self.f_string = ""
        self.f_bool_2 = False
        self.f_int32_2 = 0
        self.f_string_2 = ""

    @staticmethod
    def make(mut rng: Rng) -> Message:
        return Message(
            rng.below(2) == 1,
            Int32(rng.below(2000000) - 1000000),
            Int64(rng.next_u64() % 1000000000000),
            rng.unit_float(),
            rng.word(6, 16),
            rng.below(2) == 1,
            Int32(rng.below(1000)),
            rng.word(3, 12),
        )

    def equals(self, other: Message) -> Bool:
        return (
            self.f_bool == other.f_bool
            and self.f_int32 == other.f_int32
            and self.f_int64 == other.f_int64
            and _float_eq(self.f_float64, other.f_float64)
            and self.f_string == other.f_string
            and self.f_bool_2 == other.f_bool_2
            and self.f_int32_2 == other.f_int32_2
            and self.f_string_2 == other.f_string_2
        )

    def to_value(self) raises -> Value:
        var o = Value.object()
        o.set("f_bool", Value(self.f_bool))
        o.set("f_int32", Value(Int(self.f_int32)))
        o.set("f_int64", Value(self.f_int64))
        o.set("f_float64", Value(self.f_float64))
        o.set("f_string", Value(self.f_string))
        o.set("f_bool_2", Value(self.f_bool_2))
        o.set("f_int32_2", Value(Int(self.f_int32_2)))
        o.set("f_string_2", Value(self.f_string_2))
        return o^

    @staticmethod
    def from_value(v: Value) raises -> Message:
        return Message(
            v["f_bool"].bool_value(),
            Int32(v["f_int32"].int_value()),
            v["f_int64"].int_value(),
            v["f_float64"].float_value(),
            v["f_string"].string_value(),
            v["f_bool_2"].bool_value(),
            Int32(v["f_int32_2"].int_value()),
            v["f_string_2"].string_value(),
        )


@fieldwise_init
struct BatchMessage(Copyable, Defaultable, Deinitable, Fixture):
    var items: List[Message]

    def __init__(out self):
        self.items = List[Message]()

    @staticmethod
    def label() -> String:
        return "message"

    @staticmethod
    def build(n: Int) -> BatchMessage:
        var rng = Rng(SEED)
        var items = List[Message](capacity=n)
        for _ in range(n):
            items.append(Message.make(rng))
        return BatchMessage(items^)

    def to_value(self) raises -> Value:
        var arr = Value.array()
        for i in range(len(self.items)):
            arr.append(self.items[i].to_value())
        var o = Value.object()
        o.set("items", arr^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> BatchMessage:
        var arr = v["items"]
        var n = arr.array_count()
        var items = List[Message](capacity=n)
        for i in range(n):
            items.append(Message.from_value(arr[i]))
        return BatchMessage(items^)

    def equals(self, other: BatchMessage) -> Bool:
        if len(self.items) != len(other.items):
            return False
        for i in range(len(self.items)):
            if not self.items[i].equals(other.items[i]):
                return False
        return True

    def checksum(self) -> Int:
        return len(self.items)


# --- document --------------------------------------------------------------


@fieldwise_init
struct DocumentMeta(Copyable, Defaultable):
    var region: String
    var version: Int32

    def __init__(out self):
        self.region = ""
        self.version = 0


@fieldwise_init
struct DocumentItem(Copyable, Defaultable):
    var sku: String
    var qty: Int32
    var price_minor: Int64

    def __init__(out self):
        self.sku = ""
        self.qty = 0
        self.price_minor = 0


@fieldwise_init
struct Document(Copyable, Defaultable):
    var id: String
    var status: Int32
    var meta: DocumentMeta
    var items: List[DocumentItem]

    def __init__(out self):
        self.id = ""
        self.status = 0
        self.meta = DocumentMeta()
        self.items = List[DocumentItem]()

    @staticmethod
    def make(mut rng: Rng) -> Document:
        var items = List[DocumentItem](capacity=8)
        for _ in range(8):
            items.append(
                DocumentItem(
                    rng.word(6, 14),
                    Int32(rng.below(100) + 1),
                    Int64(rng.below(100000)),
                )
            )
        return Document(
            rng.word(8, 12),
            Int32(rng.below(6)),
            DocumentMeta(rng.word(2, 4), Int32(rng.below(10) + 1)),
            items^,
        )

    def equals(self, other: Document) -> Bool:
        if (
            self.id != other.id
            or self.status != other.status
            or self.meta.region != other.meta.region
            or self.meta.version != other.meta.version
            or len(self.items) != len(other.items)
        ):
            return False
        for i in range(len(self.items)):
            if (
                self.items[i].sku != other.items[i].sku
                or self.items[i].qty != other.items[i].qty
                or self.items[i].price_minor != other.items[i].price_minor
            ):
                return False
        return True

    def to_value(self) raises -> Value:
        var o = Value.object()
        o.set("id", Value(self.id))
        o.set("status", Value(Int(self.status)))
        var meta = Value.object()
        meta.set("region", Value(self.meta.region))
        meta.set("version", Value(Int(self.meta.version)))
        o.set("meta", meta^)
        var items = Value.array()
        for i in range(len(self.items)):
            var it = Value.object()
            it.set("sku", Value(self.items[i].sku))
            it.set("qty", Value(Int(self.items[i].qty)))
            it.set("price_minor", Value(self.items[i].price_minor))
            items.append(it^)
        o.set("items", items^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> Document:
        var arr = v["items"]
        var n = arr.array_count()
        var items = List[DocumentItem](capacity=n)
        for i in range(n):
            var it = arr[i]
            items.append(
                DocumentItem(
                    it["sku"].string_value(),
                    Int32(it["qty"].int_value()),
                    it["price_minor"].int_value(),
                )
            )
        var meta = v["meta"]
        return Document(
            v["id"].string_value(),
            Int32(v["status"].int_value()),
            DocumentMeta(
                meta["region"].string_value(),
                Int32(meta["version"].int_value()),
            ),
            items^,
        )


@fieldwise_init
struct BatchDocument(Copyable, Defaultable, Deinitable, Fixture):
    var items: List[Document]

    def __init__(out self):
        self.items = List[Document]()

    @staticmethod
    def label() -> String:
        return "document"

    @staticmethod
    def build(n: Int) -> BatchDocument:
        var rng = Rng(SEED)
        var items = List[Document](capacity=n)
        for _ in range(n):
            items.append(Document.make(rng))
        return BatchDocument(items^)

    def to_value(self) raises -> Value:
        var arr = Value.array()
        for i in range(len(self.items)):
            arr.append(self.items[i].to_value())
        var o = Value.object()
        o.set("items", arr^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> BatchDocument:
        var arr = v["items"]
        var n = arr.array_count()
        var items = List[Document](capacity=n)
        for i in range(n):
            items.append(Document.from_value(arr[i]))
        return BatchDocument(items^)

    def equals(self, other: BatchDocument) -> Bool:
        if len(self.items) != len(other.items):
            return False
        for i in range(len(self.items)):
            if not self.items[i].equals(other.items[i]):
                return False
        return True

    def checksum(self) -> Int:
        return len(self.items)


# --- telemetry -------------------------------------------------------------


@fieldwise_init
struct Telemetry(Copyable, Defaultable):
    var source: String
    var ts: Int64
    var tags: List[String]
    var values: List[Float64]

    def __init__(out self):
        self.source = ""
        self.ts = 0
        self.tags = List[String]()
        self.values = List[Float64]()

    @staticmethod
    def make(mut rng: Rng) -> Telemetry:
        var tags = List[String](capacity=2)
        for _ in range(2):
            tags.append(rng.word(4, 10))
        var values = List[Float64](capacity=32)
        for _ in range(32):
            values.append(rng.unit_float())
        return Telemetry(
            rng.word(6, 12),
            Int64(1750000000000 + rng.below(86400000)),
            tags^,
            values^,
        )

    def equals(self, other: Telemetry) -> Bool:
        if (
            self.source != other.source
            or self.ts != other.ts
            or len(self.tags) != len(other.tags)
            or len(self.values) != len(other.values)
        ):
            return False
        for i in range(len(self.tags)):
            if self.tags[i] != other.tags[i]:
                return False
        for i in range(len(self.values)):
            if not _float_eq(self.values[i], other.values[i]):
                return False
        return True

    def to_value(self) raises -> Value:
        var o = Value.object()
        o.set("source", Value(self.source))
        o.set("ts", Value(self.ts))
        var tags = Value.array()
        for i in range(len(self.tags)):
            tags.append(Value(self.tags[i]))
        o.set("tags", tags^)
        var values = Value.array()
        for i in range(len(self.values)):
            values.append(Value(self.values[i]))
        o.set("values", values^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> Telemetry:
        var tarr = v["tags"]
        var tags = List[String](capacity=tarr.array_count())
        for i in range(tarr.array_count()):
            tags.append(tarr[i].string_value())
        var varr = v["values"]
        var values = List[Float64](capacity=varr.array_count())
        for i in range(varr.array_count()):
            values.append(varr[i].float_value())
        return Telemetry(
            v["source"].string_value(), v["ts"].int_value(), tags^, values^
        )


@fieldwise_init
struct BatchTelemetry(Copyable, Defaultable, Deinitable, Fixture):
    var items: List[Telemetry]

    def __init__(out self):
        self.items = List[Telemetry]()

    @staticmethod
    def label() -> String:
        return "telemetry"

    @staticmethod
    def build(n: Int) -> BatchTelemetry:
        var rng = Rng(SEED)
        var items = List[Telemetry](capacity=n)
        for _ in range(n):
            items.append(Telemetry.make(rng))
        return BatchTelemetry(items^)

    def to_value(self) raises -> Value:
        var arr = Value.array()
        for i in range(len(self.items)):
            arr.append(self.items[i].to_value())
        var o = Value.object()
        o.set("items", arr^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> BatchTelemetry:
        var arr = v["items"]
        var n = arr.array_count()
        var items = List[Telemetry](capacity=n)
        for i in range(n):
            items.append(Telemetry.from_value(arr[i]))
        return BatchTelemetry(items^)

    def equals(self, other: BatchTelemetry) -> Bool:
        if len(self.items) != len(other.items):
            return False
        for i in range(len(self.items)):
            if not self.items[i].equals(other.items[i]):
                return False
        return True

    def checksum(self) -> Int:
        return len(self.items)


# --- strings ---------------------------------------------------------------


@fieldwise_init
struct Strings(Copyable, Defaultable):
    var items: List[String]

    def __init__(out self):
        self.items = List[String]()

    @staticmethod
    def make(mut rng: Rng) -> Strings:
        var items = List[String](capacity=32)
        for _ in range(32):
            items.append(rng.word(4, 16))
        return Strings(items^)

    def equals(self, other: Strings) -> Bool:
        if len(self.items) != len(other.items):
            return False
        for i in range(len(self.items)):
            if self.items[i] != other.items[i]:
                return False
        return True

    def to_value(self) raises -> Value:
        var arr = Value.array()
        for i in range(len(self.items)):
            arr.append(Value(self.items[i]))
        var o = Value.object()
        o.set("items", arr^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> Strings:
        var arr = v["items"]
        var items = List[String](capacity=arr.array_count())
        for i in range(arr.array_count()):
            items.append(arr[i].string_value())
        return Strings(items^)


@fieldwise_init
struct BatchStrings(Copyable, Defaultable, Deinitable, Fixture):
    var items: List[Strings]

    def __init__(out self):
        self.items = List[Strings]()

    @staticmethod
    def label() -> String:
        return "strings"

    @staticmethod
    def build(n: Int) -> BatchStrings:
        var rng = Rng(SEED)
        var items = List[Strings](capacity=n)
        for _ in range(n):
            items.append(Strings.make(rng))
        return BatchStrings(items^)

    def to_value(self) raises -> Value:
        var arr = Value.array()
        for i in range(len(self.items)):
            arr.append(self.items[i].to_value())
        var o = Value.object()
        o.set("items", arr^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> BatchStrings:
        var arr = v["items"]
        var n = arr.array_count()
        var items = List[Strings](capacity=n)
        for i in range(n):
            items.append(Strings.from_value(arr[i]))
        return BatchStrings(items^)

    def equals(self, other: BatchStrings) -> Bool:
        if len(self.items) != len(other.items):
            return False
        for i in range(len(self.items)):
            if not self.items[i].equals(other.items[i]):
                return False
        return True

    def checksum(self) -> Int:
        return len(self.items)


# --- event -----------------------------------------------------------------


@fieldwise_init
struct EventAttr(Copyable, Defaultable):
    var key: String
    var value: String

    def __init__(out self):
        self.key = ""
        self.value = ""


@fieldwise_init
struct Event(Copyable, Defaultable):
    var event_id: String
    var event_type: String
    var occurred_at: Int64
    var producer: String
    var attrs: List[EventAttr]

    def __init__(out self):
        self.event_id = ""
        self.event_type = ""
        self.occurred_at = 0
        self.producer = ""
        self.attrs = List[EventAttr]()

    @staticmethod
    def make(mut rng: Rng) -> Event:
        var attrs = List[EventAttr](capacity=4)
        for _ in range(4):
            attrs.append(EventAttr(rng.word(3, 10), rng.word(4, 16)))
        return Event(
            rng.word(8, 12),
            rng.word(5, 14),
            Int64(1750000000000 + rng.below(86400000)),
            rng.word(4, 12),
            attrs^,
        )

    def equals(self, other: Event) -> Bool:
        if (
            self.event_id != other.event_id
            or self.event_type != other.event_type
            or self.occurred_at != other.occurred_at
            or self.producer != other.producer
            or len(self.attrs) != len(other.attrs)
        ):
            return False
        for i in range(len(self.attrs)):
            if (
                self.attrs[i].key != other.attrs[i].key
                or self.attrs[i].value != other.attrs[i].value
            ):
                return False
        return True

    def to_value(self) raises -> Value:
        var o = Value.object()
        o.set("event_id", Value(self.event_id))
        o.set("event_type", Value(self.event_type))
        o.set("occurred_at", Value(self.occurred_at))
        o.set("producer", Value(self.producer))
        var attrs = Value.array()
        for i in range(len(self.attrs)):
            var a = Value.object()
            a.set("key", Value(self.attrs[i].key))
            a.set("value", Value(self.attrs[i].value))
            attrs.append(a^)
        o.set("attrs", attrs^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> Event:
        var arr = v["attrs"]
        var n = arr.array_count()
        var attrs = List[EventAttr](capacity=n)
        for i in range(n):
            var a = arr[i]
            attrs.append(
                EventAttr(a["key"].string_value(), a["value"].string_value())
            )
        return Event(
            v["event_id"].string_value(),
            v["event_type"].string_value(),
            v["occurred_at"].int_value(),
            v["producer"].string_value(),
            attrs^,
        )


@fieldwise_init
struct BatchEvent(Copyable, Defaultable, Deinitable, Fixture):
    var items: List[Event]

    def __init__(out self):
        self.items = List[Event]()

    @staticmethod
    def label() -> String:
        return "event"

    @staticmethod
    def build(n: Int) -> BatchEvent:
        var rng = Rng(SEED)
        var items = List[Event](capacity=n)
        for _ in range(n):
            items.append(Event.make(rng))
        return BatchEvent(items^)

    def to_value(self) raises -> Value:
        var arr = Value.array()
        for i in range(len(self.items)):
            arr.append(self.items[i].to_value())
        var o = Value.object()
        o.set("items", arr^)
        return o^

    @staticmethod
    def from_value(v: Value) raises -> BatchEvent:
        var arr = v["items"]
        var n = arr.array_count()
        var items = List[Event](capacity=n)
        for i in range(n):
            items.append(Event.from_value(arr[i]))
        return BatchEvent(items^)

    def equals(self, other: BatchEvent) -> Bool:
        if len(self.items) != len(other.items):
            return False
        for i in range(len(self.items)):
            if not self.items[i].equals(other.items[i]):
                return False
        return True

    def checksum(self) -> Int:
        return len(self.items)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


struct Stat(Copyable):
    """One lane's result: per-call nanoseconds, or `supported=False`."""

    var median_ns: Float64
    var min_ns: Float64
    var bytes: Int
    var ok: Bool
    var supported: Bool
    var note: String

    def __init__(out self):
        self.median_ns = 0.0
        self.min_ns = 0.0
        self.bytes = 0
        self.ok = True
        self.supported = True
        self.note = ""

    @staticmethod
    def unsupported(note: String) -> Stat:
        var s = Stat()
        s.supported = False
        s.note = note
        return s^


def _iters_for(est_ns: Int) -> Int:
    """Iteration count that puts one run near TARGET_NS."""
    if est_ns <= 0:
        return MAX_ITERS
    var it = TARGET_NS // est_ns
    if it < 1:
        return 1
    if it > MAX_ITERS:
        return MAX_ITERS
    return it


def _median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v
    return xs[len(xs) // 2]


def _finish(var samples: List[Float64], bytes: Int, ok: Bool) -> Stat:
    var s = Stat()
    s.min_ns = samples[0]
    for i in range(len(samples)):
        if samples[i] < s.min_ns:
            s.min_ns = samples[i]
    s.median_ns = _median(samples^)
    s.bytes = bytes
    s.ok = ok
    return s^


# --- lanes -----------------------------------------------------------------


def bench_serialize[F: Fixture](fx: F) raises -> Stat:
    for _ in range(WARMUP):
        _ = serialize_json(fx)
    var t0 = Int(perf_counter_ns())
    var payload = serialize_json(fx)
    var t1 = Int(perf_counter_ns())
    var iters = _iters_for(t1 - t0)

    var samples = List[Float64](capacity=RUNS)
    for _ in range(RUNS):
        var a = Int(perf_counter_ns())
        for _ in range(iters):
            var s = serialize_json(fx)
            _ = s.byte_length()
        var b = Int(perf_counter_ns())
        samples.append(Float64(b - a) / Float64(iters))
    return _finish(samples^, payload.byte_length(), True)


def bench_deserialize[F: Fixture](fx: F, payload: String) raises -> Stat:
    try:
        var probe = deserialize_json[F](payload)
        if not fx.equals(probe):
            return Stat.unsupported("fidelity")
    except e:
        var msg = String(e)
        var cut = 42 if msg.byte_length() > 42 else msg.byte_length()
        return Stat.unsupported(String(msg[byte=0:cut]))

    for _ in range(WARMUP):
        _ = deserialize_json[F](payload)
    var t0 = Int(perf_counter_ns())
    _ = deserialize_json[F](payload)
    var t1 = Int(perf_counter_ns())
    var iters = _iters_for(t1 - t0)

    var samples = List[Float64](capacity=RUNS)
    for _ in range(RUNS):
        var a = Int(perf_counter_ns())
        for _ in range(iters):
            var out = deserialize_json[F](payload)
            _ = out.checksum()
        var b = Int(perf_counter_ns())
        samples.append(Float64(b - a) / Float64(iters))
    return _finish(samples^, payload.byte_length(), True)


def bench_loads_walk[F: Fixture](fx: F, payload: String) raises -> Stat:
    var probe = F.from_value(loads(payload))
    var ok = fx.equals(probe)

    for _ in range(WARMUP):
        _ = F.from_value(loads(payload))
    var t0 = Int(perf_counter_ns())
    _ = F.from_value(loads(payload))
    var t1 = Int(perf_counter_ns())
    var iters = _iters_for(t1 - t0)

    var samples = List[Float64](capacity=RUNS)
    for _ in range(RUNS):
        var a = Int(perf_counter_ns())
        for _ in range(iters):
            var out = F.from_value(loads(payload))
            _ = out.checksum()
        var b = Int(perf_counter_ns())
        samples.append(Float64(b - a) / Float64(iters))
    return _finish(samples^, payload.byte_length(), ok)


def bench_dumps[F: Fixture](fx: F) raises -> Stat:
    var tree = fx.to_value()
    for _ in range(WARMUP):
        _ = dumps(tree)
    var t0 = Int(perf_counter_ns())
    var out = dumps(tree)
    var t1 = Int(perf_counter_ns())
    var iters = _iters_for(t1 - t0)

    var samples = List[Float64](capacity=RUNS)
    for _ in range(RUNS):
        var a = Int(perf_counter_ns())
        for _ in range(iters):
            var s = dumps(tree)
            _ = s.byte_length()
        var b = Int(perf_counter_ns())
        samples.append(Float64(b - a) / Float64(iters))
    return _finish(samples^, out.byte_length(), True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pad(s: String, width: Int, left: Bool) -> String:
    var out = s
    while out.byte_length() < width:
        if left:
            out = " " + out
        else:
            out += " "
    return out^


def _fmt_us(ns: Float64) -> String:
    var us = ns / 1000.0
    var scaled = Int64(us * 1000.0 + 0.5)
    var whole = scaled // 1000
    var frac = scaled % 1000
    var frac_s = String(frac)
    while frac_s.byte_length() < 3:
        frac_s = "0" + frac_s
    return String(whole) + "." + frac_s


def _row(fixture: String, n: Int, lane: String, s: Stat):
    if not s.supported:
        print(
            "  ",
            _pad(fixture, 10, False),
            _pad(String(n), 4, True),
            _pad(lane, 12, False),
            _pad("n/a", 12, True),
            _pad("-", 12, True),
            _pad("-", 8, True),
            " ",
            s.note,
        )
        return
    print(
        "  ",
        _pad(fixture, 10, False),
        _pad(String(n), 4, True),
        _pad(lane, 12, False),
        _pad(_fmt_us(s.median_ns), 12, True),
        _pad(_fmt_us(s.min_ns), 12, True),
        _pad(String(s.bytes), 8, True),
        " ",
        "ok" if s.ok else "MISMATCH",
    )


def _json_row(
    mut first: Bool, fixture: String, n: Int, lane: String, s: Stat
) raises:
    if not first:
        print(",")
    first = False
    var o = Value.object()
    o.set("fixture", Value(fixture))
    o.set("n", Value(n))
    o.set("lane", Value(lane))
    o.set("supported", Value(s.supported))
    if s.supported:
        o.set("median_us", Value(s.median_ns / 1000.0))
        o.set("min_us", Value(s.min_ns / 1000.0))
        o.set("bytes", Value(s.bytes))
        o.set("ok", Value(s.ok))
    else:
        o.set("note", Value(s.note))
    print("    " + dumps(o), end="")


def run_fixture[F: Fixture](n: Int, as_json: Bool, mut first: Bool) raises:
    var fx = F.build(n)
    var label = F.label()
    var payload = serialize_json(fx)

    var ser = bench_serialize[F](fx)
    var deser = bench_deserialize[F](fx, payload)
    var walk = bench_loads_walk[F](fx, payload)
    var dmp = bench_dumps[F](fx)

    if as_json:
        _json_row(first, label, n, "serialize", ser)
        _json_row(first, label, n, "deserialize", deser)
        _json_row(first, label, n, "loads+walk", walk)
        _json_row(first, label, n, "dumps", dmp)
    else:
        _row(label, n, "serialize", ser)
        _row(label, n, "deserialize", deser)
        _row(label, n, "loads+walk", walk)
        _row(label, n, "dumps", dmp)


def main() raises:
    var as_json = False
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--json":
            as_json = True

    if as_json:
        print("[")
    else:
        print("Typed serde benchmark -- seeded fixtures, times in microseconds")
        print(
            "median of",
            RUNS,
            "runs (each a calibrated iteration batch) after",
            WARMUP,
            "warmup",
        )
        print()
        print(
            "  ",
            _pad("fixture", 10, False),
            _pad("n", 4, True),
            _pad("lane", 12, False),
            _pad("median", 12, True),
            _pad("best", 12, True),
            _pad("bytes", 8, True),
        )
        print("  " + "-" * 76)

    var first = True
    for n in [1, 100]:
        run_fixture[BatchMessage](n, as_json, first)
        run_fixture[BatchDocument](n, as_json, first)
        run_fixture[BatchTelemetry](n, as_json, first)
        run_fixture[BatchStrings](n, as_json, first)
        run_fixture[BatchEvent](n, as_json, first)
        if not as_json:
            print()

    if as_json:
        print()
        print("]")
