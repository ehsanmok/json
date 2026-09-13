# Shortest round-trip decimal conversion for `Float64` (Grisu2).
#
# Why not the stdlib. `String(Float64)` does not always produce a
# representation that reads back as the value it was given: the double
# `0x43903c0e61516cab` formats to `2.924564535875448e+17`, which
# denotes its neighbour. The Dragonbox core underneath it,
# `_to_decimal`, returns the same digits for both of those doubles, so
# the shortfall is in the digit generation rather than in the
# formatting around it. A JSON writer built on that can emit a number
# that nothing, including this library, reads back as what was
# written, which is the one property a serializer may not lose.
#
# Grisu2 (Loitsch, "Printing Floating-Point Numbers Quickly and
# Accurately with Integers", PLDI 2010) generates digits from a
# 64-bit fixed-point approximation with an explicit uncertainty
# interval, and only emits a digit string it has proved lies inside
# the rounding interval of the original double. So its output always
# reads back as the value it came from. It is not always the shortest
# such string -- in roughly one case in a thousand it emits one digit
# more than necessary -- but it is never wrong, and one extra digit is
# a far better failure than a different number.
#
# The cached powers of ten are the standard table: 87 entries at
# every eighth decimal exponent from 1e-348 to 1e340, each a 64-bit
# normalized significand with its binary exponent. Between them, one
# 64x64 multiply reaches any decimal exponent in range.

from std.bit import count_leading_zeros
from std.memory import bitcast

comptime _POW10_COUNT = 87
comptime _POW10_MIN_K = -348
comptime _POW10_STEP = 8

comptime _POW10_SIGNIFICAND = _make_pow10_significand()
comptime _POW10_EXPONENT = _make_pow10_exponent()


def _make_pow10_significand(out table: InlineArray[UInt64, _POW10_COUNT]):
    """Cached powers of ten as normalized 64-bit significands."""
    table = InlineArray[UInt64, _POW10_COUNT](uninitialized=True)
    table[0] = 0xFA8FD5A0081C0288
    table[1] = 0xBAAEE17FA23EBF76
    table[2] = 0x8B16FB203055AC76
    table[3] = 0xCF42894A5DCE35EA
    table[4] = 0x9A6BB0AA55653B2D
    table[5] = 0xE61ACF033D1A45DF
    table[6] = 0xAB70FE17C79AC6CA
    table[7] = 0xFF77B1FCBEBCDC4F
    table[8] = 0xBE5691EF416BD60C
    table[9] = 0x8DD01FAD907FFC3C
    table[10] = 0xD3515C2831559A83
    table[11] = 0x9D71AC8FADA6C9B5
    table[12] = 0xEA9C227723EE8BCB
    table[13] = 0xAECC49914078536D
    table[14] = 0x823C12795DB6CE57
    table[15] = 0xC21094364DFB5637
    table[16] = 0x9096EA6F3848984F
    table[17] = 0xD77485CB25823AC7
    table[18] = 0xA086CFCD97BF97F4
    table[19] = 0xEF340A98172AACE5
    table[20] = 0xB23867FB2A35B28E
    table[21] = 0x84C8D4DFD2C63F3B
    table[22] = 0xC5DD44271AD3CDBA
    table[23] = 0x936B9FCEBB25C996
    table[24] = 0xDBAC6C247D62A584
    table[25] = 0xA3AB66580D5FDAF6
    table[26] = 0xF3E2F893DEC3F126
    table[27] = 0xB5B5ADA8AAFF80B8
    table[28] = 0x87625F056C7C4A8B
    table[29] = 0xC9BCFF6034C13053
    table[30] = 0x964E858C91BA2655
    table[31] = 0xDFF9772470297EBD
    table[32] = 0xA6DFBD9FB8E5B88F
    table[33] = 0xF8A95FCF88747D94
    table[34] = 0xB94470938FA89BCF
    table[35] = 0x8A08F0F8BF0F156B
    table[36] = 0xCDB02555653131B6
    table[37] = 0x993FE2C6D07B7FAC
    table[38] = 0xE45C10C42A2B3B06
    table[39] = 0xAA242499697392D3
    table[40] = 0xFD87B5F28300CA0E
    table[41] = 0xBCE5086492111AEB
    table[42] = 0x8CBCCC096F5088CC
    table[43] = 0xD1B71758E219652C
    table[44] = 0x9C40000000000000
    table[45] = 0xE8D4A51000000000
    table[46] = 0xAD78EBC5AC620000
    table[47] = 0x813F3978F8940984
    table[48] = 0xC097CE7BC90715B3
    table[49] = 0x8F7E32CE7BEA5C70
    table[50] = 0xD5D238A4ABE98068
    table[51] = 0x9F4F2726179A2245
    table[52] = 0xED63A231D4C4FB27
    table[53] = 0xB0DE65388CC8ADA8
    table[54] = 0x83C7088E1AAB65DB
    table[55] = 0xC45D1DF942711D9A
    table[56] = 0x924D692CA61BE758
    table[57] = 0xDA01EE641A708DEA
    table[58] = 0xA26DA3999AEF774A
    table[59] = 0xF209787BB47D6B85
    table[60] = 0xB454E4A179DD1877
    table[61] = 0x865B86925B9BC5C2
    table[62] = 0xC83553C5C8965D3D
    table[63] = 0x952AB45CFA97A0B3
    table[64] = 0xDE469FBD99A05FE3
    table[65] = 0xA59BC234DB398C25
    table[66] = 0xF6C69A72A3989F5C
    table[67] = 0xB7DCBF5354E9BECE
    table[68] = 0x88FCF317F22241E2
    table[69] = 0xCC20CE9BD35C78A5
    table[70] = 0x98165AF37B2153DF
    table[71] = 0xE2A0B5DC971F303A
    table[72] = 0xA8D9D1535CE3B396
    table[73] = 0xFB9B7CD9A4A7443C
    table[74] = 0xBB764C4CA7A44410
    table[75] = 0x8BAB8EEFB6409C1A
    table[76] = 0xD01FEF10A657842C
    table[77] = 0x9B10A4E5E9913129
    table[78] = 0xE7109BFBA19C0C9D
    table[79] = 0xAC2820D9623BF429
    table[80] = 0x80444B5E7AA7CF85
    table[81] = 0xBF21E44003ACDD2D
    table[82] = 0x8E679C2F5E44FF8F
    table[83] = 0xD433179D9C8CB841
    table[84] = 0x9E19DB92B4E31BA9
    table[85] = 0xEB96BF6EBADF77D9
    table[86] = 0xAF87023B9BF0EE6B


def _make_pow10_exponent(out table: InlineArray[Int32, _POW10_COUNT]):
    """Binary exponents matching `_POW10_SIGNIFICAND`."""
    table = InlineArray[Int32, _POW10_COUNT](uninitialized=True)
    table[0] = -1220
    table[1] = -1193
    table[2] = -1166
    table[3] = -1140
    table[4] = -1113
    table[5] = -1087
    table[6] = -1060
    table[7] = -1034
    table[8] = -1007
    table[9] = -980
    table[10] = -954
    table[11] = -927
    table[12] = -901
    table[13] = -874
    table[14] = -847
    table[15] = -821
    table[16] = -794
    table[17] = -768
    table[18] = -741
    table[19] = -715
    table[20] = -688
    table[21] = -661
    table[22] = -635
    table[23] = -608
    table[24] = -582
    table[25] = -555
    table[26] = -529
    table[27] = -502
    table[28] = -475
    table[29] = -449
    table[30] = -422
    table[31] = -396
    table[32] = -369
    table[33] = -343
    table[34] = -316
    table[35] = -289
    table[36] = -263
    table[37] = -236
    table[38] = -210
    table[39] = -183
    table[40] = -157
    table[41] = -130
    table[42] = -103
    table[43] = -77
    table[44] = -50
    table[45] = -24
    table[46] = 3
    table[47] = 30
    table[48] = 56
    table[49] = 83
    table[50] = 109
    table[51] = 136
    table[52] = 162
    table[53] = 189
    table[54] = 216
    table[55] = 242
    table[56] = 269
    table[57] = 295
    table[58] = 322
    table[59] = 348
    table[60] = 375
    table[61] = 402
    table[62] = 428
    table[63] = 455
    table[64] = 481
    table[65] = 508
    table[66] = 534
    table[67] = 561
    table[68] = 588
    table[69] = 614
    table[70] = 641
    table[71] = 667
    table[72] = 694
    table[73] = 720
    table[74] = 747
    table[75] = 774
    table[76] = 800
    table[77] = 827
    table[78] = 853
    table[79] = 880
    table[80] = 907
    table[81] = 933
    table[82] = 960
    table[83] = 986
    table[84] = 1013
    table[85] = 1039
    table[86] = 1066


# ---------------------------------------------------------------------------
# Fixed-point arithmetic
# ---------------------------------------------------------------------------


@fieldwise_init
struct _DiyFp(Copyable, ImplicitlyCopyable, Movable):
    """A significand and a binary exponent: `f * 2**e`.

    Deliberately not a `Float64`: the whole point is to do the digit
    generation in integer arithmetic, where the error is exactly
    trackable rather than accumulated silently.
    """

    var f: UInt64
    var e: Int


comptime _SIGNIFICAND_BITS = 52
comptime _EXPONENT_BIAS = 0x3FF + _SIGNIFICAND_BITS
comptime _HIDDEN_BIT: UInt64 = UInt64(1) << UInt64(52)
comptime _SIGNIFICAND_MASK: UInt64 = _HIDDEN_BIT - 1
comptime _EXPONENT_MASK: UInt64 = UInt64(0x7FF) << 52


@always_inline
def _multiply(a: _DiyFp, b: _DiyFp) -> _DiyFp:
    """`a * b`, keeping the top 64 bits and rounding to nearest."""
    var product = UInt128(a.f) * UInt128(b.f)
    var high = UInt64(product >> 64)
    var low = UInt64(product & UInt128(UInt64.MAX))
    # Round half up on the discarded half.
    high += low >> 63
    return _DiyFp(high, a.e + b.e + 64)


@always_inline
def _normalize(value: _DiyFp) -> _DiyFp:
    """Shift the significand until its top bit is set."""
    var shift = Int(count_leading_zeros(value.f))
    return _DiyFp(value.f << UInt64(shift), value.e - shift)


@always_inline
def _boundaries(value: Float64) -> Tuple[_DiyFp, _DiyFp, _DiyFp]:
    """The value and the midpoints to its two neighbours.

    Any decimal strictly inside (`minus`, `plus`) reads back as
    `value`, which is the interval the digit generator has to land in.
    """
    var bits = bitcast[DType.uint64](value)
    var biased_exponent = Int((bits & _EXPONENT_MASK) >> _SIGNIFICAND_BITS)
    var significand = bits & _SIGNIFICAND_MASK

    var w: _DiyFp
    if biased_exponent == 0:
        # Subnormal: no hidden bit, and the exponent is fixed.
        w = _DiyFp(significand, 1 - _EXPONENT_BIAS)
    else:
        w = _DiyFp(significand + _HIDDEN_BIT, biased_exponent - _EXPONENT_BIAS)

    var plus = _normalize(_DiyFp((w.f << 1) + 1, w.e - 1))

    var minus: _DiyFp
    if w.f == _HIDDEN_BIT and biased_exponent != 1:
        # At a power of two the gap below is half the gap above.
        minus = _DiyFp((w.f << 2) - 1, w.e - 2)
    else:
        minus = _DiyFp((w.f << 1) - 1, w.e - 1)
    minus = _DiyFp(minus.f << UInt64(minus.e - plus.e), plus.e)

    return (_normalize(w), minus, plus)


def _cached_power_for(exponent: Int) -> Tuple[_DiyFp, Int]:
    """A cached power of ten that brings `exponent` into range.

    Returns the power and its decimal exponent. The index arithmetic
    is the standard one: estimate the decimal exponent from the binary
    one by multiplying by log10(2), then round up to the next multiple
    of eight, which is the table's step.
    """
    # Digit generation needs the scaled exponent to land in roughly
    # [-59, -32]: high enough that the integer part fits 32 bits, low
    # enough that `1 << -e` still fits 64. Solving that for the
    # decimal exponent and rounding up to the table's step of eight is
    # this arithmetic. The offset keeps the estimate positive across
    # the whole range a `Float64` exponent can take, so the ceiling is
    # a comparison against the truncation rather than a call.
    var estimate = (
        Float64(-60 - exponent) * 0.30102999566398114 - Float64(_POW10_MIN_K)
    ) / Float64(_POW10_STEP)
    var index = Int(estimate)
    if estimate - Float64(index) > 0.0:
        index += 1
    var table_f = materialize[_POW10_SIGNIFICAND]()
    var table_e = materialize[_POW10_EXPONENT]()
    var decimal_exponent = _POW10_MIN_K + index * _POW10_STEP
    return (_DiyFp(table_f[index], Int(table_e[index])), decimal_exponent)


# ---------------------------------------------------------------------------
# Digit generation
# ---------------------------------------------------------------------------


@always_inline
def _decimal_length(value: UInt32) -> Int:
    """How many decimal digits `value` needs.

    Halving the range each time rather than walking down from ten, so
    the answer costs four comparisons instead of up to ten.
    """
    if value >= 100000:
        if value >= 10000000:
            if value >= 1000000000:
                return 10
            if value >= 100000000:
                return 9
            return 8
        if value >= 1000000:
            return 7
        return 6
    if value >= 1000:
        if value >= 10000:
            return 5
        return 4
    if value >= 100:
        return 3
    if value >= 10:
        return 2
    return 1


@always_inline
def _round_weed(
    mut digits: InlineArray[UInt8, 24],
    length: Int,
    delta: UInt64,
    var rest: UInt64,
    ten_kappa: UInt64,
    weight: UInt64,
):
    """Nudge the last digit toward the value when that stays in range.

    Grisu2 stops as soon as the remaining uncertainty is smaller than
    the interval, which can leave the last digit one step further from
    the true value than it needs to be. Walking it back while the
    result provably stays inside the interval is what makes the output
    shortest more often.
    """
    while (
        rest < weight
        and delta - rest >= ten_kappa
        and (
            rest + ten_kappa < weight
            or weight - rest > rest + ten_kappa - weight
        )
    ):
        digits[length - 1] -= 1
        rest += ten_kappa


def _generate_digits(
    w: _DiyFp,
    minus: _DiyFp,
    plus: _DiyFp,
    mut digits: InlineArray[UInt8, 24],
) -> Tuple[Int, Int]:
    """Emit digits of a value in (`minus`, `plus`). Returns (count, K).

    The value is split at the binary point of the scaled
    representation: the part above it is an integer that yields digits
    by division, and the part below yields digits by repeated
    multiplication. Generation stops the moment the accumulated
    uncertainty is smaller than the interval, which is what guarantees
    the digits read back as the original double.
    """
    var one = _DiyFp(UInt64(1) << UInt64(-plus.e), plus.e)
    var integral = UInt32(plus.f >> UInt64(-one.e))
    var fractional = plus.f & (one.f - 1)
    var delta = plus.f - minus.f

    var length = 0
    var width = _decimal_length(integral)

    # The divisor is a power of ten chosen by the digit count, so the
    # loop is unrolled over the ten possible widths and every division
    # is by a constant. Left as `pow10[kappa - 1]` the compiler has to
    # emit a real 32-bit division per digit, which is an order of
    # magnitude slower than the multiply-and-shift it generates for a
    # literal. `width` cannot exceed ten, so at most ten of these arms
    # are live and the first digit is never zero.
    comptime for step in range(10, 0, -1):
        if width >= step:
            comptime divisor = UInt32(10) ** UInt32(step - 1)
            var digit = integral // divisor
            integral %= divisor
            if digit != 0 or length != 0:
                digits[length] = UInt8(0x30 + Int(digit))
                length += 1
            var remainder = (UInt64(integral) << UInt64(-one.e)) + fractional
            if remainder <= delta:
                _round_weed(
                    digits,
                    length,
                    delta,
                    remainder,
                    UInt64(divisor) << UInt64(-one.e),
                    plus.f - w.f,
                )
                return (length, step - 1)

    var kappa = 0
    # Ten to the number of steps taken, carried along rather than
    # looked up on the way out: the exit needs it once, and reading it
    # from a `comptime` array costs a copy of the whole array.
    var power: UInt64 = 1

    while True:
        fractional *= 10
        delta *= 10
        power *= 10
        var digit = fractional >> UInt64(-one.e)
        if digit != 0 or length != 0:
            digits[length] = UInt8(0x30 + Int(digit))
            length += 1
        fractional &= one.f - 1
        kappa -= 1
        if fractional < delta:
            # Past nine steps the scale no longer fits sixty-four bits,
            # and the weight it would carry is smaller than the
            # remaining uncertainty anyway, so the nudge is skipped
            # rather than guessed.
            var weight: UInt64 = 0
            if -kappa < 9:
                weight = (plus.f - w.f) * power
            _round_weed(digits, length, delta, fractional, one.f, weight)
            return (length, kappa)


def shortest_digits(
    value: Float64, mut digits: InlineArray[UInt8, 24]
) -> Tuple[Int, Int]:
    """Decimal digits of a finite non-zero `value`, and its exponent.

    The value equals `0.d1 d2 ... dn * 10**(K + n)` where `n` is the
    returned count and `K` the returned exponent; equivalently, the
    digit string read as an integer, times ten to the `K`.

    Args:
        value: A finite, non-zero, positive `Float64`.
        digits: Filled with ASCII digits.

    Returns:
        The digit count and the decimal exponent `K`.
    """
    var bounds = _boundaries(value)
    var w = bounds[0]
    var minus = bounds[1]
    var plus = bounds[2]

    var cached = _cached_power_for(plus.e)
    var scale = cached[0]
    var scale_k = cached[1]

    var scaled_w = _multiply(w, scale)
    var scaled_minus = _multiply(minus, scale)
    var scaled_plus = _multiply(plus, scale)

    # Pull the interval in by one unit on each side so that a decimal
    # landing exactly on a boundary is not accepted: the boundary
    # belongs to the neighbouring double as often as to this one.
    scaled_minus = _DiyFp(scaled_minus.f + 1, scaled_minus.e)
    scaled_plus = _DiyFp(scaled_plus.f - 1, scaled_plus.e)

    var generated = _generate_digits(
        scaled_w, scaled_minus, scaled_plus, digits
    )
    return (generated[0], generated[1] - scale_k)
