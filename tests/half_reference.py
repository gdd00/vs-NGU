"""Test-only, exact integer binary16 FMA (no float32 arithmetic)."""
from __future__ import annotations

from functools import lru_cache
import random
import struct
import subprocess
import sys


def bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def value(code):
    return struct.unpack("<e", struct.pack("<H", code))[0]


@lru_cache(maxsize=65536)
def units(code):
    """Signed integer coefficient of 2**-24, for a finite binary16."""
    exponent, mantissa = (code >> 10) & 31, code & 1023
    if exponent == 31:
        raise ValueError("finite half operands required by the integer oracle")
    magnitude = (1024 + mantissa) << (exponent - 1) if exponent else mantissa
    return -magnitude if code & 32768 else magnitude


def fma_bits(a, b, c):
    # Exact product/addend in units of 2**-48; Python integers have no overflow.
    n = units(a) * units(b) + (units(c) << 24)
    if not n:
        return 32768 if ((a ^ b) & 32768) and c == 32768 else 0
    sign = 32768 if n < 0 else 0
    n = abs(n)
    exponent = n.bit_length() - 49
    shift = max(24, n.bit_length() - 11)
    q, remainder = divmod(n, 1 << shift)
    midpoint = 1 << (shift - 1)
    q += remainder > midpoint or (remainder == midpoint and (q & 1))
    code = q if exponent < -14 else (exponent + 14) * 1024 + q
    return sign | min(code, 31744)


def fma(a, b, c):
    return value(fma_bits(bits(a), bits(b), bits(c)))


def check_host(helper):
    finite = [n for n in range(65536) if n & 31744 != 31744]
    cases = [(n, 15360, 32768) for n in finite]  # every finite code, signed zeros too
    cases += [(0x3c01, 0x3bff, 0xbc00), (1, 0x3800, 0),
              (1, 0x3800, 1), (0x7bff, 0x4000, 0xfbff),
              (0x8000, 0x3c00, 0x8000), (0x3c00, 0x3c00, 0xbc00)]
    rng = random.Random(0x16f00d)
    cases += [tuple(rng.choice(finite) for _ in range(3)) for _ in range(100000)]
    expected = [fma_bits(*c) for c in cases]
    result = subprocess.run([str(helper), "--half-fma"], input="".join(
        f"{a} {b} {c}\n" for a, b, c in cases), text=True, capture_output=True, check=True)
    actual = [int(s) for s in result.stdout.split()]
    if actual != expected:
        mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
        raise AssertionError((mismatch, cases[mismatch] if mismatch is not None else "wrong count"))
    # This oracle must distinguish FMA from a separately rounded product/add.
    assert fma(1.0009765625, 0.99951171875, -1.0) != value(bits(
        value(bits(1.0009765625 * 0.99951171875)) - 1.0))
    print(f"Exact integer oracle: {len(cases)} half FMA triples matched the host reference")


if __name__ == "__main__":
    check_host(sys.argv[1])
