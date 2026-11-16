#pragma once

#include <bit>
#include <cstdint>

namespace ngu {

// IEEE binary16 storage conversions, round-to-nearest/ties-to-even, with
// gradual underflow. Integer bit operations make this independent of the host
// floating-point rounding mode and avoid requiring native FP16 instructions.
inline std::uint16_t float_to_half(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = (bits >> 16) & 0x8000u;
    const auto exponent = (bits >> 23) & 0xffu;
    const auto mantissa = bits & 0x7fffffu;
    if (exponent == 255)
        return static_cast<std::uint16_t>(sign | 0x7c00u |
            (mantissa ? (0x200u | (mantissa >> 13)) : 0u));
    if (exponent >= 143)
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent < 102)
        return static_cast<std::uint16_t>(sign);
    if (exponent < 113) {
        const auto shift = 126u - exponent;
        const auto significand = mantissa | 0x800000u;
        auto result = significand >> shift;
        const auto remainder = significand & ((1u << shift) - 1u);
        const auto halfway = 1u << (shift - 1u);
        result += remainder > halfway || (remainder == halfway && (result & 1u));
        return static_cast<std::uint16_t>(sign | result);
    }
    auto result = ((exponent - 112u) << 10) | (mantissa >> 13);
    const auto remainder = mantissa & 0x1fffu;
    result += remainder > 0x1000u || (remainder == 0x1000u && (result & 1u));
    return static_cast<std::uint16_t>(sign | result);
}

inline float half_to_float(std::uint16_t value) noexcept {
    const auto sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    const auto exponent = (value >> 10) & 0x1fu;
    auto mantissa = static_cast<std::uint32_t>(value & 0x3ffu);
    if (exponent == 0) {
        if (mantissa == 0)
            return std::bit_cast<float>(sign);
        int e = -14;
        while (!(mantissa & 0x400u)) {
            mantissa <<= 1;
            --e;
        }
        return std::bit_cast<float>(sign | (static_cast<std::uint32_t>(e + 127) << 23) |
            ((mantissa & 0x3ffu) << 13));
    }
    return std::bit_cast<float>(sign | ((exponent == 31 ? 255u : exponent + 112u) << 23) |
        (mantissa << 13));
}

inline float round_half(float value) noexcept {
    return half_to_float(float_to_half(value));
}

} // namespace ngu
