#pragma once

#include <cstdint>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>

namespace ngu {

struct HalfCapabilities {
    bool arithmetic, storage, signed_zero_inf_nan, rte, denorm_preserve;
};

inline void require_half_capabilities(const HalfCapabilities &c) {
    const char *missing = !c.arithmetic ? "shaderFloat16" :
        !c.storage ? "storageBuffer16BitAccess" :
        !c.signed_zero_inf_nan ? "shaderSignedZeroInfNanPreserveFloat16" :
        !c.rte ? "shaderRoundingModeRTEFloat16" :
        !c.denorm_preserve ? "shaderDenormPreserveFloat16" : nullptr;
    if (missing)
        throw std::runtime_error(std::string("Vulkan GRAYH requires ") + missing +
            "; no FP32 fallback is allowed. Use GRAYS explicitly for FP32 inference.");
}

inline void replace_shader_text(std::string &s, const std::string &from, const std::string &to) {
    for (std::size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size())
        s.replace(pos, from.size(), to);
}

// Specialize the fusion and fallback generators into pure native binary16.
// All real types, literals, parameters, shared tiles and SSBOs become binary16;
// integer indices remain 32-bit. Fused half parameters pack eight binary16
// coefficients densely into each 16-byte uvec4 (two f16vec4s in .xy and .zw,
// zero wasted constant-cache bytes) and decode via OpBitcast without requiring
// uniformAndStorageBuffer16BitAccess. Convolution MACs emit inline GLSL.std.450
// binary16 Fma directly without function-call overhead.
inline std::string half_shader(std::string s) {
    const bool has_dense_ubo = s.find("NGU_UBO_DECL") != std::string::npos;
    if (has_dense_ubo) {
        replace_shader_text(s,
            "#define NGU_UBO_DECL(n4, n8) layout(set=0, binding=2, std140) uniform Params { vec4 ubo[n4]; };",
            "__NGU_HALF_UBO_DECL__");
        replace_shader_text(s, "#define NGU_P0(i2) ubo[(i2) * 2u]", "__NGU_HALF_P0__");
        replace_shader_text(s, "#define NGU_P1(i2) ubo[(i2) * 2u + 1u]", "__NGU_HALF_P1__");
        replace_shader_text(s,
            "#define NGU_LOAD_PAIR(i2, uvar, wa, wb) vec4 wa = ubo[(i2) * 2u], wb = ubo[(i2) * 2u + 1u]",
            "__NGU_HALF_LOAD_PAIR__");
        replace_shader_text(s, "#define NGU_MAC(acc, sv, wv) acc += (sv) * (wv)", "__NGU_HALF_MAC__");
    }

    replace_shader_text(s, "#version 460", R"glsl(#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require)glsl");
    replace_shader_text(s, "4461, 32", "4461, 16");
    replace_shader_text(s, "4462, 32", "4462, 16");
    s.insert(s.find("layout("), "spirv_execution_mode(capabilities=[4464], 4459, 16);\n");
    replace_shader_text(s, "#define NGU_RAW uint", "#define NGU_RAW float16_t");
    replace_shader_text(s, "#define NGU_FROM_RAW uintBitsToFloat", "#define NGU_FROM_RAW(x) (x)");
    replace_shader_text(s, "#define NGU_TO_RAW floatBitsToUint", "#define NGU_TO_RAW(x) (x)");

    // Exact binary16 roundings of the FP32 baseline constants.
    replace_shader_text(s, "uintBitsToFloat(0x3f5f3333u)", "0.8720703125");
    replace_shader_text(s, "uintBitsToFloat(0x3e766666u)", "0.2406005859375");
    replace_shader_text(s, "uintBitsToFloat(0xbdaccccdu)", "(-0.0843505859375)");
    replace_shader_text(s, "uintBitsToFloat(0xbce66666u)", "(-0.0281219482421875)");
    s = std::regex_replace(s, std::regex(R"(\bfloat\b)"), "float16_t");
    s = std::regex_replace(s, std::regex(R"(\bvec4\b)"), "f16vec4");
    s = std::regex_replace(s, std::regex(R"(\b([0-9]+\.[0-9]+)\b)"), "$1hf");

    // Legacy uniform Params block support if present without NGU_UBO_DECL.
    const auto params = s.find("layout(set=0, binding=2, std140) uniform Params");
    if (params != std::string::npos) {
        const auto end = s.find("};", params) + 2;
        std::string block = s.substr(params, end - params);
        replace_shader_text(block, "f16vec4", "uvec4");
        s.erase(params, end - params);
        s = std::regex_replace(s, std::regex(R"(\b([wb](?:[0-9]+|h))\[([^\]]+)\])"),
                               "ngu_half_bits($1[$2].xy)");
        s.insert(params, block);
        s.insert(s.find("layout("),
                 "spirv_instruction(id=124) f16vec4 ngu_half_bits(uvec2 bits);\n");
    }

    // Fallback shader scalar*vector MAC contraction to inline native binary16 FMA.
    s = std::regex_replace(s,
        std::regex(R"(\b(o[0-9]+|q[0-9]+_[0-9]+|p[0-9]+_[0-9]+|h[0-9]+|hsum|sum)\s*\+=\s*(\w+(?:\.[xyzw])?)\s*\*\s*([^;]+);)"),
        "$1 = fma(f16vec4($2), $3, $1);");

    // No implicit contraction/reassociation outside the explicit half FMA.
    replace_shader_text(s, "precise ", "");
    s = std::regex_replace(s,
        std::regex(R"(\b(float16_t|f16vec4)(?=\s+\w+\s*(?:\[|=|;|,)))"), "precise $1");

    if (has_dense_ubo) {
        s.insert(s.find("layout("),
                 "spirv_instruction(id=124) f16vec4 ngu_half_bits(uvec2 bits);\n");
        replace_shader_text(s, "__NGU_HALF_UBO_DECL__",
            "#define NGU_UBO_DECL(n4, n8) layout(set=0, binding=2, std140) uniform Params { uvec4 ubo[n8]; };");
        replace_shader_text(s, "__NGU_HALF_P0__",
            "#define NGU_P0(i2) ngu_half_bits(ubo[i2].xy)");
        replace_shader_text(s, "__NGU_HALF_P1__",
            "#define NGU_P1(i2) ngu_half_bits(ubo[i2].zw)");
        replace_shader_text(s, "__NGU_HALF_LOAD_PAIR__",
            "#define NGU_LOAD_PAIR(i2, uvar, wa, wb) uvec4 uvar = ubo[i2]; "
            "precise f16vec4 wa = ngu_half_bits(uvar.xy), wb = ngu_half_bits(uvar.zw)");
        replace_shader_text(s, "__NGU_HALF_MAC__",
            "#define NGU_MAC(acc, sv, wv) acc = fma(f16vec4(sv), wv, acc)");
    }
    return s;
}

// Check the actual compiler output before creating any pipeline. With only one
// floating type width, neither arithmetic nor intermediate storage can silently
// widen (or narrow). This also catches accidental FP32 literals/conversions in
// future generator edits. SPIR-V integer addressing is deliberately unrestricted.
inline void require_spirv_precision(std::span<const std::uint32_t> code, unsigned bits) {
    if (code.size() < 5 || code[0] != 0x07230203u)
        throw std::runtime_error("invalid Vulkan shader SPIR-V");
    bool found = false;
    for (std::size_t i = 5; i < code.size();) {
        const auto words = code[i] >> 16, opcode = code[i] & 0xffffu;
        if (!words || words > code.size() - i)
            throw std::runtime_error("invalid Vulkan shader SPIR-V instruction");
        if (opcode == 22) { // OpTypeFloat: result id, bit width
            if (words != 3 || code[i + 2] != bits)
                throw std::runtime_error("Vulkan shader violates full FP" + std::to_string(bits) +
                    " precision contract (unexpected floating-point type)");
            found = true;
        }
        i += words;
    }
    if (!found)
        throw std::runtime_error("Vulkan inference shader has no floating-point type");
}

} // namespace ngu
