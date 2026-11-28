#define VK_NO_PROTOTYPES
#include <VSVulkan4.h>
#include "ngu_vulkan.hpp"
#include "ngu_half.hpp"
#include "ngu_half_shader.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef NGU_DUMP_SHADERS
#include <cstdlib>
#include <filesystem>
#include <fstream>
#endif

namespace ngu {
namespace {

// Fallback compute kernel for unsupported topologies or small UBO/shared-memory
// limits. precise propagates NoContraction to the arithmetic feeding each result.
constexpr char fallback_shader_source[] = R"glsl(#version 460
#extension GL_EXT_spirv_intrinsics : require
// SPIR-V float controls: SignedZeroInfNanPreserve and RoundingModeRTE (FP32).
// DenormPreserve is not universally supported, even on Vulkan 1.4 devices.
spirv_execution_mode(capabilities=[4466], 4461, 32);
spirv_execution_mode(capabilities=[4467], 4462, 32);
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };
layout(set=0, binding=1, std430) writeonly buffer Destination { float d[]; };
layout(set=0, binding=2, std430) readonly buffer Parameters { float p[]; };
float parameter_at(uint i) { return p[i]; };
layout(set=0, binding=3, std430) readonly buffer Auxiliary { float a[]; };
layout(push_constant, std430) uniform Push {
    uint op, width, height, ci, co, src_stride, dst_stride, aux_stride;
    uint weights, bias;
} pc;

float sample_at(int x, int y, uint c) {
    return s[uint(clamp(y, 0, int(pc.height)-1))*pc.src_stride +
             uint(clamp(x, 0, int(pc.width)-1))*pc.ci + c];
}
vec4 param4(uint i) { return vec4(parameter_at(i), parameter_at(i+1), parameter_at(i+2), parameter_at(i+3)); }

void main() {
    uint x=gl_GlobalInvocationID.x, y=gl_GlobalInvocationID.y;
    uint ow=pc.width, oh=pc.height;
    if (pc.op==1) ow*=2;
    if (pc.op==2) oh*=2;
    if (pc.op==3) { ow+=2; oh+=2; }
    if (pc.op==4 || pc.op==5) { ow*=2; oh*=2; }
    if (x>=ow || y>=oh) return;
    uint out_index=y*pc.dst_stride;
    if (pc.op==0 || pc.op==6) {
        uint oc=gl_GlobalInvocationID.z*4;
        precise vec4 sum=param4(pc.bias+oc);
        for (int ky=0; ky<3; ++ky)
            for (int kx=0; kx<3; ++kx)
                for (uint ic=0; ic<pc.ci; ++ic) {
                    float v=sample_at(int(x)+kx-1, int(y)+ky-1, ic);
                    uint wi=pc.weights+((uint(ky*3+kx)*pc.ci+ic)*pc.co+oc);
                    sum += v*param4(wi);
                }
        if (pc.op==0)
            for (int c=0; c<4; ++c)
                if (sum[c]<0.0) sum[c]*=0.1;
        out_index+=x*pc.co+oc;
        for (uint c=0; c<4; ++c) d[out_index+c]=sum[c];
    } else if (pc.op==1 || pc.op==2) {
        uint pos=pc.op==1 ? x : y;
        int first=int(pos/2)-((pos&1)!=0 ? 1 : 2);
        float ca=uintBitsToFloat(0x3f5f3333u), cb=uintBitsToFloat(0x3e766666u);
        float cc=uintBitsToFloat(0xbdaccccdu), cd=uintBitsToFloat(0xbce66666u);
        float coeff[4];
        if ((pos&1)==0) { coeff[0]=cd; coeff[1]=cb; coeff[2]=ca; coeff[3]=cc; }
        else { coeff[0]=cc; coeff[1]=ca; coeff[2]=cb; coeff[3]=cd; }
        precise float sum=0.0;
        for (int k=0; k<4; ++k)
            sum+=coeff[k]*(pc.op==1 ? sample_at(first+k,int(y),0) : sample_at(int(x),first+k,0));
        d[out_index+x]=sum;
    } else if (pc.op==3) {
        precise vec4 sum=vec4(parameter_at(pc.bias));
        for (uint ic=0; ic<pc.ci; ++ic)
            for (int dy=0; dy<2; ++dy)
                for (int dx=0; dx<2; ++dx) {
                    float v=sample_at(int(x)-dx,int(y)-dy,ic);
                    uint wi=pc.weights+ic*16+uint(dy*8+dx*2);
                    sum+=v*vec4(parameter_at(wi),parameter_at(wi+1),parameter_at(wi+4),parameter_at(wi+5));
                }
        for (uint c=0; c<4; ++c) d[out_index+x*4+c]=sum[c];
    } else if (pc.op==4) {
        uint lane=3-(x&1)-2*(y&1);
        precise float v=a[y*pc.aux_stride+x]+s[((y+1)/2)*pc.src_stride+((x+1)/2)*4+lane];
        d[out_index+x]=v;
    } else {
        uint ix=x/2, iy=y/2;
        precise float v=sample_at(int(ix),int(iy),0);
        uint hi=iy*pc.aux_stride+ix*4;
        if ((x&1)!=0 && (y&1)!=0) {
            v=((v+sample_at(int(ix)+1,int(iy),0))+sample_at(int(ix),int(iy)+1,0))+
                sample_at(int(ix)+1,int(iy)+1,0);
            v=v*0.25+a[hi+2];
        } else if ((x&1)!=0) {
            v=(v+sample_at(int(ix)+1,int(iy),0))*0.5+a[hi+1];
        } else if ((y&1)!=0) {
            v=(v+sample_at(int(ix),int(iy)+1,0))*0.5+a[hi];
        }
        d[out_index+x]=v;
    }
}
)glsl";

constexpr char shader_preamble[] = R"glsl(#version 460
#define NGU_RAW uint
#define NGU_RAW2 uvec2
#define NGU_FROM_RAW uintBitsToFloat
#define NGU_TO_RAW floatBitsToUint
#define NGU_PACK_RAW2(a, b) uvec2((a), floatBitsToUint(b))
#define NGU_PACK_F2(a, b) uvec2(floatBitsToUint(a), floatBitsToUint(b))
#extension GL_EXT_spirv_intrinsics : require
#extension GL_EXT_control_flow_attributes : enable
// SPIR-V float controls: SignedZeroInfNanPreserve and RoundingModeRTE (FP32).
spirv_execution_mode(capabilities=[4466], 4461, 32);
spirv_execution_mode(capabilities=[4467], 4462, 32);
#define NGU_UBO_DECL(n4, n8) layout(set=0, binding=2, std140) uniform Params { vec4 ubo[n4]; };
#define NGU_P0(i2) ubo[(i2) * 2u]
#define NGU_P1(i2) ubo[(i2) * 2u + 1u]
#define NGU_LOAD_PAIR(i2, uvar, wa, wb) vec4 wa = ubo[(i2) * 2u], wb = ubo[(i2) * 2u + 1u]
#define NGU_MAC(acc, sv, wv) acc = fma(vec4(sv), wv, acc)
#define NGU_LEAKY(v) max((v), (v) * 0.1)
)glsl";

void vk_check(VkResult result, const char *operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed (VkResult " + std::to_string(result) +
                                 ")");
}

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1)
        return value;
    return ((value + alignment - 1) / alignment) * alignment;
}

struct BufferDeleter {
    const VSVULKANAPI *api;
    void operator()(VSGPUBuffer *buffer) const noexcept {
        if (buffer)
            api->destroyGPUBuffer(buffer);
    }
};
using Buffer = std::unique_ptr<VSGPUBuffer, BufferDeleter>;

struct Exec {
    const VSVULKANAPI *api;
    VSGPUExecContext *context;
    ~Exec() {
        if (context)
            api->gpuExecAbandon(context);
    }
    void submit(std::uint64_t *signaled = nullptr) {
        char error[1024]{};
        if (api->gpuExecSubmit(std::exchange(context, nullptr), signaled, error, sizeof(error)))
            throw std::runtime_error(std::string("GPU submit: ") + error);
    }
};

struct FallbackPush {
    std::uint32_t op, width, height, ci, co, src_stride, dst_stride, aux_stride, weights, bias;
};
static_assert(sizeof(FallbackPush) == 40);

struct FastPush {
    std::uint32_t width, height, plane_stride, src_s, dst_s2, y0, y1, src_y0, dst_y0, src_y_max;
};
static_assert(sizeof(FastPush) == 40);

struct LayerInfo {
    std::uint32_t ci, co, weights, bias;
};

struct FastPass {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDeviceSize ubo_offset = 0;
    VkDeviceSize ubo_size = 0;
    std::uint32_t tile_w = 1;
    std::uint32_t tile_h = 1;
    std::uint32_t layers = 1; // vertical receptive-field radius (number of 3x3 stages in pass)
    int src_slice = -1;       // -1 = input frame, >=0 = workspace slice index
    int dst_slice = -2;       // -2 = output frame, >=0 = workspace slice index
};

void align_uvec4_floats(std::vector<float> &dst) {
    if (dst.size() % 8 != 0)
        dst.resize(dst.size() + (8 - (dst.size() % 8)), 0.0f);
}

void append_floats(std::vector<float> &dst, std::span<const float> src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

std::uint32_t pack_1toC_aligned(std::vector<float> &dst, const Layer &layer) {
    align_uvec4_floats(dst);
    const auto u_off = static_cast<std::uint32_t>(dst.size() / 8);
    const int co = layer.output_channels();
    const int g_out = (co + 3) / 4;
    const int padded_floats_per_tap = ((g_out + 1) & ~1) * 4;
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) {
            for (int o = 0; o < co; ++o)
                dst.push_back(layer.weight_at(o * 9 + ky * 3 + kx));
            for (int p = co; p < padded_floats_per_tap; ++p)
                dst.push_back(0.0f);
        }
    return u_off;
}

std::uint32_t pack_nc4_ordered(std::vector<float> &dst, const Layer &layer, bool col_order) {
    align_uvec4_floats(dst);
    const auto u_off = static_cast<std::uint32_t>(dst.size() / 8);
    const int ci = layer.input_channels();
    const int co = layer.output_channels();
    const int igs = (ci + 3) / 4;
    const int ogs = (co + 3) / 4;
    const int co_padded = ogs * 4;
    auto emit_tap = [&](int ky, int kx, int ig) {
        for (int c = 0; c < 4; ++c) {
            const int ic = ig * 4 + c;
            for (int o = 0; o < co_padded; ++o) {
                const float w = (ic < ci && o < co)
                    ? layer.weight_at(((o * ci + ic) * 3 + ky) * 3 + kx)
                    : 0.0f;
                dst.push_back(w);
            }
        }
    };
    if (col_order) {
        for (int kx = 0; kx < 3; ++kx)
            for (int ig = 0; ig < igs; ++ig)
                for (int ky = 0; ky < 3; ++ky)
                    emit_tap(ky, kx, ig);
    } else {
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx)
                for (int ig = 0; ig < igs; ++ig)
                    emit_tap(ky, kx, ig);
    }
    align_uvec4_floats(dst);
    return u_off;
}

std::uint32_t pack_aa_head_ordered(std::vector<float> &dst, const Layer &head, bool col_order) {
    return pack_nc4_ordered(dst, head, col_order);
}

std::uint32_t pack_k4_head_ordered(std::vector<float> &dst, const Layer &head, bool col_order) {
    align_uvec4_floats(dst);
    const auto u_off = static_cast<std::uint32_t>(dst.size() / 8);
    const int ci = head.input_channels();
    const int igs = (ci + 3) / 4;
    std::vector<float> spatial(static_cast<std::size_t>(9) * igs * 16, 0.0f);
    for (int sub = 0; sub < 4; ++sub) {
        const int sx = sub & 1, sy = sub >> 1;
        const int lane = 3 - sx - 2 * sy;
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int kx = (sx - dx) + 1;
                const int ky = (sy - dy) + 1;
                for (int ic = 0; ic < ci; ++ic)
                    spatial[((ky * 3 + kx) * igs * 16) + (ic >> 2) * 16 + (ic & 3) * 4 + sub] =
                        head.weight_at(ic * 16 + (lane / 2 + dy * 2) * 4 + lane % 2 + dx * 2);
            }
    }
    auto emit_tap = [&](int ky, int kx, int ig) {
        const std::size_t base = (static_cast<std::size_t>(ky * 3 + kx) * igs + ig) * 16;
        dst.insert(dst.end(), spatial.begin() + base, spatial.begin() + base + 16);
    };
    if (col_order) {
        for (int kx = 0; kx < 3; ++kx)
            for (int ig = 0; ig < igs; ++ig)
                for (int ky = 0; ky < 3; ++ky)
                    emit_tap(ky, kx, ig);
    } else {
        dst.insert(dst.end(), spatial.begin(), spatial.end());
    }
    align_uvec4_floats(dst);
    return u_off;
}

std::uint32_t pack_bias_aligned(std::vector<float> &dst, const Layer &layer) {
    align_uvec4_floats(dst);
    const auto u_off = static_cast<std::uint32_t>(dst.size() / 8);
    for (int i = 0; i < layer.output_channels(); ++i)
        dst.push_back(layer.bias_at(i));
    align_uvec4_floats(dst);
    return u_off;
}

std::uint32_t pack_k4_bias_aligned(std::vector<float> &dst, const Layer &head) {
    align_uvec4_floats(dst);
    const auto u_off = static_cast<std::uint32_t>(dst.size() / 8);
    const float bh = head.bias_at(0);
    dst.insert(dst.end(), 4, bh);
    align_uvec4_floats(dst);
    return u_off;
}

std::vector<float> pack_bias(const Layer &layer) {
    std::vector<float> out(layer.output_channels());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = layer.bias_at(i);
    return out;
}

struct StageDesc {
    int c_in;
    int c_out;
    bool col_order;
    bool is_head;
    std::uint32_t w_uoff;
    std::uint32_t b_uoff;
};

struct ShaderGenerator {
    bool half;

    static const char *fast_push_decl() {
        return "layout(push_constant, std430) uniform Push { "
               "uint width, height, plane_stride, src_s, dst_s2, y0, y1, src_y0, dst_y0, src_y_max; } pc;\n";
    }

    static std::string emit_bias_init_n(int g_out, const std::string &b_uoff_expr, const char *pfx,
                                        int n_px, const char *indent, int uid) {
        std::string r;
        for (int og = 0; og < g_out; og += 2) {
            const std::string i2 = "(" + b_uoff_expr + " + " + std::to_string(og / 2) + "u)";
            const std::string tag = std::to_string(uid) + "_" + std::to_string(og);
            if (og + 1 < g_out) {
                r += std::string(indent) + "NGU_LOAD_PAIR(" + i2 + ", ub_" + tag + ", ba_" + tag + ", bb_" + tag + ");\n";
                for (int p = 0; p < n_px; ++p) {
                    r += std::string(indent) + pfx + std::to_string(p) + "_" + std::to_string(og) + " = ba_" + tag + ";\n";
                    r += std::string(indent) + pfx + std::to_string(p) + "_" + std::to_string(og + 1) + " = bb_" + tag + ";\n";
                }
            } else {
                r += std::string(indent) + "vec4 ba_" + tag + " = NGU_P0(" + i2 + ");\n";
                const std::string sfx = (g_out == 1) ? "" : ("_" + std::to_string(og));
                for (int p = 0; p < n_px; ++p)
                    r += std::string(indent) + pfx + std::to_string(p) + sfx + " = ba_" + tag + ";\n";
            }
        }
        return r;
    }

    // 1. Low Quality 1-Pass Pure Warp-Stripe (zero workspace VRAM, zero intermediate DRAM, 100% literal UBO):
    //    - FP16 primary: 58x26 (16w, 2py, px=2, Spatial exact, smem=38.4 KB) / portable: 26x26 (8w, 4py, px=1, smem=19.9 KB)
    //    - FP32 primary: 26x26 (16w, 2py, px=1, Col, smem=39.7 KB) / portable: 26x18 (8w, 3py, px=1, Col, smem=29.9 KB)
    std::string gen_low_stripe(std::span<const StageDesc> stages, std::uint32_t total_uvec4,
                               int wg_y, int py, int px, bool aa) const {
        const int s0_w = 32 * px;
        const int s0_h = wg_y * py;
        const int tw = s0_w - 6;
        const int th = s0_h - 6;
        const int src_w = s0_w + 2;
        const int src_h = s0_h + 2;
        const int tile_w_pad = s0_w + 2;
        const char *comps[4] = {"x", "y", "z", "w"};
        const std::string rd_pre = aa ? "NGU_FROM_RAW(" : "";
        const std::string rd_post = aa ? ")" : "";

        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=" + std::to_string(wg_y) + ", local_size_z=1) in;\n";
        s += aa ? "layout(set=0, binding=0, std430) readonly buffer Source { NGU_RAW s[]; };\n"
                  "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW2 d2[]; };\n"
                : "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
                  "layout(set=0, binding=1, std430) writeonly buffer Dest4K { vec2 d2[]; };\n";
        s += "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n"
             "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n";
        s += fast_push_decl();
        s += (aa ? "shared NGU_RAW src_t[" : "shared float src_t[") +
             std::to_string(src_h) + "][" + std::to_string(src_w + 2) + "];\n";
        s += "shared vec4 tile[2][" + std::to_string(s0_h) + "][" + std::to_string(tile_w_pad) + "];\n"
             "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * " + std::to_string(tw) +
             ", by = int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    {\n";
        for (int cx = 0; cx < px; ++cx)
            s += "        uint gx" + std::to_string(cx) + " = uint(clamp(bx + " + std::to_string(cx * 32) +
                 " + int(tx) - 4, 0, int(pc.width) - 1));\n";
        s += "        uint gx_r = uint(clamp(bx + " + std::to_string(s0_w) + " + int(tx) - 4, 0, int(pc.width) - 1));\n"
             "        for (uint sy = ty; sy < " + std::to_string(src_h) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "            uint row = uint(clamp(by + int(sy) - 4, 0, int(pc.height) - 1)) * pc.src_s;\n";
        for (int cx = 0; cx < px; ++cx)
            s += "            src_t[sy][" + std::to_string(cx * 32) + "u + tx] = s[row + gx" + std::to_string(cx) + "];\n";
        s += "            if (tx < 2u) src_t[sy][" + std::to_string(s0_w) + "u + tx] = s[row + gx_r];\n"
             "        }\n    }\n    barrier();\n";

        // Stage 0: b0 (1 -> 8 Spatial) on s0_w x s0_h (100% active)
        s += "    {\n";
        for (int cx = 0; cx < px; ++cx)
            for (int p = 0; p < py; ++p)
                s += "        vec4 q" + std::to_string(cx) + "_" + std::to_string(p) + "_0, q" +
                     std::to_string(cx) + "_" + std::to_string(p) + "_1;\n";
        s += "        { NGU_LOAD_PAIR(" + std::to_string(stages[0].b_uoff) + "u, ub0, ba0, bb0); ";
        for (int cx = 0; cx < px; ++cx)
            for (int p = 0; p < py; ++p)
                s += "q" + std::to_string(cx) + "_" + std::to_string(p) + "_0=ba0; q" +
                     std::to_string(cx) + "_" + std::to_string(p) + "_1=bb0; ";
        s += "}\n";
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                const std::string tg = "b0_" + std::to_string(ky * 3 + kx);
                s += "        { NGU_LOAD_PAIR(" + std::to_string(stages[0].w_uoff + ky * 3 + kx) +
                     "u, uw_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                for (int cx = 0; cx < px; ++cx)
                    for (int p = 0; p < py; ++p) {
                        const std::string sv = "sv" + std::to_string(cx) + "_" + std::to_string(p);
                        const std::string rd = "src_t[yl+" + std::to_string(ky + p) + "u][" +
                                               std::to_string(cx * 32 + kx) + "u+tx]";
                        s += "float " + sv + " = " + rd_pre + rd + rd_post + "; "
                             "NGU_MAC(q" + std::to_string(cx) + "_" + std::to_string(p) + "_0, " + sv + ", wa_" + tg + "); "
                             "NGU_MAC(q" + std::to_string(cx) + "_" + std::to_string(p) + "_1, " + sv + ", wb_" + tg + "); ";
                    }
                s += "}\n";
            }
        for (int cx = 0; cx < px; ++cx)
            for (int p = 0; p < py; ++p) {
                const std::string q0 = "q" + std::to_string(cx) + "_" + std::to_string(p) + "_0";
                const std::string q1 = "q" + std::to_string(cx) + "_" + std::to_string(p) + "_1";
                s += "        tile[0][yl+" + std::to_string(p) + "u][" + std::to_string(cx * 32) + "u+tx] = NGU_LEAKY(" + q0 + ");\n"
                     "        tile[1][yl+" + std::to_string(p) + "u][" + std::to_string(cx * 32) + "u+tx] = NGU_LEAKY(" + q1 + ");\n";
            }
        s += "    }\n    barrier();\n";

        // Stages 1 & 2: b1 (8->8) and b2 (8->8), 100% literal UBO addressing
        const auto emit_mid = [&](const StageDesc &st, int stg_id, int pad, int max_x, int max_y, int out_w, int out_h) {
            s += "    {\n";
            for (int cx = 0; cx < px; ++cx)
                for (int p = 0; p < py; ++p)
                    s += "        vec4 q" + std::to_string(cx) + "_" + std::to_string(p) + "_0, q" +
                         std::to_string(cx) + "_" + std::to_string(p) + "_1;\n";
            s += "        if (yl < " + std::to_string(out_h) + "u) {\n"
                 "            { NGU_LOAD_PAIR(" + std::to_string(st.b_uoff) + "u, ub" + std::to_string(stg_id) + ", ba, bb); ";
            for (int cx = 0; cx < px; ++cx)
                for (int p = 0; p < py; ++p)
                    s += "q" + std::to_string(cx) + "_" + std::to_string(p) + "_0=ba; q" +
                         std::to_string(cx) + "_" + std::to_string(p) + "_1=bb; ";
            s += "}\n"
                 "            int xl = max(0, " + std::to_string(pad) + " - bx), xh = min(" + std::to_string(max_x) +
                 ", int(pc.width) - 1 + " + std::to_string(pad) + " - bx);\n"
                 "            int ylo = max(0, " + std::to_string(pad) + " - by), yhi = min(" + std::to_string(max_y) +
                 ", int(pc.height) - 1 + " + std::to_string(pad) + " - by);\n";
            for (int r = 0; r < py + 2; ++r)
                s += "            uint ry" + std::to_string(r) + " = uint(clamp(int(yl) + " + std::to_string(r) + ", ylo, yhi));\n";
            if (st.col_order) {
                for (int kx = 0; kx < 3; ++kx) {
                    s += "            {\n";
                    for (int cx = 0; cx < px; ++cx)
                        s += "                uint rx" + std::to_string(cx) + " = uint(clamp(" +
                             std::to_string(cx * 32 + kx) + " + int(tx), xl, xh));\n";
                    for (int ig = 0; ig < 2; ++ig) {
                        s += "                {\n";
                        for (int cx = 0; cx < px; ++cx)
                            for (int r = 0; r < py + 2; ++r)
                                s += "                    vec4 t" + std::to_string(cx) + "_" + std::to_string(r) +
                                     " = tile[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx" + std::to_string(cx) + "];\n";
                        const std::uint32_t base = st.w_uoff + static_cast<std::uint32_t>((kx * 2 + ig) * 12);
                        for (int ky = 0; ky < 3; ++ky)
                            for (int c = 0; c < 4; ++c) {
                                const std::string tg = "m" + std::to_string(stg_id) + "_" + std::to_string(kx) + "_" +
                                                       std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(c);
                                s += "                    { NGU_LOAD_PAIR(" + std::to_string(base + ky * 4 + c) +
                                     "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                                for (int cx = 0; cx < px; ++cx)
                                    for (int p = 0; p < py; ++p)
                                        s += "NGU_MAC(q" + std::to_string(cx) + "_" + std::to_string(p) + "_0, t" +
                                             std::to_string(cx) + "_" + std::to_string(ky + p) + "." + comps[c] + ", wa_" + tg + "); "
                                             "NGU_MAC(q" + std::to_string(cx) + "_" + std::to_string(p) + "_1, t" +
                                             std::to_string(cx) + "_" + std::to_string(ky + p) + "." + comps[c] + ", wb_" + tg + "); ";
                                s += "}\n";
                            }
                        s += "                }\n";
                    }
                    s += "            }\n";
                }
            } else {
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 3; ++kx) {
                        s += "            {\n";
                        for (int cx = 0; cx < px; ++cx)
                            s += "                uint rx" + std::to_string(cx) + " = uint(clamp(" +
                                 std::to_string(cx * 32 + kx) + " + int(tx), xl, xh));\n";
                        for (int ig = 0; ig < 2; ++ig) {
                            s += "                {\n";
                            for (int cx = 0; cx < px; ++cx)
                                for (int p = 0; p < py; ++p)
                                    s += "                    vec4 t" + std::to_string(cx) + "_" + std::to_string(p) +
                                         " = tile[" + std::to_string(ig) + "][ry" + std::to_string(ky + p) + "][rx" + std::to_string(cx) + "];\n";
                            const std::uint32_t base = st.w_uoff + static_cast<std::uint32_t>(((ky * 3 + kx) * 2 + ig) * 4);
                            for (int c = 0; c < 4; ++c) {
                                const std::string tg = "s" + std::to_string(stg_id) + "_" + std::to_string(ky) + "_" +
                                                       std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(c);
                                s += "                    { NGU_LOAD_PAIR(" + std::to_string(base + c) +
                                     "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                                for (int cx = 0; cx < px; ++cx)
                                    for (int p = 0; p < py; ++p)
                                        s += "NGU_MAC(q" + std::to_string(cx) + "_" + std::to_string(p) + "_0, t" +
                                             std::to_string(cx) + "_" + std::to_string(p) + "." + comps[c] + ", wa_" + tg + "); "
                                             "NGU_MAC(q" + std::to_string(cx) + "_" + std::to_string(p) + "_1, t" +
                                             std::to_string(cx) + "_" + std::to_string(p) + "." + comps[c] + ", wb_" + tg + "); ";
                                s += "}\n";
                            }
                            s += "                }\n";
                        }
                        s += "            }\n";
                    }
            }
            s += "        }\n        barrier();\n";
            for (int p = 0; p < py; ++p) {
                s += "        if (yl + " + std::to_string(p) + "u < " + std::to_string(out_h) + "u) {\n";
                for (int cx = 0; cx < px; ++cx) {
                    if (cx == px - 1)
                        s += "            if (" + std::to_string(cx * 32) + "u + tx < " + std::to_string(out_w) + "u) {\n";
                    const std::string q0 = "q" + std::to_string(cx) + "_" + std::to_string(p) + "_0";
                    const std::string q1 = "q" + std::to_string(cx) + "_" + std::to_string(p) + "_1";
                    s += "            tile[0][yl+" + std::to_string(p) + "u][" + std::to_string(cx * 32) + "u+tx] = NGU_LEAKY(" + q0 + ");\n"
                         "            tile[1][yl+" + std::to_string(p) + "u][" + std::to_string(cx * 32) + "u+tx] = NGU_LEAKY(" + q1 + ");\n";
                    if (cx == px - 1)
                        s += "            }\n";
                }
                s += "        }\n";
            }
            s += "        barrier();\n    }\n";
        };
        emit_mid(stages[1], 1, 3, s0_w - 1, s0_h - 1, s0_w - 2, s0_h - 2);
        emit_mid(stages[2], 2, 2, s0_w - 3, s0_h - 3, s0_w - 4, s0_h - 4);

        // Stage 3: Head (8 -> 4) on tw x th + combine
        for (int cx = 0; cx < px; ++cx)
            for (int p = 0; p < py; ++p)
                s += "    vec4 h" + std::to_string(cx) + "_" + std::to_string(p) + ";\n";
        s += "    if (yl < " + std::to_string(th) + "u) {\n"
             "        { vec4 bah = NGU_P0(" + std::to_string(stages[3].b_uoff) + "u); ";
        for (int cx = 0; cx < px; ++cx)
            for (int p = 0; p < py; ++p)
                s += "h" + std::to_string(cx) + "_" + std::to_string(p) + "=bah; ";
        s += "}\n"
             "        int xl = max(0, 1 - bx), xh = min(" + std::to_string(tw + 1) + ", int(pc.width) - bx);\n"
             "        int ylo = max(0, 1 - by), yhi = min(" + std::to_string(th + 1) + ", int(pc.height) - by);\n";
        for (int r = 0; r < py + 2; ++r)
            s += "        uint ry" + std::to_string(r) + " = uint(clamp(int(yl) + " + std::to_string(r) + ", ylo, yhi));\n";
        if (stages[3].col_order) {
            for (int kx = 0; kx < 3; ++kx) {
                s += "        {\n";
                for (int cx = 0; cx < px; ++cx)
                    s += "            uint rx" + std::to_string(cx) + " = uint(clamp(" +
                         std::to_string(cx * 32 + kx) + " + int(tx), xl, xh));\n";
                for (int ig = 0; ig < 2; ++ig) {
                    s += "            {\n";
                    for (int cx = 0; cx < px; ++cx)
                        for (int r = 0; r < py + 2; ++r)
                            s += "                vec4 t" + std::to_string(cx) + "_" + std::to_string(r) +
                                 " = tile[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx" + std::to_string(cx) + "];\n";
                    const std::uint32_t base = stages[3].w_uoff + static_cast<std::uint32_t>((kx * 2 + ig) * 6);
                    for (int ky = 0; ky < 3; ++ky)
                        for (int c = 0; c < 4; c += 2) {
                            const std::string tg = "hc_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" +
                                                   std::to_string(ky) + "_" + std::to_string(c);
                            s += "                { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 + c / 2) +
                                 "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int cx = 0; cx < px; ++cx)
                                for (int p = 0; p < py; ++p)
                                    s += "NGU_MAC(h" + std::to_string(cx) + "_" + std::to_string(p) + ", t" +
                                         std::to_string(cx) + "_" + std::to_string(ky + p) + "." + comps[c] + ", wa_" + tg + "); "
                                         "NGU_MAC(h" + std::to_string(cx) + "_" + std::to_string(p) + ", t" +
                                         std::to_string(cx) + "_" + std::to_string(ky + p) + "." + comps[c + 1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                    s += "            }\n";
                }
                s += "        }\n";
            }
        } else {
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    s += "        {\n";
                    for (int cx = 0; cx < px; ++cx)
                        s += "            uint rx" + std::to_string(cx) + " = uint(clamp(" +
                             std::to_string(cx * 32 + kx) + " + int(tx), xl, xh));\n";
                    for (int ig = 0; ig < 2; ++ig) {
                        s += "            {\n";
                        for (int cx = 0; cx < px; ++cx)
                            for (int p = 0; p < py; ++p)
                                s += "                vec4 t" + std::to_string(cx) + "_" + std::to_string(p) +
                                     " = tile[" + std::to_string(ig) + "][ry" + std::to_string(ky + p) + "][rx" + std::to_string(cx) + "];\n";
                        const std::uint32_t base = stages[3].w_uoff + static_cast<std::uint32_t>(((ky * 3 + kx) * 2 + ig) * 2);
                        for (int c = 0; c < 4; c += 2) {
                            const std::string tg = "hs_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" +
                                                   std::to_string(ig) + "_" + std::to_string(c);
                            s += "                { NGU_LOAD_PAIR(" + std::to_string(base + c / 2) +
                                 "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int cx = 0; cx < px; ++cx)
                                for (int p = 0; p < py; ++p)
                                    s += "NGU_MAC(h" + std::to_string(cx) + "_" + std::to_string(p) + ", t" +
                                         std::to_string(cx) + "_" + std::to_string(p) + "." + comps[c] + ", wa_" + tg + "); "
                                         "NGU_MAC(h" + std::to_string(cx) + "_" + std::to_string(p) + ", t" +
                                         std::to_string(cx) + "_" + std::to_string(p) + "." + comps[c + 1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                        s += "            }\n";
                    }
                    s += "        }\n";
                }
        }
        s += "    }\n";

        if (!aa) {
            s += "    barrier();\n"
                 "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint sy = ty; sy < " + std::to_string(th + 4) + "u; sy += " + std::to_string(wg_y) + "u) {\n";
            for (int cx = 0; cx < px; ++cx) {
                if (cx == px - 1)
                    s += "            if (" + std::to_string(cx * 32) + "u + tx < " + std::to_string(tw) + "u) {\n";
                s += "            float s0=src_t[sy+2u][" + std::to_string(cx * 32 + 2) + "u+tx], s1=src_t[sy+2u][" +
                     std::to_string(cx * 32 + 3) + "u+tx], s2=src_t[sy+2u][" + std::to_string(cx * 32 + 4) +
                     "u+tx], s3=src_t[sy+2u][" + std::to_string(cx * 32 + 5) + "u+tx], s4=src_t[sy+2u][" +
                     std::to_string(cx * 32 + 6) + "u+tx];\n"
                     "            float bx0 = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                     "            float bx1 = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                     "            tile[0][sy][" + std::to_string(cx * 32) + "u+tx].xy = vec2(bx0, bx1);\n";
                if (cx == px - 1)
                    s += "            }\n";
            }
            s += "        }\n    }\n    barrier();\n"
                 "    if (yl >= " + std::to_string(th) + "u) return;\n"
                 "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n";
            const int max_bx_row = th + 3;
            for (int cx = 0; cx < px; ++cx) {
                s += "    {\n        uint lx = " + std::to_string(cx * 32) + "u + tx, x = uint(bx) + lx;\n"
                     "        if (lx < " + std::to_string(tw) + "u && x < pc.width) {\n";
                for (int r = 0; r < py + 4; ++r)
                    s += "            vec2 r" + std::to_string(r) + " = tile[0][min(yl + " + std::to_string(r) +
                         "u, " + std::to_string(max_bx_row) + "u)][lx].xy;\n";
                for (int p = 0; p < py; ++p) {
                    const std::string hn = "h" + std::to_string(cx) + "_" + std::to_string(p);
                    s += "            if (yl + " + std::to_string(p) + "u < " + std::to_string(th) +
                         "u && uint(by) + yl + " + std::to_string(p) + "u < pc.height) {\n"
                         "                uint oy = (uint(by) + yl + " + std::to_string(p) + "u) * 2u;\n"
                         "                vec2 top = ((((cd*r" + std::to_string(p) + ")+(cb*r" + std::to_string(p + 1) +
                         "))+(ca*r" + std::to_string(p + 2) + "))+(cc*r" + std::to_string(p + 3) + ")) + " + hn + ".xy;\n"
                         "                vec2 bot = ((((cc*r" + std::to_string(p + 1) + ")+(ca*r" + std::to_string(p + 2) +
                         "))+(cb*r" + std::to_string(p + 3) + "))+(cd*r" + std::to_string(p + 4) + ")) + " + hn + ".zw;\n"
                         "                d2[(oy + 0u) * pc.dst_s2 + x] = top;\n"
                         "                d2[(oy + 1u) * pc.dst_s2 + x] = bot;\n"
                         "            }\n";
                }
                s += "        }\n    }\n";
            }
        } else {
            s += "    if (yl >= " + std::to_string(th) + "u) return;\n";
            const int max_src_row = src_h - 1;
            for (int cx = 0; cx < px; ++cx) {
                s += "    {\n        uint lx = " + std::to_string(cx * 32) + "u + tx, x = uint(bx) + lx;\n"
                     "        if (lx < " + std::to_string(tw) + "u && x < pc.width) {\n";
                for (int r = 0; r <= py; ++r)
                    s += "            uint sr" + std::to_string(r) + " = min(yl + " + std::to_string(r + 4) +
                         "u, " + std::to_string(max_src_row) + "u);\n"
                         "            NGU_RAW u" + std::to_string(r) + "_0 = src_t[sr" + std::to_string(r) + "][lx+4u];\n"
                         "            NGU_RAW u" + std::to_string(r) + "_1 = src_t[sr" + std::to_string(r) + "][lx+5u];\n"
                         "            float s" + std::to_string(r) + "_0 = NGU_FROM_RAW(u" + std::to_string(r) + "_0);\n"
                         "            float s" + std::to_string(r) + "_1 = NGU_FROM_RAW(u" + std::to_string(r) + "_1);\n";
                for (int p = 0; p < py; ++p) {
                    const std::string hn = "h" + std::to_string(cx) + "_" + std::to_string(p);
                    s += "            if (yl + " + std::to_string(p) + "u < " + std::to_string(th) +
                         "u && uint(by) + yl + " + std::to_string(p) + "u < pc.height) {\n"
                         "                uint oy = (uint(by) + yl + " + std::to_string(p) + "u) * 2u;\n"
                         "                float p10 = (s" + std::to_string(p) + "_0 + s" + std::to_string(p) + "_1) * 0.5 + " + hn + ".y;\n"
                         "                float p01 = (s" + std::to_string(p) + "_0 + s" + std::to_string(p + 1) + "_0) * 0.5 + " + hn + ".x;\n"
                         "                float p11 = (((s" + std::to_string(p) + "_0 + s" + std::to_string(p) + "_1) + s" +
                         std::to_string(p + 1) + "_0) + s" + std::to_string(p + 1) + "_1) * 0.25 + " + hn + ".z;\n"
                         "                d2[(oy + 0u) * pc.dst_s2 + x] = NGU_PACK_RAW2(u" + std::to_string(p) + "_0, p10);\n"
                         "                d2[(oy + 1u) * pc.dst_s2 + x] = NGU_PACK_F2(p01, p11);\n"
                         "            }\n";
                }
                s += "        }\n    }\n";
            }
        }
        s += "}\n";
        return s;
    }

    // 2. Zero-Waste b0 + b1 (1 -> C_mid -> C_out) on EXACT 32 x (wg_y * py) strip-aware output tile:
    //    b0 fills 34 x (th + 2) in smem so heavy C_mid -> C_out b1 runs at 100.0% warp efficiency!
    std::string gen_zw_b0_b1(std::span<const StageDesc> stages, std::uint32_t total_uvec4,
                             int wg_y, int py) const {
        const int g_mid = stages[0].c_out / 4;
        const int g_out = stages[1].c_out / 4;
        const int th = wg_y * py;
        const int b0_h = th + 2;
        const int src_h = th + 4;
        const int uvec4_per_tap = (g_mid + 1) / 2;
        const char *comps[4] = {"x", "y", "z", "w"};

        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=" + std::to_string(wg_y) + ", local_size_z=1) in;\n"
             "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
             "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n"
             "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n"
             "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n";
        s += fast_push_decl();
        s += "shared float src_t[" + std::to_string(src_h) + "][36];\n"
             "shared vec4 smem[" + std::to_string(g_mid) + "][" + std::to_string(b0_h) + "][34];\n"
             "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 32u + tx, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 32, by = int(pc.y0) + int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    {\n"
             "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "        for (uint sy = ty; sy < " + std::to_string(src_h) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "            uint srow = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
             "            src_t[sy][tx] = s[srow + gx];\n"
             "            if (tx < 4u) src_t[sy][32u + tx] = s[srow + gx_r];\n"
             "        }\n    }\n    barrier();\n"
             "    for (uint idx = tid; idx < " + std::to_string(34 * b0_h) + "u; idx += " + std::to_string(32 * wg_y) + "u) {\n"
             "        uint ly = idx / 34u, lx = idx - ly * 34u;\n";
        for (int og = 0; og < g_mid; ++og)
            s += "        vec4 q" + std::to_string(og) + ";\n";
        for (int og = 0; og < g_mid; og += 2) {
            const std::string i2 = std::to_string(stages[0].b_uoff + og / 2) + "u";
            const std::string tg = "b0_" + std::to_string(og);
            if (og + 1 < g_mid)
                s += "        { NGU_LOAD_PAIR(" + i2 + ", ub_" + tg + ", ba_" + tg + ", bb_" + tg + "); q" +
                     std::to_string(og) + "=ba_" + tg + "; q" + std::to_string(og + 1) + "=bb_" + tg + "; }\n";
            else
                s += "        q" + std::to_string(og) + " = NGU_P0(" + i2 + ");\n";
        }
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                const int tap = ky * 3 + kx;
                s += "        { float sv = src_t[ly+" + std::to_string(ky) + "u][lx+" + std::to_string(kx) + "u]; ";
                for (int og = 0; og < g_mid; og += 2) {
                    const std::string i2 = std::to_string(stages[0].w_uoff + tap * uvec4_per_tap + og / 2) + "u";
                    const std::string tg = "s0_" + std::to_string(tap) + "_" + std::to_string(og);
                    if (og + 1 < g_mid)
                        s += "{ NGU_LOAD_PAIR(" + i2 + ", uw_" + tg + ", wa_" + tg + ", wb_" + tg + "); "
                             "NGU_MAC(q" + std::to_string(og) + ", sv, wa_" + tg + "); "
                             "NGU_MAC(q" + std::to_string(og + 1) + ", sv, wb_" + tg + "); } ";
                    else
                        s += "{ vec4 wa_" + tg + " = NGU_P0(" + i2 + "); NGU_MAC(q" + std::to_string(og) + ", sv, wa_" + tg + "); } ";
                }
                s += "}\n";
            }
        for (int og = 0; og < g_mid; ++og)
            s += "        smem[" + std::to_string(og) + "][ly][lx] = NGU_LEAKY(q" + std::to_string(og) + ");\n";
        s += "    }\n    barrier();\n"
             "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n"
             "    if (x >= pc.width || y0 >= pc.y1) return;\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g_out; ++og)
                s += "    vec4 p" + std::to_string(p) + "_" + std::to_string(og) + ";\n";
        s += emit_bias_init_n(g_out, std::to_string(stages[1].b_uoff) + "u", "p", py, "    ", 1) +
             "    int xl = max(0, 1 - bx), xh = min(33, int(pc.width) - bx);\n"
             "    int ylo = max(0, 1 - by), yhi = min(" + std::to_string(th + 1) + ", int(pc.height) - by);\n";
        for (int r = 0; r < py + 2; ++r)
            s += "    uint ry" + std::to_string(r) + " = uint(clamp(int(yl) + " + std::to_string(r) + ", ylo, yhi));\n";
        for (int kx = 0; kx < 3; ++kx) {
            s += "    { uint rx = uint(clamp(int(tx) + " + std::to_string(kx) + ", xl, xh));\n";
            for (int ig = 0; ig < g_mid; ++ig) {
                s += "      {\n";
                for (int r = 0; r < py + 2; ++r)
                    s += "        vec4 t" + std::to_string(r) + " = smem[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx];\n";
                const std::uint32_t base = stages[1].w_uoff + static_cast<std::uint32_t>((kx * g_mid + ig) * (6 * g_out));
                for (int ky = 0; ky < 3; ++ky)
                    for (int f = 0; f < 4 * g_out; f += 2) {
                        const int c0 = f / g_out, og0 = f % g_out, c1 = (f + 1) / g_out, og1 = (f + 1) % g_out;
                        const std::string tg = "u1_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" +
                                               std::to_string(ky) + "_" + std::to_string(f / 2);
                        s += "        { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g_out + f / 2) +
                             "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                        for (int p = 0; p < py; ++p)
                            s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                        for (int p = 0; p < py; ++p)
                            s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                        s += "}\n";
                    }
                s += "      }\n";
            }
            s += "    }\n";
        }
        for (int p = 0; p < py; ++p) {
            if (p > 0)
                s += "    if (y0 + " + std::to_string(p) + "u < pc.y1) {\n";
            s += "    { uint ob = (y0 + " + std::to_string(p) + "u - pc.dst_y0) * pc.width + x;\n";
            for (int og = 0; og < g_out; ++og) {
                const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
                s += "      d[" + std::to_string(og) + "u * pc.plane_stride + ob] = NGU_LEAKY(" + pn + ");\n";
            }
            s += "    }\n";
            if (p > 0)
                s += "    }\n";
        }
        s += "}\n";
        return s;
    }

    // 3. Strip-Aware 1-Layer Warp-Stripe (32 x (wg_y * py)), 100% immediate UBO addressing
    std::string gen_1L_stripe(const StageDesc &st, std::uint32_t total_uvec4,
                              int wg_y, int py, bool spatial = false) const {
        const int g = st.c_in / 4;
        const int th = wg_y * py, rows = th + 2;
        const char *comps[4] = {"x", "y", "z", "w"};
        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=" + std::to_string(wg_y) + ", local_size_z=1) in;\n"
             "layout(set=0, binding=0, std430) readonly buffer Source { vec4 s[]; };\n"
             "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n"
             "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n"
             "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n";
        s += fast_push_decl();
        s += "shared vec4 tile[" + std::to_string(g) + "][" + std::to_string(rows) + "][34];\n"
             "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 32, by = int(pc.y0) + int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    uint gx = uint(clamp(bx + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "    uint gx_r = uint(clamp(bx + 32 + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "    for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "        uint cy = uint(clamp(by + int(sy) - 1, int(pc.src_y0), int(pc.src_y_max)));\n"
             "        uint row = (cy - pc.src_y0) * pc.width;\n"
             "        for (int gi = 0; gi < " + std::to_string(g) + "; ++gi) {\n"
             "            uint ps = uint(gi) * pc.plane_stride + row;\n"
             "            tile[gi][sy][tx] = s[ps + gx];\n"
             "            if (tx < 2u) tile[gi][sy][32u + tx] = s[ps + gx_r];\n"
             "        }\n    }\n    barrier();\n"
             "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n"
             "    if (x >= pc.width || y0 >= pc.y1) return;\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g; ++og)
                s += "    vec4 p" + std::to_string(p) + "_" + std::to_string(og) + ";\n";
        s += emit_bias_init_n(g, std::to_string(st.b_uoff) + "u", "p", py, "    ", 1);

        if (spatial) {
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    for (int ig = 0; ig < g; ++ig) {
                        s += "    {\n";
                        for (int p = 0; p < py; ++p)
                            s += "        vec4 t" + std::to_string(p) + " = tile[" + std::to_string(ig) + "][yl + " +
                                 std::to_string(ky + p) + "u][tx + " + std::to_string(kx) + "u];\n";
                        const std::uint32_t base = st.w_uoff + static_cast<std::uint32_t>(((ky * 3 + kx) * g + ig) * (2 * g));
                        for (int f = 0; f < 4 * g; f += 2) {
                            const int c0 = f / g, og0 = f % g, c1 = (f + 1) / g, og1 = (f + 1) % g;
                            const std::string tg = "sp_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" +
                                                   std::to_string(ig) + "_" + std::to_string(f / 2);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(p) + "." + comps[c0] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(p) + "." + comps[c1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                        s += "    }\n";
                    }
        } else {
            for (int kx = 0; kx < 3; ++kx)
                for (int ig = 0; ig < g; ++ig) {
                    s += "    {\n";
                    for (int r = 0; r < py + 2; ++r)
                        s += "        vec4 t" + std::to_string(r) + " = tile[" + std::to_string(ig) + "][yl + " +
                             std::to_string(r) + "u][tx + " + std::to_string(kx) + "u];\n";
                    const std::uint32_t base = st.w_uoff + static_cast<std::uint32_t>((kx * g + ig) * (6 * g));
                    for (int ky = 0; ky < 3; ++ky)
                        for (int f = 0; f < 4 * g; f += 2) {
                            const int c0 = f / g, og0 = f % g, c1 = (f + 1) / g, og1 = (f + 1) % g;
                            const std::string tg = std::to_string(kx) + "_" + std::to_string(ig) + "_" +
                                                   std::to_string(ky) + "_" + std::to_string(f / 2);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                    s += "    }\n";
                }
        }
        for (int p = 0; p < py; ++p) {
            if (p > 0)
                s += "    if (y0 + " + std::to_string(p) + "u < pc.y1) {\n";
            s += "    { uint ob = (y0 + " + std::to_string(p) + "u - pc.dst_y0) * pc.width + x;\n";
            for (int og = 0; og < g; ++og) {
                const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
                s += "      d[" + std::to_string(og) + "u * pc.plane_stride + ob] = NGU_LEAKY(" + pn + ");\n";
            }
            s += "    }\n";
            if (p > 0)
                s += "    }\n";
        }
        s += "}\n";
        return s;
    }

    // 4. Strip-Aware 2-Layer Warp-Stripe (30 x (wg_y*py - 2)): C->C->C or b_last + head + coalesced combine
    std::string gen_2L_stripe(std::span<const StageDesc> stages, std::uint32_t total_uvec4,
                              bool aa, int wg_y, int py) const {
        const bool tail_out = stages[1].is_head;
        const int g_in0 = stages[0].c_in / 4;
        const int g_mid = stages[0].c_out / 4;
        const int g_last = tail_out ? 1 : (stages[1].c_out / 4);
        const int g_smem = std::max(g_in0, g_mid);
        const int th = wg_y * py - 2, rows = wg_y * py + 2;
        const char *comps[4] = {"x", "y", "z", "w"};

        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=" + std::to_string(wg_y) + ", local_size_z=1) in;\n"
             "layout(set=0, binding=0, std430) readonly buffer Source { vec4 s[]; };\n";
        if (tail_out) {
            s += aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW2 d2[]; };\n"
                    : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { vec2 d2[]; };\n";
        } else {
            s += "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n";
        }
        s += "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n";
        if (tail_out) {
            s += aa ? "layout(set=0, binding=3, std430) readonly buffer OrigSrc { NGU_RAW src_in[]; };\n"
                    : "layout(set=0, binding=3, std430) readonly buffer OrigSrc { float src_in[]; };\n";
        } else {
            s += "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n";
        }
        s += fast_push_decl();
        s += "shared vec4 smem[" + std::to_string(g_smem) + "][" + std::to_string(rows) + "][34];\n";
        if (tail_out && aa)
            s += "shared NGU_RAW src_u_t[" + std::to_string(th + 1) + "][32];\n";
        else if (tail_out)
            s += "shared float src_t[" + std::to_string(rows) + "][36];\n";

        s += "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 30, by = int(pc.y0) + int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    {\n"
             "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "        for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "            uint cy = uint(clamp(by + int(sy) - 2, int(pc.src_y0), int(pc.src_y_max)));\n"
             "            uint row = (cy - pc.src_y0) * pc.width;\n"
             "            for (int gi = 0; gi < " + std::to_string(g_in0) + "; ++gi) {\n"
             "                uint ps = uint(gi) * pc.plane_stride + row;\n"
             "                smem[gi][sy][tx] = s[ps + gx];\n"
             "                if (tx < 2u) smem[gi][sy][32u + tx] = s[ps + gx_r];\n"
             "            }\n";
        if (tail_out && !aa) {
            s += "            uint ocy = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1));\n"
                 "            uint srow = ocy * pc.src_s;\n"
                 "            src_t[sy][tx] = src_in[srow + gx];\n"
                 "            if (tx < 4u) src_t[sy][32u + tx] = src_in[srow + gx_r];\n";
        }
        s += "        }\n    }\n";
        if (tail_out && aa) {
            s += "    if (tx < 31u) {\n"
                 "        uint sgx = uint(clamp(bx + int(tx), 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < " + std::to_string(th + 1) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                 "            uint srow = uint(clamp(by + int(sy), 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            src_u_t[sy][tx] = src_in[srow + sgx];\n"
                 "        }\n    }\n";
        }
        s += "    barrier();\n";

        // Stage 0 (100% active threads: 32 x (wg_y * py))
        s += "    {\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g_mid; ++og)
                s += "        vec4 q" + std::to_string(p) + "_" + std::to_string(og) + ";\n";
        s += emit_bias_init_n(g_mid, std::to_string(stages[0].b_uoff) + "u", "q", py, "        ", 10);
        if (stages[0].col_order) {
            for (int kx = 0; kx < 3; ++kx)
                for (int ig = 0; ig < g_in0; ++ig) {
                    s += "        {\n";
                    for (int r = 0; r < py + 2; ++r)
                        s += "            vec4 t" + std::to_string(r) + " = smem[" + std::to_string(ig) + "][yl+" +
                             std::to_string(r) + "u][tx+" + std::to_string(kx) + "u];\n";
                    const std::uint32_t base = stages[0].w_uoff + static_cast<std::uint32_t>((kx * g_in0 + ig) * (6 * g_mid));
                    for (int ky = 0; ky < 3; ++ky)
                        for (int f = 0; f < 4 * g_mid; f += 2) {
                            const int c0 = f / g_mid, og0 = f % g_mid, c1 = (f + 1) / g_mid, og1 = (f + 1) % g_mid;
                            const std::string tg = "c0_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" +
                                                   std::to_string(ky) + "_" + std::to_string(f / 2);
                            s += "            { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g_mid + f / 2) +
                                 "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                    s += "        }\n";
                }
        } else {
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    for (int ig = 0; ig < g_in0; ++ig) {
                        s += "        {\n";
                        for (int p = 0; p < py; ++p)
                            s += "            vec4 t" + std::to_string(p) + " = smem[" + std::to_string(ig) + "][yl+" +
                                 std::to_string(ky + p) + "u][tx+" + std::to_string(kx) + "u];\n";
                        const std::uint32_t base = stages[0].w_uoff + static_cast<std::uint32_t>(((ky * 3 + kx) * g_in0 + ig) * (2 * g_mid));
                        for (int f = 0; f < 4 * g_mid; f += 2) {
                            const int c0 = f / g_mid, og0 = f % g_mid, c1 = (f + 1) / g_mid, og1 = (f + 1) % g_mid;
                            const std::string tg = "sp_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" +
                                                   std::to_string(ig) + "_" + std::to_string(f / 2);
                            s += "            { NGU_LOAD_PAIR(" + std::to_string(base + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(p) + "." + comps[c0] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(p) + "." + comps[c1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                        s += "        }\n";
                    }
        }
        s += "        barrier();\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g_mid; ++og) {
                const std::string qn = "q" + std::to_string(p) + "_" + std::to_string(og);
                s += "        smem[" + std::to_string(og) + "][yl+" + std::to_string(p) + "u][tx] = NGU_LEAKY(" + qn + ");\n";
            }
        s += "        barrier();\n    }\n";

        // Stage 1: ColumnGroupRows (or Spatial if head !col_order) on 30 x th
        const char *pfx = tail_out ? "h" : "p";
        const bool need_post_bar = tail_out && !aa;
        if (!need_post_bar)
            s += "    if (yl >= " + std::to_string(th) + "u || tx >= 30u || uint(bx) + tx >= pc.width || uint(by) + yl >= pc.y1) return;\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g_last; ++og)
                s += "    vec4 " + std::string(pfx) + std::to_string(p) + ((g_last == 1) ? "" : ("_" + std::to_string(og))) + ";\n";
        if (need_post_bar)
            s += "    if (yl < " + std::to_string(th) + "u && tx < 30u) {\n";
        const char *ind = need_post_bar ? "        " : "    ";
        s += emit_bias_init_n(g_last, std::to_string(stages[1].b_uoff) + "u", pfx, py, ind, 20) +
             std::string(ind) + "int xl = max(0, 1 - bx), xh = min(31, int(pc.width) - bx);\n" +
             std::string(ind) + "int ylo = max(0, 1 - by), yhi = min(" + std::to_string(th + 1) + ", int(pc.height) - by);\n";
        for (int r = 0; r < py + 2; ++r)
            s += std::string(ind) + "uint ry" + std::to_string(r) + " = uint(clamp(int(yl) + " + std::to_string(r) + ", ylo, yhi));\n";
        for (int kx = 0; kx < 3; ++kx) {
            s += std::string(ind) + "{ uint rx = uint(clamp(int(tx) + " + std::to_string(kx) + ", xl, xh));\n";
            for (int ig = 0; ig < g_mid; ++ig) {
                s += std::string(ind) + "  {\n";
                for (int r = 0; r < py + 2; ++r)
                    s += std::string(ind) + "    vec4 t" + std::to_string(r) + " = smem[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx];\n";
                const std::uint32_t base = stages[1].w_uoff + static_cast<std::uint32_t>((kx * g_mid + ig) * (6 * g_last));
                for (int ky = 0; ky < 3; ++ky)
                    for (int f = 0; f < 4 * g_last; f += 2) {
                        const int c0 = f / g_last, og0 = f % g_last, c1 = (f + 1) / g_last, og1 = (f + 1) % g_last;
                        const std::string s0 = (g_last == 1) ? "" : ("_" + std::to_string(og0));
                        const std::string s1 = (g_last == 1) ? "" : ("_" + std::to_string(og1));
                        const std::string tg = "u1_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" +
                                               std::to_string(ky) + "_" + std::to_string(f / 2);
                        s += std::string(ind) + "    { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g_last + f / 2) +
                             "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                        for (int p = 0; p < py; ++p)
                            s += "NGU_MAC(" + std::string(pfx) + std::to_string(p) + s0 + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                        for (int p = 0; p < py; ++p)
                            s += "NGU_MAC(" + std::string(pfx) + std::to_string(p) + s1 + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                        s += "}\n";
                    }
                s += std::string(ind) + "  }\n";
            }
            s += std::string(ind) + "}\n";
        }
        if (need_post_bar)
            s += "    }\n";

        if (!tail_out) {
            s += "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n";
            for (int p = 0; p < py; ++p) {
                if (p > 0)
                    s += "    if (yl + " + std::to_string(p) + "u < " + std::to_string(th) + "u && y0 + " + std::to_string(p) + "u < pc.y1) {\n";
                s += "    { uint ob = (y0 + " + std::to_string(p) + "u - pc.dst_y0) * pc.width + x;\n";
                for (int og = 0; og < g_last; ++og) {
                    const std::string v = "p" + std::to_string(p) + "_" + std::to_string(og);
                    s += "      d[" + std::to_string(og) + "u * pc.plane_stride + ob] = NGU_LEAKY(" + v + ");\n";
                }
                s += "    }\n";
                if (p > 0)
                    s += "    }\n";
            }
        } else if (!aa) {
            s += "    barrier();\n"
                 "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                 "            float s0=src_t[sy][tx+0u], s1=src_t[sy][tx+1u], s2=src_t[sy][tx+2u], s3=src_t[sy][tx+3u], s4=src_t[sy][tx+4u];\n"
                 "            float bx0 = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                 "            float bx1 = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                 "            smem[0][sy][tx].xy = vec2(bx0, bx1);\n"
                 "        }\n    }\n    barrier();\n"
                 "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n"
                 "    if (yl >= " + std::to_string(th) + "u || tx >= 30u || x >= pc.width || y0 >= pc.y1) return;\n"
                 "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n";
            for (int r = 0; r < py + 4; ++r)
                s += "    vec2 r" + std::to_string(r) + " = smem[0][min(yl + " + std::to_string(r) +
                     "u, " + std::to_string(rows - 1) + "u)][tx].xy;\n";
            for (int p = 0; p < py; ++p) {
                const std::string hn = "h" + std::to_string(p);
                if (p > 0)
                    s += "    if (yl + " + std::to_string(p) + "u < " + std::to_string(th) + "u && y0 + " + std::to_string(p) + "u < pc.y1) {\n";
                s += "    { uint oy = (y0 + " + std::to_string(p) + "u) * 2u;\n"
                     "      vec2 top = ((((cd*r" + std::to_string(p) + ")+(cb*r" + std::to_string(p + 1) +
                     "))+(ca*r" + std::to_string(p + 2) + "))+(cc*r" + std::to_string(p + 3) + ")) + " + hn + ".xy;\n"
                     "      vec2 bot = ((((cc*r" + std::to_string(p + 1) + ")+(ca*r" + std::to_string(p + 2) +
                     "))+(cb*r" + std::to_string(p + 3) + "))+(cd*r" + std::to_string(p + 4) + ")) + " + hn + ".zw;\n"
                     "      d2[(oy + 0u) * pc.dst_s2 + x] = top;\n"
                     "      d2[(oy + 1u) * pc.dst_s2 + x] = bot;\n"
                     "    }\n";
                if (p > 0)
                    s += "    }\n";
            }
        } else {
            s += "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n";
            for (int r = 0; r <= py; ++r)
                s += "    uint sr" + std::to_string(r) + " = min(yl + " + std::to_string(r) + "u, " + std::to_string(th) + "u);\n"
                     "    NGU_RAW u" + std::to_string(r) + "_0 = src_u_t[sr" + std::to_string(r) + "][tx+0u];\n"
                     "    NGU_RAW u" + std::to_string(r) + "_1 = src_u_t[sr" + std::to_string(r) + "][tx+1u];\n"
                     "    float s" + std::to_string(r) + "_0 = NGU_FROM_RAW(u" + std::to_string(r) + "_0);\n"
                     "    float s" + std::to_string(r) + "_1 = NGU_FROM_RAW(u" + std::to_string(r) + "_1);\n";
            for (int p = 0; p < py; ++p) {
                const std::string hn = "h" + std::to_string(p);
                if (p > 0)
                    s += "    if (yl + " + std::to_string(p) + "u < " + std::to_string(th) + "u && y0 + " + std::to_string(p) + "u < pc.y1) {\n";
                s += "    { uint oy = (y0 + " + std::to_string(p) + "u) * 2u;\n"
                     "      float p10 = (s" + std::to_string(p) + "_0 + s" + std::to_string(p) + "_1) * 0.5 + " + hn + ".y;\n"
                     "      float p01 = (s" + std::to_string(p) + "_0 + s" + std::to_string(p + 1) + "_0) * 0.5 + " + hn + ".x;\n"
                     "      float p11 = (((s" + std::to_string(p) + "_0 + s" + std::to_string(p) + "_1) + s" +
                     std::to_string(p + 1) + "_0) + s" + std::to_string(p + 1) + "_1) * 0.25 + " + hn + ".z;\n"
                     "      d2[(oy + 0u) * pc.dst_s2 + x] = NGU_PACK_RAW2(u" + std::to_string(p) + "_0, p10);\n"
                     "      d2[(oy + 1u) * pc.dst_s2 + x] = NGU_PACK_F2(p01, p11);\n"
                     "    }\n";
                if (p > 0)
                    s += "    }\n";
            }
        }
        s += "}\n";
        return s;
    }

    // 5. Strip-Aware Standalone Head + Vectorized Combine + Coalesced Stores (32 x (wg_y * py))
    std::string gen_head_stripe(const StageDesc &head, std::uint32_t total_uvec4,
                                int wg_y, int py, bool aa, bool alias_bx = false) const {
        const int g = head.c_in / 4;
        const int th = wg_y * py;
        const int rows = th + 2;
        const int srows = th + 4;
        const char *comps[4] = {"x", "y", "z", "w"};

        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=" + std::to_string(wg_y) + ", local_size_z=1) in;\n"
             "layout(set=0, binding=0, std430) readonly buffer Features { vec4 feat[]; };\n";
        s += aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW2 d2[]; };\n"
                : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { vec2 d2[]; };\n";
        s += "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n";
        s += aa ? "layout(set=0, binding=3, std430) readonly buffer OrigSrc { NGU_RAW src_in[]; };\n"
                : "layout(set=0, binding=3, std430) readonly buffer OrigSrc { float src_in[]; };\n";
        s += fast_push_decl();
        s += "shared vec4 tile[" + std::to_string(g) + "][" + std::to_string(rows) + "][34];\n";
        if (aa)
            s += "shared NGU_RAW src_u_t[" + std::to_string(th + 1) + "][34];\n";
        else if (!alias_bx)
            s += "shared float src_t[" + std::to_string(srows) + "][36];\n"
                 "shared vec2 bx_t[" + std::to_string(srows) + "][34];\n";
        s += "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 32, by = int(pc.y0) + int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    {\n"
             "        uint gx = uint(clamp(bx + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "        for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "            uint cy = uint(clamp(by + int(sy) - 1, int(pc.src_y0), int(pc.src_y_max)));\n"
             "            uint row = (cy - pc.src_y0) * pc.width;\n"
             "            for (int gi = 0; gi < " + std::to_string(g) + "; ++gi) {\n"
             "                uint ps = uint(gi) * pc.plane_stride + row;\n"
             "                tile[gi][sy][tx] = feat[ps + gx];\n"
             "                if (tx < 2u) tile[gi][sy][32u + tx] = feat[ps + gx_r];\n"
             "            }\n        }\n    }\n";
        if (aa) {
            s += "    {\n"
                 "        uint gx = uint(clamp(bx + int(tx), 0, int(pc.width) - 1));\n"
                 "        uint gx_r = uint(clamp(bx + 32 + int(tx), 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < " + std::to_string(th + 1) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                 "            uint srow = uint(clamp(by + int(sy), 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            src_u_t[sy][tx] = src_in[srow + gx];\n"
                 "            if (tx < 1u) src_u_t[sy][32u + tx] = src_in[srow + gx_r];\n"
                 "        }\n    }\n";
        } else if (!alias_bx) {
            s += "    {\n"
                 "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < " + std::to_string(srows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                 "            uint srow = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            src_t[sy][tx] = src_in[srow + gx];\n"
                 "            if (tx < 4u) src_t[sy][32u + tx] = src_in[srow + gx_r];\n"
                 "        }\n    }\n";
        }
        s += "    barrier();\n";
        if (!aa && !alias_bx) {
            s += "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint sy = ty; sy < " + std::to_string(srows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                 "            float s0=src_t[sy][tx+0u], s1=src_t[sy][tx+1u], s2=src_t[sy][tx+2u], s3=src_t[sy][tx+3u], s4=src_t[sy][tx+4u];\n"
                 "            float bx0 = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                 "            float bx1 = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                 "            bx_t[sy][tx] = vec2(bx0, bx1);\n"
                 "        }\n    }\n    barrier();\n";
        }
        if (!alias_bx)
            s += "    if (uint(bx) + tx >= pc.width || uint(by) + yl >= pc.y1) return;\n";
        for (int p = 0; p < py; ++p)
            s += "    vec4 h" + std::to_string(p) + ";\n";
        s += emit_bias_init_n(1, std::to_string(head.b_uoff) + "u", "h", py, "    ", 4);
        if (head.col_order) {
            for (int kx = 0; kx < 3; ++kx)
                for (int ig = 0; ig < g; ++ig) {
                    s += "    {\n";
                    for (int r = 0; r < py + 2; ++r)
                        s += "        vec4 t" + std::to_string(r) + " = tile[" + std::to_string(ig) + "][yl+" +
                             std::to_string(r) + "u][tx+" + std::to_string(kx) + "u];\n";
                    const std::uint32_t base = head.w_uoff + static_cast<std::uint32_t>((kx * g + ig) * 6);
                    for (int ky = 0; ky < 3; ++ky)
                        for (int c = 0; c < 4; c += 2) {
                            const std::string tg = "hc_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" +
                                                   std::to_string(ky) + "_" + std::to_string(c);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 + c / 2) +
                                 "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(h" + std::to_string(p) + ", t" + std::to_string(ky + p) + "." + comps[c] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(h" + std::to_string(p) + ", t" + std::to_string(ky + p) + "." + comps[c + 1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                    s += "    }\n";
                }
        } else {
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    for (int ig = 0; ig < g; ++ig) {
                        s += "    {\n";
                        for (int p = 0; p < py; ++p)
                            s += "        vec4 t" + std::to_string(p) + " = tile[" + std::to_string(ig) + "][yl+" +
                                 std::to_string(ky + p) + "u][tx+" + std::to_string(kx) + "u];\n";
                        const std::uint32_t base = head.w_uoff + static_cast<std::uint32_t>(((ky * 3 + kx) * g + ig) * 2);
                        for (int c = 0; c < 4; c += 2) {
                            const std::string tg = "hs_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" +
                                                   std::to_string(ig) + "_" + std::to_string(c);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + c / 2) +
                                 "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(h" + std::to_string(p) + ", t" + std::to_string(p) + "." + comps[c] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(h" + std::to_string(p) + ", t" + std::to_string(p) + "." + comps[c + 1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                        s += "    }\n";
                    }
        }
        if (!aa && alias_bx) {
            // Reuse tile[0..1] for OrigSrc -> bx on strict 32-KB devices (requires rows >= srows/2 = (th+4)/2)
            s += "    barrier();\n"
                 "    {\n"
                 "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < " + std::to_string(srows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                 "            uint row = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            uint sl = sy >> 1u;\n"
                 "            if ((sy & 1u) == 0u) { tile[0][sl][tx].x = src_in[row + gx]; if (tx < 4u) tile[0][sl][32u + tx].x = src_in[row + gx_r]; }\n"
                 "            else                 { tile[0][sl][tx].y = src_in[row + gx]; if (tx < 4u) tile[0][sl][32u + tx].y = src_in[row + gx_r]; }\n"
                 "        }\n    }\n    barrier();\n"
                 "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint sy = ty; sy < " + std::to_string(srows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                 "            uint sl = sy >> 1u;\n"
                 "            float s0 = (sy&1u)==0u ? tile[0][sl][tx+0u].x : tile[0][sl][tx+0u].y;\n"
                 "            float s1 = (sy&1u)==0u ? tile[0][sl][tx+1u].x : tile[0][sl][tx+1u].y;\n"
                 "            float s2 = (sy&1u)==0u ? tile[0][sl][tx+2u].x : tile[0][sl][tx+2u].y;\n"
                 "            float s3 = (sy&1u)==0u ? tile[0][sl][tx+3u].x : tile[0][sl][tx+3u].y;\n"
                 "            float s4 = (sy&1u)==0u ? tile[0][sl][tx+4u].x : tile[0][sl][tx+4u].y;\n"
                 "            float bx0 = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                 "            float bx1 = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                 "            if ((sy&1u)==0u) tile[1][sl][tx].xy = vec2(bx0, bx1);\n"
                 "            else             tile[1][sl][tx].zw = vec2(bx0, bx1);\n"
                 "        }\n    }\n    barrier();\n"
                 "    if (uint(bx) + tx >= pc.width || uint(by) + yl >= pc.y1) return;\n";
        }
        s += "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n";
        if (!aa) {
            s += "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n";
            for (int r = 0; r < py + 4; ++r) {
                if (alias_bx)
                    s += "    vec2 r" + std::to_string(r) + " = (((yl+" + std::to_string(r) + "u)&1u)==0u) ? tile[1][(yl+" +
                         std::to_string(r) + "u)>>1u][tx].xy : tile[1][(yl+" + std::to_string(r) + "u)>>1u][tx].zw;\n";
                else
                    s += "    vec2 r" + std::to_string(r) + " = bx_t[yl+" + std::to_string(r) + "u][tx];\n";
            }
            for (int p = 0; p < py; ++p) {
                const std::string hn = "h" + std::to_string(p);
                if (p > 0)
                    s += "    if (y0 + " + std::to_string(p) + "u < pc.y1) {\n";
                s += "    { uint oy = (y0 + " + std::to_string(p) + "u) * 2u;\n"
                     "      vec2 top = ((((cd*r" + std::to_string(p) + ")+(cb*r" + std::to_string(p + 1) +
                     "))+(ca*r" + std::to_string(p + 2) + "))+(cc*r" + std::to_string(p + 3) + ")) + " + hn + ".xy;\n"
                     "      vec2 bot = ((((cc*r" + std::to_string(p + 1) + ")+(ca*r" + std::to_string(p + 2) +
                     "))+(cb*r" + std::to_string(p + 3) + "))+(cd*r" + std::to_string(p + 4) + ")) + " + hn + ".zw;\n"
                     "      d2[(oy + 0u) * pc.dst_s2 + x] = top;\n"
                     "      d2[(oy + 1u) * pc.dst_s2 + x] = bot;\n"
                     "    }\n";
                if (p > 0)
                    s += "    }\n";
            }
        } else {
            for (int r = 0; r <= py; ++r)
                s += "    NGU_RAW u" + std::to_string(r) + "_0 = src_u_t[yl+" + std::to_string(r) + "u][tx+0u];\n"
                     "    NGU_RAW u" + std::to_string(r) + "_1 = src_u_t[yl+" + std::to_string(r) + "u][tx+1u];\n"
                     "    float s" + std::to_string(r) + "_0 = NGU_FROM_RAW(u" + std::to_string(r) + "_0);\n"
                     "    float s" + std::to_string(r) + "_1 = NGU_FROM_RAW(u" + std::to_string(r) + "_1);\n";
            for (int p = 0; p < py; ++p) {
                const std::string hn = "h" + std::to_string(p);
                if (p > 0)
                    s += "    if (y0 + " + std::to_string(p) + "u < pc.y1) {\n";
                s += "    { uint oy = (y0 + " + std::to_string(p) + "u) * 2u;\n"
                     "      float p10 = (s" + std::to_string(p) + "_0 + s" + std::to_string(p) + "_1) * 0.5 + " + hn + ".y;\n"
                     "      float p01 = (s" + std::to_string(p) + "_0 + s" + std::to_string(p + 1) + "_0) * 0.5 + " + hn + ".x;\n"
                     "      float p11 = (((s" + std::to_string(p) + "_0 + s" + std::to_string(p) + "_1) + s" +
                     std::to_string(p + 1) + "_0) + s" + std::to_string(p + 1) + "_1) * 0.25 + " + hn + ".z;\n"
                     "      d2[(oy + 0u) * pc.dst_s2 + x] = NGU_PACK_RAW2(u" + std::to_string(p) + "_0, p10);\n"
                     "      d2[(oy + 1u) * pc.dst_s2 + x] = NGU_PACK_F2(p01, p11);\n"
                     "    }\n";
                if (p > 0)
                    s += "    }\n";
            }
        }
        s += "}\n";
        return s;
    }
};

enum class Topology { Fallback, Low, Medium, High, VeryHigh };

Topology classify_topology(const Model &model, VkPhysicalDeviceLimits limits, bool half) {
    const auto &b = model.body;
    const auto &h = model.head;
    const int expected_head_k = model.aa ? 3 : 4;
    if (h.kernel() != expected_head_k || h.output_channels() != (model.aa ? 4 : 1))
        return Topology::Fallback;
    const auto is_conv3 = [](const Layer &l, int ci, int co) {
        return l.kernel() == 3 && l.input_channels() == ci && l.output_channels() == co;
    };
    Topology topo = Topology::Fallback;
    VkDeviceSize min_smem = 0;
    VkDeviceSize max_ubo = 0;
    if (b.size() == 3 && is_conv3(b[0], 1, 8) && is_conv3(b[1], 8, 8) && is_conv3(b[2], 8, 8) &&
        h.input_channels() == 8) {
        topo = Topology::Low;
        min_smem = half ? 19856 : 29856;
        max_ubo = half ? 3104 : 6208;
    } else if (b.size() == 5 && is_conv3(b[0], 1, 12) && is_conv3(b[1], 12, 8) &&
               is_conv3(b[2], 8, 8) && is_conv3(b[3], 8, 8) && is_conv3(b[4], 8, 8) &&
               h.input_channels() == 8) {
        topo = Topology::Medium;
        min_smem = half ? 23232 : 28288;
        max_ubo = half ? 2368 : 4736;
    } else if (b.size() == 6 && is_conv3(b[0], 1, 12) && is_conv3(b[1], 12, 12) &&
               is_conv3(b[2], 12, 12) && is_conv3(b[3], 12, 12) && is_conv3(b[4], 12, 12) &&
               is_conv3(b[5], 12, 12) && h.input_channels() == 12) {
        topo = Topology::High;
        min_smem = half ? 23232 : 29504;
        max_ubo = half ? 3520 : 5312;
    } else if (b.size() == 8 && is_conv3(b[0], 1, 16) && is_conv3(b[1], 16, 16) &&
               is_conv3(b[2], 16, 16) && is_conv3(b[3], 16, 16) && is_conv3(b[4], 16, 16) &&
               is_conv3(b[5], 16, 16) && is_conv3(b[6], 16, 16) && is_conv3(b[7], 16, 16) &&
               h.input_channels() == 16) {
        topo = Topology::VeryHigh;
        min_smem = half ? 30304 : 30464;
        max_ubo = half ? 4992 : 9920;
    }
    if (topo == Topology::Fallback)
        return Topology::Fallback;
    if (limits.maxComputeSharedMemorySize < min_smem || limits.maxUniformBufferRange < max_ubo ||
        limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 32 ||
        limits.maxComputeWorkGroupSize[1] < 8)
        return Topology::Fallback;
    return topo;
}

} // namespace

struct Vulkan::Impl {
    VSCore *core = nullptr;
    const VSAPI *api = nullptr;
    const VSVULKANAPI *gpu = nullptr;
    const VSVulkanFunctions *vk = nullptr;
    VSVulkanCoreHandles handles{};
    VkPhysicalDeviceLimits limits{};
    VSGPUExecPool *pool = nullptr;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkDescriptorUpdateTemplate descriptor_template = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkPipeline> owned_pipelines;
    Buffer parameters{nullptr, BufferDeleter{nullptr}};
    VSVulkanBufferInfo parameter_info{};
    int width = 0, height = 0;
    bool aa = false;
    bool half_compute = false;
    VSVideoFormat format{};

    bool fast_path = false;
    std::vector<FastPass> fast_passes;
    std::array<VkDescriptorBufferInfo, 2> fast_slices{};
    std::uint32_t strip_height = 0;
    std::uint32_t slice_height = 0;
    std::uint32_t total_rem_layers = 0;
    VkDeviceSize workspace_size = 0;

    VkPipeline fallback_pipeline = VK_NULL_HANDLE;
    std::vector<LayerInfo> layers;
    std::array<VkDescriptorBufferInfo, 5> fallback_slices{};

    ~Impl() {
        if (gpu && pool) {
            gpu->freeGPUExecPool(pool);
            pool = nullptr;
        }
        if (vk && handles.device != VK_NULL_HANDLE) {
            for (VkPipeline p : owned_pipelines)
                if (p != VK_NULL_HANDLE)
                    vk->vkDestroyPipeline(handles.device, p, nullptr);
            if (descriptor_template != VK_NULL_HANDLE)
                vk->vkDestroyDescriptorUpdateTemplate(handles.device, descriptor_template, nullptr);
            if (layout != VK_NULL_HANDLE)
                vk->vkDestroyPipelineLayout(handles.device, layout, nullptr);
            if (descriptors != VK_NULL_HANDLE)
                vk->vkDestroyDescriptorSetLayout(handles.device, descriptors, nullptr);
        }
    }

    Buffer buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                  VkMemoryPropertyFlags preferred, VSVulkanBufferInfo &info) const {
        char error[1024]{};
        VSGPUBuffer *raw = gpu->createGPUBuffer(core, bytes, usage, required, preferred, &info,
                                                error, sizeof(error));
        if (!raw)
            throw std::runtime_error(std::string("GPU buffer: ") + error);
        return Buffer(raw, BufferDeleter{gpu});
    }

    template<typename T>
    void upload_device_buffer(const std::vector<T> &values, VkBuffer dst_buffer) const {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(values.size()) * sizeof(T);
        VSVulkanBufferInfo staging_info{};
        Buffer staging = buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                0, staging_info);
        if (!staging_info.mapped)
            throw std::runtime_error("staging buffer is not host-mapped");
        std::memcpy(staging_info.mapped, values.data(), bytes);
        char error[1024]{};
        Exec exec{gpu, gpu->gpuExecAcquire(pool, error, sizeof(error))};
        if (!exec.context)
            throw std::runtime_error(std::string("GPU upload acquire: ") + error);
        gpu->gpuExecUsesBuffer(exec.context, staging.release());
        const VkCommandBuffer cmd = gpu->gpuExecCommandBuffer(exec.context);
        const VkBufferCopy2 region{VK_STRUCTURE_TYPE_BUFFER_COPY_2, nullptr, 0, 0, bytes};
        const VkCopyBufferInfo2 copy{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2, nullptr,
                                     staging_info.buffer, dst_buffer, 1, &region};
        vk->vkCmdCopyBuffer2(cmd, &copy);
        exec.submit();
        if (gpu->gpuExecPoolWaitIdle(pool, error, sizeof(error)) != gdDrained)
            throw std::runtime_error(std::string("GPU upload drain: ") + error);
    }

    VkPipeline compile_pipeline(const char *source, VkPipelineCache cache = VK_NULL_HANDLE) {
        const std::string specialized = half_compute ? half_shader(source) : source;
        char log[8192]{};
        VSGPUShader *shader = gpu->compileGPUShader(core, slGLSL, specialized.c_str(), log, sizeof(log));
        if (!shader)
            throw std::runtime_error(std::string("shader compile: ") + log);
        std::size_t code_size = 0;
        const std::uint32_t *code = gpu->getGPUShaderCode(shader, &code_size);
        try {
            require_spirv_precision({code, code_size / sizeof(*code)}, half_compute ? 16 : 32);
#ifdef NGU_DUMP_SHADERS
            if (const char *directory = std::getenv("NGU_SHADER_DUMP_DIR")) {
                std::filesystem::create_directories(directory);
                std::uint64_t hash = 14695981039346656037ull;
                for (std::size_t i=0; i<code_size/sizeof(*code); ++i)
                    hash = (hash ^ code[i]) * 1099511628211ull;
                const auto stem = std::filesystem::path(directory) /
                    ((half_compute ? "fp16-" : "fp32-") + std::to_string(hash));
                std::ofstream binary(stem.string()+".spv",std::ios::binary);
                binary.exceptions(std::ios::badbit | std::ios::failbit);
                binary.write(reinterpret_cast<const char *>(code),static_cast<std::streamsize>(code_size));
                std::ofstream glsl(stem.string()+".glsl");
                glsl.exceptions(std::ios::badbit | std::ios::failbit);
                glsl << specialized;
            }
#endif
        } catch (...) {
            gpu->freeGPUShader(shader);
            throw;
        }
        VkShaderModuleCreateInfo mi{};
        mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mi.codeSize = code_size;
        mi.pCode = code;
        VkShaderModule module = VK_NULL_HANDLE;
        const VkResult module_result = vk->vkCreateShaderModule(handles.device, &mi, nullptr, &module);
        gpu->freeGPUShader(shader);
        vk_check(module_result, "shader module");
        VkComputePipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pi.stage.module = module;
        pi.stage.pName = "main";
        pi.layout = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult result =
            vk->vkCreateComputePipelines(handles.device, cache, 1, &pi, nullptr, &pipe);
        vk->vkDestroyShaderModule(handles.device, module, nullptr);
        vk_check(result, "compute pipeline");
        owned_pipelines.push_back(pipe);
        return pipe;
    }

    VkDeviceSize add_slice(int w, int h, int c, VkDeviceSize &cursor) const {
        const std::size_t count = checked_elements(w, h, c);
        if (count > std::numeric_limits<VkDeviceSize>::max() / 4)
            throw std::invalid_argument("Vulkan workspace slice overflow");
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * (half_compute ? 2 : 4);
        if (bytes > limits.maxStorageBufferRange)
            throw std::invalid_argument("Vulkan workspace slice exceeds storage buffer range");
        const VkDeviceSize align = std::max<VkDeviceSize>(16, limits.minStorageBufferOffsetAlignment);
        const VkDeviceSize offset = align_up(cursor, align);
        if (bytes > std::numeric_limits<VkDeviceSize>::max() - offset)
            throw std::invalid_argument("Vulkan workspace size overflow");
        cursor = offset + bytes;
        return offset;
    }

    void setup_descriptors_and_layout(bool use_ubo, std::uint32_t push_size) {
        const VkDescriptorType param_type =
            use_ubo ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        const std::array<VkDescriptorSetLayoutBinding, 4> bindings{{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, param_type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        }};
        VkDescriptorSetLayoutCreateInfo dsl{};
        dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
        dsl.bindingCount = static_cast<std::uint32_t>(bindings.size());
        dsl.pBindings = bindings.data();
        vk_check(vk->vkCreateDescriptorSetLayout(handles.device, &dsl, nullptr, &descriptors),
                 "descriptor set layout");
        const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &descriptors;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &push_range;
        vk_check(vk->vkCreatePipelineLayout(handles.device, &pl, nullptr, &layout), "pipeline layout");

        const std::array<VkDescriptorUpdateTemplateEntry, 4> entries{{
            {0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0 * sizeof(VkDescriptorBufferInfo), sizeof(VkDescriptorBufferInfo)},
            {1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 * sizeof(VkDescriptorBufferInfo), sizeof(VkDescriptorBufferInfo)},
            {2, 0, 1, param_type,                        2 * sizeof(VkDescriptorBufferInfo), sizeof(VkDescriptorBufferInfo)},
            {3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * sizeof(VkDescriptorBufferInfo), sizeof(VkDescriptorBufferInfo)},
        }};
        VkDescriptorUpdateTemplateCreateInfo ti{};
        ti.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
        ti.descriptorUpdateEntryCount = static_cast<std::uint32_t>(entries.size());
        ti.pDescriptorUpdateEntries = entries.data();
        ti.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS;
        ti.descriptorSetLayout = descriptors;
        ti.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
        ti.pipelineLayout = layout;
        ti.set = 0;
        vk_check(vk->vkCreateDescriptorUpdateTemplate(handles.device, &ti, nullptr, &descriptor_template),
                 "descriptor update template");
    }

    void init_fast(const Model &model, Topology topo) {
        fast_path = true;
        setup_descriptors_and_layout(true, sizeof(FastPush));
        const ShaderGenerator shader{half_compute};
        VkPipelineCacheCreateInfo pcci{};
        pcci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        VkPipelineCache pcache = VK_NULL_HANDLE;
        vk->vkCreatePipelineCache(handles.device, &pcci, nullptr, &pcache);
        struct CacheGuard {
            const VSVulkanFunctions *vk;
            VkDevice dev;
            VkPipelineCache cache;
            ~CacheGuard() {
                if (cache != VK_NULL_HANDLE)
                    vk->vkDestroyPipelineCache(dev, cache, nullptr);
            }
        } cache_guard{vk, handles.device, pcache};

        std::vector<float> ubo_data;
        const VkDeviceSize ubo_align = std::max<VkDeviceSize>(16, limits.minUniformBufferOffsetAlignment);
        const VkDeviceSize elem_bytes = half_compute ? 2 : 4;
        const bool can_512 = limits.maxComputeWorkGroupInvocations >= 512 &&
                             limits.maxComputeWorkGroupSize[1] >= 16;

        const auto append_ubo = [&](std::vector<float> &blk, VkDeviceSize &off, VkDeviceSize &sz) {
            align_uvec4_floats(blk);
            const VkDeviceSize cur_bytes = static_cast<VkDeviceSize>(ubo_data.size()) * elem_bytes;
            off = align_up(cur_bytes, ubo_align);
            if (off > cur_bytes)
                ubo_data.resize(static_cast<std::size_t>(off / elem_bytes), 0.0f);
            sz = static_cast<VkDeviceSize>(blk.size()) * elem_bytes;
            if (sz > limits.maxUniformBufferRange)
                throw std::runtime_error("fused parameters exceed Vulkan buffer range");
            ubo_data.insert(ubo_data.end(), blk.begin(), blk.end());
            return static_cast<std::uint32_t>(blk.size() / 8);
        };

        const auto make_body = [&](std::vector<float> &blk, std::size_t idx, bool col_order) -> StageDesc {
            const Layer &l = model.body[idx];
            const auto w_off = (l.input_channels() == 1)
                ? pack_1toC_aligned(blk, l)
                : pack_nc4_ordered(blk, l, col_order);
            const auto b_off = pack_bias_aligned(blk, l);
            return StageDesc{l.input_channels(), l.output_channels(), col_order, false, w_off, b_off};
        };
        const auto make_head = [&](std::vector<float> &blk, bool col_order) -> StageDesc {
            const Layer &h = model.head;
            const auto w_off = aa ? pack_aa_head_ordered(blk, h, col_order)
                                  : pack_k4_head_ordered(blk, h, col_order);
            const auto b_off = aa ? pack_bias_aligned(blk, h) : pack_k4_bias_aligned(blk, h);
            return StageDesc{h.input_channels(), 4, col_order, true, w_off, b_off};
        };

        int slice_channels = 0;
        VkDeviceSize target_ws_bytes = 0;
        std::uint32_t strip_align_h = 16;

        if (topo == Topology::Low) {
            // FP16 preserves exact Spatial traversal for bit-exact half_reference match;
            // FP32 uses ColumnGroupRows on b1/b2/head for 3x fewer shared loads.
            const bool col_mid = !half_compute;
            std::vector<float> blk;
            std::array<StageDesc, 4> st{{
                make_body(blk, 0, false),
                make_body(blk, 1, col_mid),
                make_body(blk, 2, col_mid),
                make_head(blk, col_mid),
            }};
            VkDeviceSize off = 0, sz = 0;
            const auto nu = append_ubo(blk, off, sz);
            int wg_y = 8, py = 4, px = 1;
            if (half_compute) {
                if (can_512 && limits.maxComputeSharedMemorySize >= 38416) {
                    wg_y = 16; py = 2; px = 2; // 58x26
                } else {
                    wg_y = 8; py = 4; px = 1;  // 26x26 (19.9 KB smem, 256 threads)
                }
            } else {
                if (can_512 && limits.maxComputeSharedMemorySize >= 39712) {
                    wg_y = 16; py = 2; px = 1; // 26x26
                } else {
                    wg_y = 8; py = 3; px = 1;  // 26x18 (29.9 KB smem < 32 KB, 256 threads)
                }
            }
            const auto tw = static_cast<std::uint32_t>(32 * px - 6);
            const auto th = static_cast<std::uint32_t>(wg_y * py - 6);
            VkPipeline pipe = compile_pipeline(shader.gen_low_stripe(st, nu, wg_y, py, px, aa).c_str(), pcache);
            fast_passes.push_back(FastPass{pipe, off, sz, tw, th, 4, -1, -2});
        } else if (topo == Topology::Medium) {
            // 3-Pass Pure Warp-Stripe: Zero-Waste (b0+b1), 2L (b2+b3), 2L (b4+head).
            // FP16 keeps exact Spatial->Col pairs; FP32 uses Col->Col on passes 1 & 2.
            slice_channels = 8;
            target_ws_bytes = 32ull << 20; // <= 32 MB L2-resident ping-pong slices (256 MB total VRAM!)
            strip_align_h = half_compute ? 30u : 22u;
            const bool col_s0 = !half_compute;
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 0, false), make_body(blk, 1, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const int wg_y = half_compute ? 8 : 6;
                const int py = half_compute ? 3 : 2;
                const auto th = static_cast<std::uint32_t>(wg_y * py);
                VkPipeline pipe = compile_pipeline(shader.gen_zw_b0_b1(st, nu, wg_y, py).c_str(), pcache);
                fast_passes.push_back(FastPass{pipe, off, sz, 32, th, 2, -1, 0});
            }
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 2, col_s0), make_body(blk, 3, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const int wg_y = 8;
                const int py = half_compute ? 4 : 3;
                const auto th = static_cast<std::uint32_t>(wg_y * py - 2);
                VkPipeline pipe = compile_pipeline(shader.gen_2L_stripe(st, nu, aa, wg_y, py).c_str(), pcache);
                fast_passes.push_back(FastPass{pipe, off, sz, 30, th, 2, 0, 1});
            }
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 4, col_s0), make_head(blk, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                int wg_y = 8, py = 3;
                if (half_compute) {
                    wg_y = can_512 ? 16 : 8;
                    py = can_512 ? 2 : 4;
                } else {
                    wg_y = 8;
                    py = (aa || limits.maxComputeSharedMemorySize >= 32032) ? 3 : 2;
                }
                const auto th = static_cast<std::uint32_t>(wg_y * py - 2);
                VkPipeline pipe = compile_pipeline(shader.gen_2L_stripe(st, nu, aa, wg_y, py).c_str(), pcache);
                fast_passes.push_back(FastPass{pipe, off, sz, 30, th, 2, 1, -2});
            }
        } else if (topo == Topology::High) {
            // FP16: ZW (b0+b1) + 3x 1L (b2..b4, 8w3py) + Fused Tail (b5 Spatial + head Col, 16w2py).
            // FP32: ZW (b0+b1) + 4x 1L (b2..b5, ALL Col, 6w2py) + Vectorized Head (6w2py) with L2-resident strips!
            slice_channels = 12;
            target_ws_bytes = half_compute ? (33ull << 20) : (32ull << 20); // L2-resident (<= 256 MB total VRAM!)
            strip_align_h = half_compute ? 24u : 12u;
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 0, false), make_body(blk, 1, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const int wg_y = half_compute ? 8 : 6;
                const int py = half_compute ? 3 : 2;
                const auto th = static_cast<std::uint32_t>(wg_y * py);
                VkPipeline pipe = compile_pipeline(shader.gen_zw_b0_b1(st, nu, wg_y, py).c_str(), pcache);
                fast_passes.push_back(FastPass{pipe, off, sz, 32, th, 2, -1, 0});
            }
            int cur_slice = 0;
            if (half_compute) {
                VkPipeline p_1l = VK_NULL_HANDLE;
                for (std::size_t i = 2; i <= 4; ++i) {
                    std::vector<float> blk;
                    StageDesc st = make_body(blk, i, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    if (p_1l == VK_NULL_HANDLE)
                        p_1l = compile_pipeline(shader.gen_1L_stripe(st, nu, 8, 3, false).c_str(), pcache);
                    const int next_slice = cur_slice ^ 1;
                    fast_passes.push_back(FastPass{p_1l, off, sz, 32, 24, 1, cur_slice, next_slice});
                    cur_slice = next_slice;
                }
                {
                    std::vector<float> blk;
                    std::array<StageDesc, 2> st{{make_body(blk, 5, false), make_head(blk, true)}};
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    const bool use_16w = can_512 && limits.maxComputeSharedMemorySize >= 30192;
                    const int wg_y = use_16w ? 16 : 8;
                    const int py = use_16w ? 2 : 3;
                    const auto th = static_cast<std::uint32_t>(wg_y * py - 2);
                    VkPipeline pipe = compile_pipeline(shader.gen_2L_stripe(st, nu, aa, wg_y, py).c_str(), pcache);
                    fast_passes.push_back(FastPass{pipe, off, sz, 30, th, 2, cur_slice, -2});
                }
            } else {
                VkPipeline p_1l = VK_NULL_HANDLE;
                for (std::size_t i = 2; i <= 5; ++i) {
                    std::vector<float> blk;
                    StageDesc st = make_body(blk, i, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    if (p_1l == VK_NULL_HANDLE)
                        p_1l = compile_pipeline(shader.gen_1L_stripe(st, nu, 6, 2, false).c_str(), pcache);
                    const int next_slice = cur_slice ^ 1;
                    fast_passes.push_back(FastPass{p_1l, off, sz, 32, 12, 1, cur_slice, next_slice});
                    cur_slice = next_slice;
                }
                {
                    std::vector<float> blk;
                    StageDesc st = make_head(blk, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    VkPipeline pipe = compile_pipeline(shader.gen_head_stripe(st, nu, 6, 2, aa, false).c_str(), pcache);
                    fast_passes.push_back(FastPass{pipe, off, sz, 32, 12, 1, cur_slice, -2});
                }
            }
        } else {
            // VeryHigh (C=16): ZW (b0+b1) + 6x 1L (b2..b7) + Vectorized Head.
            // - FP16: 8w4py body + 3-strip (~44 MB: -128 MB VRAM & -14 W / -20% power at 60 fps real video!)
            // - FP32: 6w2py body + Col Head + L2-resident 9-strip (~31.4 MB: -253 MB VRAM & zero DRAM thrash!)
            slice_channels = 16;
            target_ws_bytes = half_compute ? (48ull << 20) : (32ull << 20);
            strip_align_h = half_compute ? 32u : 12u;
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 0, false), make_body(blk, 1, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                int wg_y = 8, py = 3;
                if (!half_compute) {
                    if (limits.maxComputeSharedMemorySize >= 42048) {
                        wg_y = 8; py = 2; // 32x16
                    } else {
                        wg_y = 5; py = 2; // 32x10 (28.1 KB smem < 32 KB)
                    }
                }
                const auto th = static_cast<std::uint32_t>(wg_y * py);
                VkPipeline pipe = compile_pipeline(shader.gen_zw_b0_b1(st, nu, wg_y, py).c_str(), pcache);
                fast_passes.push_back(FastPass{pipe, off, sz, 32, th, 2, -1, 0});
            }
            int cur_slice = 0;
            {
                const int wg_y = half_compute ? 8 : 6;
                const int py = (half_compute && limits.maxComputeSharedMemorySize >= 36992) ? 4 : 2;
                const auto th = static_cast<std::uint32_t>(wg_y * py);
                VkPipeline p_1l = VK_NULL_HANDLE;
                for (std::size_t i = 2; i < model.body.size(); ++i) {
                    std::vector<float> blk;
                    StageDesc st = make_body(blk, i, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    if (p_1l == VK_NULL_HANDLE)
                        p_1l = compile_pipeline(shader.gen_1L_stripe(st, nu, wg_y, py, false).c_str(), pcache);
                    const int next_slice = cur_slice ^ 1;
                    fast_passes.push_back(FastPass{p_1l, off, sz, 32, th, 1, cur_slice, next_slice});
                    cur_slice = next_slice;
                }
            }
            {
                std::vector<float> blk;
                const bool head_col = !half_compute || aa;
                StageDesc st = make_head(blk, head_col);
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                int wg_y = 8, py = 2;
                bool alias_bx = false;
                if (!half_compute) {
                    const VkDeviceSize need = aa ? 41480 : 47488;
                    if (limits.maxComputeSharedMemorySize < need) {
                        if (aa) {
                            wg_y = 4; py = 2; // 23.0 KB smem
                        } else {
                            wg_y = 6; py = 2; alias_bx = true; // 30.5 KB smem
                        }
                    }
                }
                const auto th = static_cast<std::uint32_t>(wg_y * py);
                VkPipeline pipe = compile_pipeline(shader.gen_head_stripe(st, nu, wg_y, py, aa, alias_bx).c_str(), pcache);
                fast_passes.push_back(FastPass{pipe, off, sz, 32, th, 1, cur_slice, -2});
            }
        }

        strip_height = static_cast<std::uint32_t>(height);
        slice_height = static_cast<std::uint32_t>(height);
        total_rem_layers = 0;
        for (std::size_t i = 1; i < fast_passes.size(); ++i)
            total_rem_layers += fast_passes[i].layers;

        if (slice_channels > 0) {
            const auto h_u = static_cast<std::uint32_t>(height);
            const VkDeviceSize row_bytes_2slices =
                static_cast<VkDeviceSize>(width) * slice_channels * elem_bytes * 2;
            const VkDeviceSize full_ws_bytes = row_bytes_2slices * h_u;
            if (target_ws_bytes > 0 && full_ws_bytes > target_ws_bytes && row_bytes_2slices > 0) {
                const auto eval_max_sh = [&](std::uint32_t sh) {
                    std::uint32_t m = 0;
                    for (std::uint32_t sy = 0; sy < h_u; sy += sh) {
                        const std::uint32_t ey = std::min(h_u, sy + sh);
                        const std::uint32_t y0 = (sy > total_rem_layers) ? (sy - total_rem_layers) : 0u;
                        const std::uint32_t y1 = std::min(h_u, ey + total_rem_layers);
                        m = std::max(m, y1 - y0);
                    }
                    return m;
                };
                for (std::uint32_t n = 2; n <= h_u; ++n) {
                    const std::uint32_t bal = (h_u + n - 1) / n;
                    const std::uint32_t aln = ((bal + strip_align_h - 1) / strip_align_h) * strip_align_h;
                    const std::uint32_t m_aln = eval_max_sh(aln);
                    if (aln < h_u && static_cast<VkDeviceSize>(m_aln) * row_bytes_2slices <= target_ws_bytes) {
                        strip_height = aln;
                        slice_height = std::min(h_u, m_aln);
                        break;
                    }
                    const std::uint32_t m_bal = eval_max_sh(bal);
                    if (bal < h_u && static_cast<VkDeviceSize>(m_bal) * row_bytes_2slices <= target_ws_bytes) {
                        strip_height = bal;
                        slice_height = std::min(h_u, m_bal);
                        break;
                    }
                }
            }

            const auto g_count = static_cast<std::uint32_t>(slice_channels / 4);
            if (static_cast<std::uint64_t>(width) * slice_height >
                std::numeric_limits<std::uint32_t>::max() / g_count)
                throw std::invalid_argument("Vulkan feature plane stride overflow");
            int max_slice_idx = -1;
            for (const auto &p : fast_passes)
                max_slice_idx = std::max(max_slice_idx, std::max(p.src_slice, p.dst_slice));
            VkDeviceSize cursor = 0;
            const VkDeviceSize slice_bytes =
                static_cast<VkDeviceSize>(width) * slice_height * slice_channels * elem_bytes;
            if (max_slice_idx >= 0) {
                const VkDeviceSize s0 = add_slice(width, static_cast<int>(slice_height), slice_channels, cursor);
                fast_slices[0] = {VK_NULL_HANDLE, s0, slice_bytes};
            }
            if (max_slice_idx >= 1) {
                const VkDeviceSize s1 = add_slice(width, static_cast<int>(slice_height), slice_channels, cursor);
                fast_slices[1] = {VK_NULL_HANDLE, s1, slice_bytes};
            }
            workspace_size = cursor;
        }

        upload_parameters(ubo_data, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    void upload_parameters(const std::vector<float> &values, VkBufferUsageFlags usage) {
        const VkDeviceSize bytes = align_up(
            static_cast<VkDeviceSize>(values.size()) * (half_compute ? 2 : 4), 4);
        const auto max_range = (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ?
            limits.maxUniformBufferRange : limits.maxStorageBufferRange;
        if (!fast_path && bytes > max_range)
            throw std::runtime_error("parameters exceed Vulkan buffer range");
        parameters = buffer(bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, parameter_info);
        if (half_compute && (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
            std::vector<std::uint32_t> packed(static_cast<std::size_t>(bytes / 4), 0);
            for (std::size_t i = 0; i < values.size(); ++i)
                packed[i >> 1] |= std::uint32_t(float_to_half(values[i])) << ((i & 1) * 16);
            upload_device_buffer(packed, parameter_info.buffer);
        } else if (half_compute) {
            std::vector<std::uint16_t> halves(static_cast<std::size_t>(bytes / 2), 0);
            for (std::size_t i = 0; i < values.size(); ++i)
                halves[i] = float_to_half(values[i]);
            upload_device_buffer(halves, parameter_info.buffer);
        } else {
            upload_device_buffer(values, parameter_info.buffer);
        }
    }

    void init_fallback(const Model &model) {
        fast_path = false;
        setup_descriptors_and_layout(false, sizeof(FallbackPush));
        int max_channels = 4;
        std::vector<float> values;
        const auto add_layer = [&](const Layer &layer) {
            if (layer.output_channels() % 4 != 0 && layer.kernel() != 4)
                throw std::runtime_error("invalid GPU correlation channel count");
            const std::size_t offset = values.size();
            if (layer.kernel() == 3) {
                if (layer.precision() == WeightPrecision::FP16) {
                    for (auto h : layer.half_interleaved())
                        values.push_back(half_to_float(h));
                } else {
                    append_floats(values, layer.interleaved());
                }
            } else {
                for (std::size_t i = 0; i < static_cast<std::size_t>(
                         layer.input_channels() * layer.output_channels() * 16); ++i)
                    values.push_back(layer.weight_at(i));
            }
            const std::size_t count = values.size() - offset;
            append_floats(values, pack_bias(layer));
            layers.push_back(LayerInfo{static_cast<std::uint32_t>(layer.input_channels()),
                static_cast<std::uint32_t>(layer.output_channels()),
                static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(offset + count)});
            max_channels = std::max(max_channels, layer.output_channels());
        };
        for (const auto &layer : model.body)
            add_layer(layer);
        add_layer(model.head);

        upload_parameters(values, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        fallback_pipeline = compile_pipeline(fallback_shader_source);

        VkDeviceSize cursor = 0;
        const auto reserve_slice = [&](int sw, int sh, int sc) -> VkDescriptorBufferInfo {
            const VkDeviceSize off = add_slice(sw, sh, sc, cursor);
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(checked_elements(sw, sh, sc)) * (half_compute ? 2 : 4);
            return {VK_NULL_HANDLE, off, bytes};
        };
        fallback_slices[0] = reserve_slice(width, height, max_channels);
        fallback_slices[1] = reserve_slice(width, height, max_channels);
        if (aa) {
            fallback_slices[2] = reserve_slice(width, height, 4);
        } else {
            fallback_slices[2] = reserve_slice(width + 2, height + 2, 4);
            fallback_slices[3] = reserve_slice(width * 2, height, 1);
            fallback_slices[4] = reserve_slice(width * 2, height * 2, 1);
        }
        workspace_size = cursor;
    }

    void init(const Model &model, int w, int h, VSCore *c, const VSAPI *a, int bits) {
        core = c;
        api = a;
        width = w;
        height = h;
        aa = model.aa;
        half_compute = bits == 16;
        if (bits != 16 && bits != 32)
            throw std::invalid_argument("Vulkan plane must be float16 or float32");
        validate_2x_dimensions(w, h);
        gpu = api->getVulkanAPI();
        char error[4096]{};
        if (gpu->getVulkanHandles(core, &handles, error, sizeof(error)))
            throw std::runtime_error(std::string("Vulkan device: ") + error);
        vk = gpu->getVulkanFunctions(core, error, sizeof(error));
        if (!vk)
            throw std::runtime_error(std::string("Vulkan functions: ") + error);
        VkPhysicalDeviceFloatControlsProperties floats{};
        floats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &floats;
        vk->vkGetPhysicalDeviceProperties2(handles.physicalDevice, &properties);
        if (half_compute) {
            VkPhysicalDeviceVulkan11Features f11{};
            f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
            VkPhysicalDeviceVulkan12Features f12{};
            f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            f11.pNext = &f12;
            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features.pNext = &f11;
            vk->vkGetPhysicalDeviceFeatures2(handles.physicalDevice, &features);
            require_half_capabilities({f12.shaderFloat16 != 0, f11.storageBuffer16BitAccess != 0,
                floats.shaderSignedZeroInfNanPreserveFloat16 != 0,
                floats.shaderRoundingModeRTEFloat16 != 0, floats.shaderDenormPreserveFloat16 != 0});
        } else if (!floats.shaderSignedZeroInfNanPreserveFloat32 || !floats.shaderRoundingModeRTEFloat32)
            throw std::runtime_error(
                "Vulkan device does not support FP32 signed-zero/NaN/Inf and RTE controls");
        limits = properties.properties.limits;
        if (limits.maxComputeWorkGroupInvocations < 64 ||
            limits.maxComputeWorkGroupSize[0] < 8 || limits.maxComputeWorkGroupSize[1] < 8)
            throw std::runtime_error("Vulkan device lacks basic 8x8 compute workgroup support");
        if ((static_cast<std::uint32_t>(w) * 2 + 7) / 8 > limits.maxComputeWorkGroupCount[0] ||
            (static_cast<std::uint32_t>(h) * 2 + 7) / 8 > limits.maxComputeWorkGroupCount[1] ||
            limits.maxComputeWorkGroupCount[2] < 4)
            throw std::invalid_argument("Vulkan dimensions exceed dispatch limits");
        if (!api->queryVideoFormat(&format, cfGray, stFloat, bits, 0, 0, core))
            throw std::runtime_error("cannot query GPU GRAYS/GRAYH format");

        pool = gpu->createGPUExecPool(core, vqCompute, error, sizeof(error));
        if (!pool)
            throw std::runtime_error(std::string("GPU exec pool: ") + error);

        const Topology topo = classify_topology(model, limits, half_compute);
        if (topo != Topology::Fallback)
            init_fast(model, topo);
        else
            init_fallback(model);
    }

    void barrier(VkCommandBuffer cmd, VkAccessFlags2 src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) const {
        VkMemoryBarrier2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.srcAccessMask = src_access;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.dstAccessMask = src_access
            ? (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
            : 0;
        VkDependencyInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        info.memoryBarrierCount = 1;
        info.pMemoryBarriers = &memory;
        vk->vkCmdPipelineBarrier2(cmd, &info);
    }

    VkDescriptorBufferInfo plane(const VSFrame *frame, int w, int h, std::uint32_t &stride) const {
        VSVulkanPlaneInfo info{};
        if (api->getFrameResidency(frame) != nrGPU || gpu->getGPUPlane(frame, 0, &info) || !info.buffer)
            throw std::invalid_argument("Vulkan backend requires GPU resident planes");
        const std::ptrdiff_t pitch = api->getStride(frame, 0);
        const auto element_bytes = static_cast<unsigned>(format.bytesPerSample);
        if (pitch <= 0 || pitch % element_bytes ||
            static_cast<std::uint64_t>(pitch) <
                static_cast<VkDeviceSize>(w) * element_bytes ||
            static_cast<std::uint64_t>(pitch) / element_bytes >
                std::numeric_limits<std::uint32_t>::max() / static_cast<std::uint32_t>(h))
            throw std::invalid_argument("invalid Vulkan plane stride/index range");
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(pitch) * h;
        if (bytes > info.bufferSize || bytes > limits.maxStorageBufferRange)
            throw std::invalid_argument("Vulkan plane exceeds storage buffer range");
        stride = static_cast<std::uint32_t>(pitch / element_bytes);
        return {info.buffer, 0, bytes};
    }

    void push_bindings(VkCommandBuffer cmd, const VkDescriptorBufferInfo &src,
                       const VkDescriptorBufferInfo &dst, const VkDescriptorBufferInfo &params,
                       const VkDescriptorBufferInfo &aux) const {
        const std::array<VkDescriptorBufferInfo, 4> buffers{src, dst, params, aux};
        vk->vkCmdPushDescriptorSetWithTemplate(cmd, descriptor_template, layout, 0, buffers.data());
    }

    template <typename PushT>
    void push_constants(VkCommandBuffer cmd, const PushT &push) const {
        VkPushConstantsInfo constants{};
        constants.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO;
        constants.layout = layout;
        constants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        constants.size = sizeof(push);
        constants.pValues = &push;
        vk->vkCmdPushConstants2(cmd, &constants);
    }

    void process(const VSFrame *source, VSFrame *output) const {
        std::uint32_t source_stride = 0, output_stride = 0;
        auto input = plane(source, width, height, source_stride);
        auto destination = plane(output, width * 2, height * 2, output_stride);
        if (fast_path && (output_stride & 1u) != 0)
            throw std::invalid_argument("Vulkan 2x output plane stride must be even for coalesced vec2 stores");

        // Acquire exec context first so completed retentions from prior frames are reaped
        // before allocating per-frame pooled scratch (zero pinned idle VRAM, zero cross-frame
        // serialization barrier, cooperative reclamation with concurrent CUDA/Vulkan filters).
        char error[1024]{};
        Exec exec{gpu, gpu->gpuExecAcquire(pool, error, sizeof(error))};
        if (!exec.context)
            throw std::runtime_error(std::string("GPU frame acquire: ") + error);
        VSVulkanBufferInfo ws_info{};
        if (workspace_size > 0) {
            Buffer ws_buf = buffer(workspace_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, ws_info);
            gpu->gpuExecUsesBuffer(exec.context, ws_buf.release());
        }
        gpu->gpuExecReadsFrame(exec.context, source);
        gpu->gpuExecWritesPlane(exec.context, output, 0);
        const VkCommandBuffer cmd = gpu->gpuExecCommandBuffer(exec.context);

        const auto w = static_cast<std::uint32_t>(width);
        const auto h = static_cast<std::uint32_t>(height);

        if (fast_path) {
            auto local = fast_slices;
            for (auto &slice : local)
                slice.buffer = ws_info.buffer;
            const std::uint32_t plane_stride = w * slice_height;
            const std::uint32_t dst_s2 = output_stride >> 1u;
            const std::uint32_t eff_sh = strip_height ? strip_height : h;
            VkPipeline bound = VK_NULL_HANDLE;

            for (std::uint32_t sy = 0; sy < h; sy += eff_sh) {
                const std::uint32_t ey = std::min(h, sy + eff_sh);
                std::uint32_t rem = total_rem_layers;
                std::uint32_t prev_y0 = 0;
                std::uint32_t prev_y1 = h;
                for (std::size_t i = 0; i < fast_passes.size(); ++i) {
                    const FastPass &pass = fast_passes[i];
                    if (i > 0)
                        rem -= pass.layers;
                    const std::uint32_t out_y0 = (eff_sh < h && sy > rem) ? (sy - rem) : 0u;
                    const std::uint32_t out_y1 = (eff_sh < h) ? std::min(h, ey + rem) : h;
                    const std::uint32_t src_y0 = (pass.src_slice < 0) ? 0u : prev_y0;
                    const std::uint32_t dst_y0 = (pass.dst_slice < 0) ? 0u : out_y0;
                    const std::uint32_t src_y_max = (pass.src_slice < 0)
                        ? (h - 1u)
                        : std::min(h - 1u, prev_y1 - 1u);

                    if (pass.pipeline != bound) {
                        vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
                        bound = pass.pipeline;
                    }
                    const auto &src = (pass.src_slice < 0) ? input : local[pass.src_slice];
                    const auto &dst = (pass.dst_slice < 0) ? destination : local[pass.dst_slice];
                    const VkDescriptorBufferInfo ubo{parameter_info.buffer, pass.ubo_offset, pass.ubo_size};
                    const FastPush push{w, h, plane_stride, source_stride, dst_s2,
                                        out_y0, out_y1, src_y0, dst_y0, src_y_max};
                    push_constants(cmd, push);
                    push_bindings(cmd, src, dst, ubo, input);
                    const std::uint32_t gx = (w + pass.tile_w - 1) / pass.tile_w;
                    const std::uint32_t gy = ((out_y1 - out_y0) + pass.tile_h - 1) / pass.tile_h;
                    vk->vkCmdDispatch(cmd, gx, gy, 1);
                    if (i + 1 < fast_passes.size())
                        barrier(cmd);
                    else if (sy + eff_sh < h)
                        barrier(cmd, 0);
                    prev_y0 = out_y0;
                    prev_y1 = out_y1;
                }
            }
        } else {
            auto local = fallback_slices;
            for (auto &slice : local)
                slice.buffer = ws_info.buffer;
            const VkDescriptorBufferInfo ssbo{parameter_info.buffer, 0, parameter_info.size};
            vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fallback_pipeline);
            const auto run_fb = [&](const FallbackPush &push, const VkDescriptorBufferInfo &src,
                                    const VkDescriptorBufferInfo &dst, const VkDescriptorBufferInfo &aux,
                                    int ow, int oh, std::uint32_t groups, bool bar) {
                push_bindings(cmd, src, dst, ssbo, aux);
                push_constants(cmd, push);
                vk->vkCmdDispatch(cmd,
                                  (static_cast<std::uint32_t>(ow) + 7) / 8,
                                  (static_cast<std::uint32_t>(oh) + 7) / 8, groups);
                if (bar)
                    barrier(cmd);
            };
            VkDescriptorBufferInfo features = input;
            std::uint32_t feature_stride = source_stride;
            for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
                const auto &l = layers[i];
                const auto &target = local[i % 2];
                run_fb(FallbackPush{0, w, h, l.ci, l.co, feature_stride, w * l.co, 0, l.weights, l.bias},
                       features, target, input, width, height, l.co / 4, true);
                features = target;
                feature_stride = w * l.co;
            }
            const auto &head = layers.back();
            if (aa) {
                run_fb(FallbackPush{6, w, h, head.ci, 4, feature_stride, w * 4, 0, head.weights, head.bias},
                       features, local[2], input, width, height, 1, true);
                run_fb(FallbackPush{5, w, h, 1, 1, source_stride, output_stride, w * 4, 0, 0},
                       input, destination, local[2], width * 2, height * 2, 1, false);
            } else {
                run_fb(FallbackPush{3, w, h, head.ci, 4, feature_stride, (w + 2) * 4, 0, head.weights, head.bias},
                       features, local[2], input, width + 2, height + 2, 1, true);
                run_fb(FallbackPush{1, w, h, 1, 1, source_stride, w * 2, 0, 0, 0},
                       input, local[3], input, width * 2, height, 1, true);
                run_fb(FallbackPush{2, w * 2, h, 1, 1, w * 2, w * 2, 0, 0, 0},
                       local[3], local[4], input, width * 2, height * 2, 1, true);
                run_fb(FallbackPush{4, w, h, 4, 1, (w + 2) * 4, output_stride, w * 2, 0, 0},
                       local[2], destination, local[4], width * 2, height * 2, 1, false);
            }
        }
        exec.submit();
    }
};

Vulkan::Vulkan(const Model &model, int width, int height, VSCore *core, const VSAPI *api, int bits)
    : impl_(std::make_unique<Impl>()) {
    impl_->init(model, width, height, core, api, bits);
}
Vulkan::~Vulkan() = default;

VSFrame *Vulkan::new_frame(const VSFrame *source) const {
    return impl_->gpu->newGPUVideoFrame(&impl_->format, impl_->width * 2, impl_->height * 2, source,
                                        impl_->core);
}
void Vulkan::process(const VSFrame *source, VSFrame *output) const { impl_->process(source, output); }

} // namespace ngu
