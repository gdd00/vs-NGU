#define VK_NO_PROTOTYPES
#include <VSVulkan4.h>
#include "ngu_vulkan.hpp"
#include "ngu_half.hpp"
#include "ngu_half_shader.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
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
#define NGU_FROM_RAW uintBitsToFloat
#define NGU_TO_RAW floatBitsToUint
#extension GL_EXT_spirv_intrinsics : require
#extension GL_EXT_control_flow_attributes : enable
// SPIR-V float controls: SignedZeroInfNanPreserve and RoundingModeRTE (FP32).
spirv_execution_mode(capabilities=[4466], 4461, 32);
spirv_execution_mode(capabilities=[4467], 4462, 32);
#define NGU_UBO_DECL(n4, n8) layout(set=0, binding=2, std140) uniform Params { vec4 ubo[n4]; };
#define NGU_P0(i2) ubo[(i2) * 2u]
#define NGU_P1(i2) ubo[(i2) * 2u + 1u]
#define NGU_LOAD_PAIR(i2, uvar, wa, wb) vec4 wa = ubo[(i2) * 2u], wb = ubo[(i2) * 2u + 1u]
#define NGU_MAC(acc, sv, wv) acc += (sv) * (wv)
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
    std::uint32_t width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3;
};
static_assert(sizeof(FastPush) == 32);

struct LayerInfo {
    std::uint32_t ci, co, weights, bias;
};

struct FastPass {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDeviceSize ubo_offset = 0;
    VkDeviceSize ubo_size = 0;
    std::uint32_t tile_w = 1;
    std::uint32_t tile_h = 1;
    int src_slice = -1; // -1 = input frame, >=0 = workspace slice index
    int dst_slice = -2; // -2 = output frame, >=0 = workspace slice index
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

    // 1. Low Quality 1-Pass 26x26 (local_size=(16,16)=256 threads, 2x2 quad/thread = 676 src / 2704 4K px/WG!).
    // Collapses b1 and b2 into a 2-step stg loop so SASS Binary Size stays ~22-28 KB (< 32 KB L1I cache!) in BOTH FP16 & FP32!
    std::string gen_low_26x26(std::span<const StageDesc> stages, std::uint32_t total_uvec4, bool aa) const {
        const char *comps[4] = {"x", "y", "z", "w"};
        std::string s = shader_preamble;
        s += "layout(local_size_x=16, local_size_y=16, local_size_z=1) in;\n";
        s += aa ? "layout(set=0, binding=0, std430) readonly buffer Source { NGU_RAW s[]; };\n"
                  "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
                : "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
                  "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
        s += "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n"
             "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
             "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n";
        s += aa ? "shared NGU_RAW src_t[34][36];\n" : "shared float src_t[34][36];\n";
        s += "shared vec4 tile[2][32][36];\n"
             "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 16u + tx;\n"
             "    int bx = int(gl_WorkGroupID.x) * 26, by = int(gl_WorkGroupID.y) * 26;\n"
             "    for (uint idx = tid; idx < 1156u; idx += 256u) {\n"
             "        uint ly = idx / 34u, lx = idx - ly * 34u;\n"
             "        uint gx = uint(clamp(bx + int(lx) - 4, 0, int(pc.width) - 1));\n"
             "        uint gy = uint(clamp(by + int(ly) - 4, 0, int(pc.height) - 1));\n"
             "        src_t[ly][lx] = s[gy * pc.src_s + gx];\n"
             "    }\n"
             "    barrier();\n"
             "    uint py0 = ty * 2u, px0 = tx * 2u;\n";

        // b0: 1 -> 8 on 32x32 (256/256 threads active = 100% utilization)
        const std::string rd_pre = aa ? "NGU_FROM_RAW(src_t" : "src_t";
        const std::string rd_post = aa ? ")" : "";
        s += "    {\n"
             "        vec4 q0_0, q0_1, q1_0, q1_1, q2_0, q2_1, q3_0, q3_1;\n" +
             emit_bias_init_n(2, std::to_string(stages[0].b_uoff) + "u", "q", 4, "        ", 0);
        for (int ky = 0; ky < 3; ++ky) {
            s += "        {\n"
                 "            float a0=" + rd_pre + "[py0+" + std::to_string(ky) + "u][px0+0u]" + rd_post + ", "
                 "a1=" + rd_pre + "[py0+" + std::to_string(ky) + "u][px0+1u]" + rd_post + ", "
                 "a2=" + rd_pre + "[py0+" + std::to_string(ky) + "u][px0+2u]" + rd_post + ", "
                 "a3=" + rd_pre + "[py0+" + std::to_string(ky) + "u][px0+3u]" + rd_post + ";\n"
                 "            float b0s=" + rd_pre + "[py0+" + std::to_string(ky + 1) + "u][px0+0u]" + rd_post + ", "
                 "b1s=" + rd_pre + "[py0+" + std::to_string(ky + 1) + "u][px0+1u]" + rd_post + ", "
                 "b2s=" + rd_pre + "[py0+" + std::to_string(ky + 1) + "u][px0+2u]" + rd_post + ", "
                 "b3s=" + rd_pre + "[py0+" + std::to_string(ky + 1) + "u][px0+3u]" + rd_post + ";\n";
            for (int kx = 0; kx < 3; ++kx) {
                const std::string i2 = std::to_string(stages[0].w_uoff + ky * 3 + kx) + "u";
                const std::string tag = "b0_" + std::to_string(ky * 3 + kx);
                s += "            { NGU_LOAD_PAIR(" + i2 + ", uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                     "NGU_MAC(q0_0, a" + std::to_string(kx) + ", wa_" + tag + "); "
                     "NGU_MAC(q1_0, a" + std::to_string(kx + 1) + ", wa_" + tag + "); "
                     "NGU_MAC(q2_0, b" + std::to_string(kx) + "s, wa_" + tag + "); "
                     "NGU_MAC(q3_0, b" + std::to_string(kx + 1) + "s, wa_" + tag + "); "
                     "NGU_MAC(q0_1, a" + std::to_string(kx) + ", wb_" + tag + "); "
                     "NGU_MAC(q1_1, a" + std::to_string(kx + 1) + ", wb_" + tag + "); "
                     "NGU_MAC(q2_1, b" + std::to_string(kx) + "s, wb_" + tag + "); "
                     "NGU_MAC(q3_1, b" + std::to_string(kx + 1) + "s, wb_" + tag + "); }\n";
            }
            s += "        }\n";
        }
        for (int og = 0; og < 2; ++og) {
            const std::string ogs = std::to_string(og);
            s += "        tile[" + ogs + "][py0+0u][px0+0u] = mix(q0_" + ogs + ", q0_" + ogs + "*0.1, lessThan(q0_" + ogs + ", vec4(0.0)));\n"
                 "        tile[" + ogs + "][py0+0u][px0+1u] = mix(q1_" + ogs + ", q1_" + ogs + "*0.1, lessThan(q1_" + ogs + ", vec4(0.0)));\n"
                 "        tile[" + ogs + "][py0+1u][px0+0u] = mix(q2_" + ogs + ", q2_" + ogs + "*0.1, lessThan(q2_" + ogs + ", vec4(0.0)));\n"
                 "        tile[" + ogs + "][py0+1u][px0+1u] = mix(q3_" + ogs + ", q3_" + ogs + "*0.1, lessThan(q3_" + ogs + ", vec4(0.0)));\n";
        }
        s += "    }\n    barrier();\n";

        // b1 (stg=0: 15x15 quads, pad=3) and b2 (stg=1: 14x14 quads, pad=2) sharing ONE loop body (< 24 KB SASS!)
        const std::uint32_t w_stride = stages[2].w_uoff - stages[1].w_uoff;
        const std::uint32_t b_stride = stages[2].b_uoff - stages[1].b_uoff;
        s += "    [[dont_unroll]] for (uint stg = 0u; stg < 2u; ++stg) {\n"
             "        uint q_dim = 15u - stg;\n"
             "        int pad = 3 - int(stg), max_in = 31 - int(stg) * 2;\n"
             "        vec4 q0_0=vec4(0.0), q0_1=vec4(0.0), q1_0=vec4(0.0), q1_1=vec4(0.0), "
             "q2_0=vec4(0.0), q2_1=vec4(0.0), q3_0=vec4(0.0), q3_1=vec4(0.0);\n"
             "        if (tx < q_dim && ty < q_dim) {\n"
             "            int xl = max(0, pad - bx), xh = min(max_in, int(pc.width) - 1 + pad - bx);\n"
             "            int ylo = max(0, pad - by), yhi = min(max_in, int(pc.height) - 1 + pad - by);\n" +
             emit_bias_init_n(2, std::to_string(stages[1].b_uoff) + "u + stg * " + std::to_string(b_stride) + "u",
                              "q", 4, "            ", 1) +
             "            uint u_idx = " + std::to_string(stages[1].w_uoff) + "u + stg * " + std::to_string(w_stride) + "u;\n"
             "            uint rx0=uint(clamp(int(px0)+0,xl,xh)), rx1=uint(clamp(int(px0)+1,xl,xh)), "
             "rx2=uint(clamp(int(px0)+2,xl,xh)), rx3=uint(clamp(int(px0)+3,xl,xh));\n"
             "            [[dont_unroll]] for (uint ky = 0u; ky < pc.k3; ++ky) {\n"
             "                uint ry0 = uint(clamp(int(py0 + ky) + 0, ylo, yhi));\n"
             "                uint ry1 = uint(clamp(int(py0 + ky) + 1, ylo, yhi));\n"
             "                vec4 t0_00=tile[0][ry0][rx0], t0_01=tile[0][ry0][rx1], t0_02=tile[0][ry0][rx2], t0_03=tile[0][ry0][rx3];\n"
             "                vec4 t0_10=tile[0][ry1][rx0], t0_11=tile[0][ry1][rx1], t0_12=tile[0][ry1][rx2], t0_13=tile[0][ry1][rx3];\n"
             "                vec4 t1_00=tile[1][ry0][rx0], t1_01=tile[1][ry0][rx1], t1_02=tile[1][ry0][rx2], t1_03=tile[1][ry0][rx3];\n"
             "                vec4 t1_10=tile[1][ry1][rx0], t1_11=tile[1][ry1][rx1], t1_12=tile[1][ry1][rx2], t1_13=tile[1][ry1][rx3];\n";
        for (int kx = 0; kx < 3; ++kx)
            for (int ig = 0; ig < 2; ++ig)
                for (int c = 0; c < 4; ++c) {
                    const int u_loc = (kx * 2 + ig) * 4 + c;
                    const std::string tag = "lb_" + std::to_string(u_loc);
                    const std::string ta0 = "t" + std::to_string(ig) + "_0" + std::to_string(kx);
                    const std::string tb0 = "t" + std::to_string(ig) + "_0" + std::to_string(kx + 1);
                    const std::string ta1 = "t" + std::to_string(ig) + "_1" + std::to_string(kx);
                    const std::string tb1 = "t" + std::to_string(ig) + "_1" + std::to_string(kx + 1);
                    s += "                { NGU_LOAD_PAIR(u_idx + " + std::to_string(u_loc) + "u, uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                         "NGU_MAC(q0_0, " + ta0 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q1_0, " + tb0 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q2_0, " + ta1 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q3_0, " + tb1 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q0_1, " + ta0 + "." + comps[c] + ", wb_" + tag + "); "
                         "NGU_MAC(q1_1, " + tb0 + "." + comps[c] + ", wb_" + tag + "); "
                         "NGU_MAC(q2_1, " + ta1 + "." + comps[c] + ", wb_" + tag + "); "
                         "NGU_MAC(q3_1, " + tb1 + "." + comps[c] + ", wb_" + tag + "); }\n";
                }
        s += "                u_idx += 24u;\n"
             "            }\n"
             "        }\n"
             "        barrier();\n"
             "        if (tx < q_dim && ty < q_dim) {\n";
        for (int og = 0; og < 2; ++og) {
            const std::string ogs = std::to_string(og);
            s += "            tile[" + ogs + "][py0+0u][px0+0u] = mix(q0_" + ogs + ", q0_" + ogs + "*0.1, lessThan(q0_" + ogs + ", vec4(0.0)));\n"
                 "            tile[" + ogs + "][py0+0u][px0+1u] = mix(q1_" + ogs + ", q1_" + ogs + "*0.1, lessThan(q1_" + ogs + ", vec4(0.0)));\n"
                 "            tile[" + ogs + "][py0+1u][px0+0u] = mix(q2_" + ogs + ", q2_" + ogs + "*0.1, lessThan(q2_" + ogs + ", vec4(0.0)));\n"
                 "            tile[" + ogs + "][py0+1u][px0+1u] = mix(q3_" + ogs + ", q3_" + ogs + "*0.1, lessThan(q3_" + ogs + ", vec4(0.0)));\n";
        }
        s += "        }\n"
             "        barrier();\n"
             "    }\n";

        // Head: 26x26 = 13x13 quads + 4K combine
        s += "    vec4 q0=vec4(0.0), q1=vec4(0.0), q2=vec4(0.0), q3=vec4(0.0);\n"
             "    if (tx < 13u && ty < 13u) {\n"
             "        int xl = max(0, 1 - bx), xh = min(27, int(pc.width) - bx);\n"
             "        int ylo = max(0, 1 - by), yhi = min(27, int(pc.height) - by);\n" +
             emit_bias_init_n(1, std::to_string(stages[3].b_uoff) + "u", "q", 4, "        ", 3) +
             "        uint u_idx = " + std::to_string(stages[3].w_uoff) + "u;\n"
             "        uint rx0=uint(clamp(int(px0)+0,xl,xh)), rx1=uint(clamp(int(px0)+1,xl,xh)), "
             "rx2=uint(clamp(int(px0)+2,xl,xh)), rx3=uint(clamp(int(px0)+3,xl,xh));\n"
             "        [[dont_unroll]] for (uint ky = 0u; ky < pc.k3; ++ky) {\n"
             "            uint ry0 = uint(clamp(int(py0 + ky) + 0, ylo, yhi));\n"
             "            uint ry1 = uint(clamp(int(py0 + ky) + 1, ylo, yhi));\n"
             "            vec4 t0_00=tile[0][ry0][rx0], t0_01=tile[0][ry0][rx1], t0_02=tile[0][ry0][rx2], t0_03=tile[0][ry0][rx3];\n"
             "            vec4 t0_10=tile[0][ry1][rx0], t0_11=tile[0][ry1][rx1], t0_12=tile[0][ry1][rx2], t0_13=tile[0][ry1][rx3];\n"
             "            vec4 t1_00=tile[1][ry0][rx0], t1_01=tile[1][ry0][rx1], t1_02=tile[1][ry0][rx2], t1_03=tile[1][ry0][rx3];\n"
             "            vec4 t1_10=tile[1][ry1][rx0], t1_11=tile[1][ry1][rx1], t1_12=tile[1][ry1][rx2], t1_13=tile[1][ry1][rx3];\n";
        for (int kx = 0; kx < 3; ++kx)
            for (int ig = 0; ig < 2; ++ig)
                for (int c = 0; c < 4; c += 2) {
                    const int u_loc = (kx * 2 + ig) * 2 + c / 2;
                    const std::string tag = "lh_" + std::to_string(u_loc);
                    const std::string ta0 = "t" + std::to_string(ig) + "_0" + std::to_string(kx);
                    const std::string tb0 = "t" + std::to_string(ig) + "_0" + std::to_string(kx + 1);
                    const std::string ta1 = "t" + std::to_string(ig) + "_1" + std::to_string(kx);
                    const std::string tb1 = "t" + std::to_string(ig) + "_1" + std::to_string(kx + 1);
                    s += "            { NGU_LOAD_PAIR(u_idx + " + std::to_string(u_loc) + "u, uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                         "NGU_MAC(q0, " + ta0 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q1, " + tb0 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q2, " + ta1 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q3, " + tb1 + "." + comps[c] + ", wa_" + tag + "); "
                         "NGU_MAC(q0, " + ta0 + "." + comps[c + 1] + ", wb_" + tag + "); "
                         "NGU_MAC(q1, " + tb0 + "." + comps[c + 1] + ", wb_" + tag + "); "
                         "NGU_MAC(q2, " + ta1 + "." + comps[c + 1] + ", wb_" + tag + "); "
                         "NGU_MAC(q3, " + tb1 + "." + comps[c + 1] + ", wb_" + tag + "); }\n";
                }
        s += "            u_idx += 12u;\n"
             "        }\n    }\n";
        if (!aa) {
            s += "    barrier();\n"
                 "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint idx = tid; idx < 780u; idx += 256u) {\n"
                 "            uint sy = idx / 26u, lx = idx - sy * 26u;\n"
                 "            float s0=src_t[sy+2u][lx+2u], s1=src_t[sy+2u][lx+3u], s2=src_t[sy+2u][lx+4u], s3=src_t[sy+2u][lx+5u], s4=src_t[sy+2u][lx+6u];\n"
                 "            tile[0][sy][lx].x = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                 "            tile[0][sy][lx].y = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                 "        }\n    }\n    barrier();\n";
        }
        s += "    if (tx >= 13u || ty >= 13u) return;\n";
        if (!aa) {
            s += "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n";
        }
        for (int qy_sub = 0; qy_sub < 2; ++qy_sub)
            for (int qx_sub = 0; qx_sub < 2; ++qx_sub) {
                const int qi = qy_sub * 2 + qx_sub;
                const std::string qn = "q" + std::to_string(qi);
                const std::string lx_s = "px0 + " + std::to_string(qx_sub) + "u";
                const std::string ly_s = "py0 + " + std::to_string(qy_sub) + "u";
                s += "    {\n"
                     "        uint x = uint(bx) + " + lx_s + ", y = uint(by) + " + ly_s + ";\n"
                     "        if (x < pc.width && y < pc.height) {\n"
                     "            uint ox = x * 2u, oy = y * 2u;\n";
                if (aa) {
                    const std::string sx0 = "px0 + " + std::to_string(4 + qx_sub) + "u";
                    const std::string sx1 = "px0 + " + std::to_string(5 + qx_sub) + "u";
                    const std::string sy0 = "py0 + " + std::to_string(4 + qy_sub) + "u";
                    const std::string sy1 = "py0 + " + std::to_string(5 + qy_sub) + "u";
                    s += "            NGU_RAW u00 = src_t[" + sy0 + "][" + sx0 + "];\n"
                         "            float s00 = NGU_FROM_RAW(u00);\n"
                         "            float s10 = NGU_FROM_RAW(src_t[" + sy0 + "][" + sx1 + "]);\n"
                         "            float s01 = NGU_FROM_RAW(src_t[" + sy1 + "][" + sx0 + "]);\n"
                         "            float s11 = NGU_FROM_RAW(src_t[" + sy1 + "][" + sx1 + "]);\n"
                         "            precise float p10 = (s00 + s10) * 0.5 + " + qn + ".y;\n"
                         "            precise float p01 = (s00 + s01) * 0.5 + " + qn + ".x;\n"
                         "            precise float p11 = (((s00 + s10) + s01) + s11) * 0.25 + " + qn + ".z;\n"
                         "            d[(oy + 0u) * pc.dst_s + ox + 0u] = u00;\n"
                         "            d[(oy + 0u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p10);\n"
                         "            d[(oy + 1u) * pc.dst_s + ox + 0u] = NGU_TO_RAW(p01);\n"
                         "            d[(oy + 1u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p11);\n";
                } else {
                    s += "            float r0a=tile[0][" + ly_s + "+0u][" + lx_s + "].x, r1a=tile[0][" + ly_s + "+1u][" + lx_s + "].x, "
                         "r2a=tile[0][" + ly_s + "+2u][" + lx_s + "].x, r3a=tile[0][" + ly_s + "+3u][" + lx_s + "].x, r4a=tile[0][" + ly_s + "+4u][" + lx_s + "].x;\n"
                         "            float r0b=tile[0][" + ly_s + "+0u][" + lx_s + "].y, r1b=tile[0][" + ly_s + "+1u][" + lx_s + "].y, "
                         "r2b=tile[0][" + ly_s + "+2u][" + lx_s + "].y, r3b=tile[0][" + ly_s + "+3u][" + lx_s + "].y, r4b=tile[0][" + ly_s + "+4u][" + lx_s + "].y;\n"
                         "            precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + " + qn + ".x;\n"
                         "            precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + " + qn + ".y;\n"
                         "            precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + " + qn + ".z;\n"
                         "            precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + " + qn + ".w;\n"
                         "            d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
                         "            d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
                         "            d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
                         "            d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
                }
                s += "        }\n    }\n";
            }
        s += "}\n";
        return s;
    }

    // 2. 1-Layer Warp-Stripe (32 x (wg_y*py), local_size=(32, wg_y)):
    // 100% immediate constant-bank UBO addressing (zero dynamic LDC VGPRs):
    // - FP16 C=16 (8w, 2py -> 32x16): regs=36, 5 blocks/SM (40 warps/SM), 354 us/layer!
    // - FP32 C=16 (6w, 2py -> 32x12): regs=81, 3 blocks/SM (18 warps/SM), 544 us/layer!
    // - FP32 C=12 (6w, 2py -> 32x12): regs=62, 4 blocks/SM (24 warps/SM), 271 us/layer!
    // - FP16 C=12 (4w, 3py -> 32x12): regs=39, 6 blocks/SM (24 warps/SM), 219 us/layer!
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
             "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
             "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n"
             "shared vec4 tile[" + std::to_string(g) + "][" + std::to_string(rows) + "][34];\n"
             "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 32, by = int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    uint gx = uint(clamp(bx + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "    uint gx_r = uint(clamp(bx + 32 + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "    for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "        uint row = uint(clamp(by + int(sy) - 1, 0, int(pc.height) - 1)) * pc.width;\n"
             "        for (int gi = 0; gi < " + std::to_string(g) + "; ++gi) {\n"
             "            uint ps = uint(gi) * pc.plane_stride + row;\n"
             "            tile[gi][sy][tx] = s[ps + gx];\n"
             "            if (tx < 2u) tile[gi][sy][32u + tx] = s[ps + gx_r];\n"
             "        }\n    }\n    barrier();\n"
             "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n"
             "    if (x >= pc.width || y0 >= pc.height) return;\n";
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
                            const std::string tg = "sp_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(f / 2);
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
                            const std::string tg = std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(f / 2);
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
            if (p > 0) s += "    if (y0 + " + std::to_string(p) + "u < pc.height) {\n";
            s += "    { uint ob = (y0 + " + std::to_string(p) + "u) * pc.width + x;\n";
            for (int og = 0; og < g; ++og) {
                const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
                s += "      d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(" + pn + ", " + pn + "*0.1, lessThan(" + pn + ", vec4(0.0)));\n";
            }
            s += "    }\n";
            if (p > 0) s += "    }\n";
        }
        s += "}\n";
        return s;
    }

    // 3. 2-Layer Same-Channel ColumnGroupRows Warp-Stripe (`30 x th`, ZERO bank conflicts, -50% DRAM traffic!):
    std::string gen_2L_col_stripe(std::span<const StageDesc> stages, std::uint32_t total_uvec4,
                                  int wg_y, int py, bool full_unroll) const {
        const int g = stages[0].c_in / 4;
        const int tw = 30, th = wg_y * py - 2, rows = wg_y * py + 2;
        const char *comps[4] = {"x", "y", "z", "w"};
        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=" + std::to_string(wg_y) + ", local_size_z=1) in;\n"
             "layout(set=0, binding=0, std430) readonly buffer Source { vec4 s[]; };\n"
             "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n"
             "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n"
             "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
             "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n"
             "shared vec4 tile[" + std::to_string(g) + "][" + std::to_string(rows) + "][34];\n"
             "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 30, by = int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "    uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "    for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "        uint row = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.width;\n"
             "        for (int gi = 0; gi < " + std::to_string(g) + "; ++gi) {\n"
             "            uint ps = uint(gi) * pc.plane_stride + row;\n"
             "            tile[gi][sy][tx] = s[ps + gx];\n"
             "            if (tx < 2u) tile[gi][sy][32u + tx] = s[ps + gx_r];\n"
             "        }\n    }\n    barrier();\n";

        if (full_unroll) {
            s += "    {\n";
            for (int p = 0; p < py; ++p)
                for (int og = 0; og < g; ++og)
                    s += "        vec4 p" + std::to_string(p) + "_" + std::to_string(og) + ";\n";
            s += emit_bias_init_n(g, std::to_string(stages[0].b_uoff) + "u", "p", py, "        ", 20);
            for (int kx = 0; kx < 3; ++kx)
                for (int ig = 0; ig < g; ++ig) {
                    s += "        {\n";
                    for (int r = 0; r < py + 2; ++r)
                        s += "            vec4 t" + std::to_string(r) + " = tile[" + std::to_string(ig) + "][yl + " +
                             std::to_string(r) + "u][tx + " + std::to_string(kx) + "u];\n";
                    const std::uint32_t base = stages[0].w_uoff + static_cast<std::uint32_t>((kx * g + ig) * (6 * g));
                    for (int ky = 0; ky < 3; ++ky)
                        for (int f = 0; f < 4 * g; f += 2) {
                            const int c0 = f / g, og0 = f % g, c1 = (f + 1) / g, og1 = (f + 1) % g;
                            const std::string tg = "c0_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(f / 2);
                            s += "            { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                    s += "        }\n";
                }
            s += "        barrier();\n";
            for (int p = 0; p < py; ++p)
                for (int og = 0; og < g; ++og) {
                    const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
                    s += "        tile[" + std::to_string(og) + "][yl+" + std::to_string(p) + "u][tx] = mix(" + pn + ", " + pn + "*0.1, lessThan(" + pn + ", vec4(0.0)));\n";
                }
            s += "        barrier();\n    }\n"
                 "    if (tx >= 30u || uint(bx) + tx >= pc.width) return;\n";
            for (int p = 0; p < py; ++p)
                for (int og = 0; og < g; ++og)
                    s += "    vec4 p" + std::to_string(p) + "_" + std::to_string(og) + ";\n";
            s += emit_bias_init_n(g, std::to_string(stages[1].b_uoff) + "u", "p", py, "    ", 21) +
                 "    int xl = max(0, 1 - bx), xh = min(31, int(pc.width) - bx);\n"
                 "    int ylo = max(0, 1 - by), yhi = min(" + std::to_string(th + 1) + ", int(pc.height) - by);\n";
            for (int r = 0; r < py + 2; ++r)
                s += "    uint ry" + std::to_string(r) + " = uint(clamp(int(yl) + " + std::to_string(r) + ", ylo, yhi));\n";
            for (int kx = 0; kx < 3; ++kx) {
                s += "    { uint rx = uint(clamp(int(tx) + " + std::to_string(kx) + ", xl, xh));\n";
                for (int ig = 0; ig < g; ++ig) {
                    s += "      {\n";
                    for (int r = 0; r < py + 2; ++r)
                        s += "        vec4 t" + std::to_string(r) + " = tile[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx];\n";
                    const std::uint32_t base = stages[1].w_uoff + static_cast<std::uint32_t>((kx * g + ig) * (6 * g));
                    for (int ky = 0; ky < 3; ++ky)
                        for (int f = 0; f < 4 * g; f += 2) {
                            const int c0 = f / g, og0 = f % g, c1 = (f + 1) / g, og1 = (f + 1) % g;
                            const std::string tg = "c1_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(f / 2);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
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
                s += "    if (yl + " + std::to_string(p) + "u < " + std::to_string(th) + "u && uint(by) + yl + " + std::to_string(p) + "u < pc.height) {\n"
                     "        uint ob = (uint(by) + yl + " + std::to_string(p) + "u) * pc.width + uint(bx) + tx;\n";
                for (int og = 0; og < g; ++og) {
                    const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
                    s += "        d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(" + pn + ", " + pn + "*0.1, lessThan(" + pn + ", vec4(0.0)));\n";
                }
                s += "    }\n";
            }
            s += "}\n";
            return s;
        }

        const std::uint32_t w_stride = stages[1].w_uoff - stages[0].w_uoff;
        const std::uint32_t b_stride = stages[1].b_uoff - stages[0].b_uoff;
        s += "    [[dont_unroll]] for (uint stg = 0u; stg < 2u; ++stg) {\n"
             "        int xl = (stg == 0u) ? 0 : max(0, 1 - bx);\n"
             "        int xh = (stg == 0u) ? 33 : min(31, int(pc.width) - bx);\n"
             "        int ylo = (stg == 0u) ? 0 : max(0, 1 - by);\n"
             "        int yhi = (stg == 0u) ? " + std::to_string(rows - 1) + " : min(" + std::to_string(th + 1) + ", int(pc.height) - by);\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g; ++og)
                s += "        vec4 p" + std::to_string(p) + "_" + std::to_string(og) + ";\n";
        s += emit_bias_init_n(g, std::to_string(stages[0].b_uoff) + "u + stg * " + std::to_string(b_stride) + "u",
                              "p", py, "        ", 2);
        for (int r = 0; r < py + 2; ++r)
            s += "        uint ry" + std::to_string(r) + " = uint(clamp(int(yl) + " + std::to_string(r) + ", ylo, yhi));\n";
        s += "        [[dont_unroll]] for (uint kx = 0u; kx < pc.k3; ++kx) {\n"
             "            uint rx = uint(clamp(int(tx + kx), xl, xh));\n"
             "            uint kbase = " + std::to_string(stages[0].w_uoff) + "u + stg * " + std::to_string(w_stride) +
             "u + kx * " + std::to_string(6 * g * g) + "u;\n";
        for (int ig = 0; ig < g; ++ig) {
            s += "            {\n";
            for (int r = 0; r < py + 2; ++r)
                s += "                vec4 t" + std::to_string(r) + " = tile[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx];\n";
            for (int ky = 0; ky < 3; ++ky)
                for (int f = 0; f < 4 * g; f += 2) {
                    const int c0 = f / g, og0 = f % g, c1 = (f + 1) / g, og1 = (f + 1) % g;
                    const std::string tg = "cs_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(f / 2);
                    s += "                { NGU_LOAD_PAIR(kbase + " + std::to_string(ig * 6 * g + ky * 2 * g + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                    for (int p = 0; p < py; ++p)
                        s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                    for (int p = 0; p < py; ++p)
                        s += "NGU_MAC(p" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                    s += "}\n";
                }
            s += "            }\n";
        }
        s += "        }\n"
             "        if (stg == 0u) {\n"
             "            barrier();\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g; ++og) {
                const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
                s += "            tile[" + std::to_string(og) + "][yl+" + std::to_string(p) + "u][tx] = mix(" + pn + ", " + pn + "*0.1, lessThan(" + pn + ", vec4(0.0)));\n";
            }
        s += "            barrier();\n"
             "        } else {\n"
             "            uint x = uint(bx) + tx, y0 = uint(by) + yl;\n"
             "            if (tx < " + std::to_string(tw) + "u && x < pc.width) {\n";
        for (int p = 0; p < py; ++p) {
            s += "                if (yl + " + std::to_string(p) + "u < " + std::to_string(th) + "u && y0 + " + std::to_string(p) + "u < pc.height) {\n"
                 "                    uint ob = (y0 + " + std::to_string(p) + "u) * pc.width + x;\n";
            for (int og = 0; og < g; ++og) {
                const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
                s += "                    d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(" + pn + ", " + pn + "*0.1, lessThan(" + pn + ", vec4(0.0)));\n";
            }
            s += "                }\n";
        }
        s += "            }\n        }\n    }\n}\n";
        return s;
    }

    // 4. Standalone 32x12 3py Head + Combine (130 us FP16 C=16, 199 us FP32 C=12, 332 us FP32 C=16).
    std::string gen_head_32x12(const StageDesc &head, std::uint32_t total_uvec4, bool aa, bool alias_bx = false) const {
        const int g = head.c_in / 4;
        const char *comps[4] = {"x", "y", "z", "w"};
        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=4, local_size_z=1) in;\n"
             "layout(set=0, binding=0, std430) readonly buffer Features { vec4 feat[]; };\n";
        s += aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
                : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
        s += "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n";
        s += aa ? "layout(set=0, binding=3, std430) readonly buffer OrigSrc { NGU_RAW src_in[]; };\n"
                : "layout(set=0, binding=3, std430) readonly buffer OrigSrc { float src_in[]; };\n";
        s += "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n"
             "shared vec4 tile[" + std::to_string(g) + "][14][34];\n";
        if (aa)
            s += "shared NGU_RAW src_u_t[14][34];\n";
        else if (!alias_bx)
            s += "shared float src_t[16][36];\nshared float bx_t[16][64];\n";
        s += "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * 3u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 32, by = int(gl_WorkGroupID.y) * 12;\n"
             "    {\n"
             "        uint gx = uint(clamp(bx + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 1, 0, int(pc.width) - 1));\n"
             "        for (uint sy = ty; sy < 14u; sy += 4u) {\n"
             "            uint row = uint(clamp(by + int(sy) - 1, 0, int(pc.height) - 1)) * pc.width;\n"
             "            for (int gi = 0; gi < " + std::to_string(g) + "; ++gi) {\n"
             "                uint ps = uint(gi) * pc.plane_stride + row;\n"
             "                tile[gi][sy][tx] = feat[ps + gx];\n"
             "                if (tx < 2u) tile[gi][sy][32u + tx] = feat[ps + gx_r];\n"
             "            }\n";
        if (aa) {
            s += "            uint srow = uint(clamp(by + int(sy) - 1, 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            src_u_t[sy][tx] = src_in[srow + gx];\n"
                 "            if (tx < 2u) src_u_t[sy][32u + tx] = src_in[srow + gx_r];\n";
        }
        s += "        }\n    }\n";
        if (!aa && !alias_bx) {
            s += "    {\n"
                 "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                 "            uint row = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            src_t[sy][tx] = src_in[row + gx];\n"
                 "            if (tx < 4u) src_t[sy][32u + tx] = src_in[row + gx_r];\n"
                 "        }\n    }\n";
        }
        s += "    barrier();\n";
        if (!aa && !alias_bx) {
            s += "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                 "            float s0=src_t[sy][tx+0u], s1=src_t[sy][tx+1u], s2=src_t[sy][tx+2u], s3=src_t[sy][tx+3u], s4=src_t[sy][tx+4u];\n"
                 "            bx_t[sy][tx*2u+0u] = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                 "            bx_t[sy][tx*2u+1u] = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                 "        }\n    }\n    barrier();\n";
        }
        if (!alias_bx)
            s += "    if (uint(bx) + tx >= pc.width || uint(by) + yl >= pc.height) return;\n";
        s += "    vec4 h0, h1, h2;\n" +
             emit_bias_init_n(1, std::to_string(head.b_uoff) + "u", "h", 3, "    ", 4);
        if (head.col_order) {
            for (int kx = 0; kx < 3; ++kx)
                for (int ig = 0; ig < g; ++ig) {
                    s += "    {\n";
                    for (int r = 0; r < 5; ++r)
                        s += "        vec4 t" + std::to_string(r) + " = tile[" + std::to_string(ig) + "][yl+" +
                             std::to_string(r) + "u][tx+" + std::to_string(kx) + "u];\n";
                    const std::uint32_t base = head.w_uoff + static_cast<std::uint32_t>((kx * g + ig) * 6);
                    for (int ky = 0; ky < 3; ++ky)
                        for (int c = 0; c < 4; c += 2) {
                            const std::string tg = "hc_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(c);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 + c / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); "
                                 "NGU_MAC(h0, t" + std::to_string(ky) + "." + comps[c] + ", wa_" + tg + "); "
                                 "NGU_MAC(h1, t" + std::to_string(ky + 1) + "." + comps[c] + ", wa_" + tg + "); "
                                 "NGU_MAC(h2, t" + std::to_string(ky + 2) + "." + comps[c] + ", wa_" + tg + "); "
                                 "NGU_MAC(h0, t" + std::to_string(ky) + "." + comps[c + 1] + ", wb_" + tg + "); "
                                 "NGU_MAC(h1, t" + std::to_string(ky + 1) + "." + comps[c + 1] + ", wb_" + tg + "); "
                                 "NGU_MAC(h2, t" + std::to_string(ky + 2) + "." + comps[c + 1] + ", wb_" + tg + "); }\n";
                        }
                    s += "    }\n";
                }
        } else {
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    for (int ig = 0; ig < g; ++ig) {
                        s += "    {\n"
                             "        vec4 t0 = tile[" + std::to_string(ig) + "][yl+" + std::to_string(ky + 0) + "u][tx+" + std::to_string(kx) + "u];\n"
                             "        vec4 t1 = tile[" + std::to_string(ig) + "][yl+" + std::to_string(ky + 1) + "u][tx+" + std::to_string(kx) + "u];\n"
                             "        vec4 t2 = tile[" + std::to_string(ig) + "][yl+" + std::to_string(ky + 2) + "u][tx+" + std::to_string(kx) + "u];\n";
                        const std::uint32_t base = head.w_uoff + static_cast<std::uint32_t>(((ky * 3 + kx) * g + ig) * 2);
                        for (int c = 0; c < 4; c += 2) {
                            const std::string tg = "hs_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(c);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + c / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); "
                                 "NGU_MAC(h0, t0." + comps[c] + ", wa_" + tg + "); "
                                 "NGU_MAC(h1, t1." + comps[c] + ", wa_" + tg + "); "
                                 "NGU_MAC(h2, t2." + comps[c] + ", wa_" + tg + "); "
                                 "NGU_MAC(h0, t0." + comps[c + 1] + ", wb_" + tg + "); "
                                 "NGU_MAC(h1, t1." + comps[c + 1] + ", wb_" + tg + "); "
                                 "NGU_MAC(h2, t2." + comps[c + 1] + ", wb_" + tg + "); }\n";
                        }
                        s += "    }\n";
                    }
        }
        if (!aa && alias_bx) {
            // Reuse tile[0..1] for OrigSrc -> bx on strict 32-KB devices without extra smem
            s += "    barrier();\n"
                 "    {\n"
                 "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                 "            uint row = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            uint sl = sy >> 1u;\n"
                 "            if ((sy & 1u) == 0u) { tile[0][sl][tx].x = src_in[row + gx]; if (tx < 4u) tile[0][sl][32u + tx].x = src_in[row + gx_r]; }\n"
                 "            else                 { tile[0][sl][tx].y = src_in[row + gx]; if (tx < 4u) tile[0][sl][32u + tx].y = src_in[row + gx_r]; }\n"
                 "        }\n    }\n    barrier();\n"
                 "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                 "            uint sl = sy >> 1u;\n"
                 "            float s0 = (sy&1u)==0u ? tile[0][sl][tx+0u].x : tile[0][sl][tx+0u].y;\n"
                 "            float s1 = (sy&1u)==0u ? tile[0][sl][tx+1u].x : tile[0][sl][tx+1u].y;\n"
                 "            float s2 = (sy&1u)==0u ? tile[0][sl][tx+2u].x : tile[0][sl][tx+2u].y;\n"
                 "            float s3 = (sy&1u)==0u ? tile[0][sl][tx+3u].x : tile[0][sl][tx+3u].y;\n"
                 "            float s4 = (sy&1u)==0u ? tile[0][sl][tx+4u].x : tile[0][sl][tx+4u].y;\n"
                 "            if ((sy&1u)==0u) { tile[1][sl][tx].x = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3); tile[1][sl][tx].y = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4); }\n"
                 "            else             { tile[1][sl][tx].z = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3); tile[1][sl][tx].w = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4); }\n"
                 "        }\n    }\n    barrier();\n"
                 "    if (uint(bx) + tx >= pc.width || uint(by) + yl >= pc.height) return;\n";
        }
        s += "    uint x = uint(bx) + tx, y0 = uint(by) + yl, ox = x * 2u, c0 = tx * 2u;\n";
        if (!aa) {
            s += "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n";
        }
        for (int p = 0; p < 3; ++p) {
            const std::string hn = "h" + std::to_string(p);
            if (p > 0) s += "    if (y0 + " + std::to_string(p) + "u < pc.height) {\n";
            s += "    { uint ry = yl + " + std::to_string(p) + "u, oy = (y0 + " + std::to_string(p) + "u) * 2u;\n";
            if (aa) {
                s += "      NGU_RAW u00 = src_u_t[ry+1u][tx+1u];\n"
                     "      float s00 = NGU_FROM_RAW(u00);\n"
                     "      float s10 = NGU_FROM_RAW(src_u_t[ry+1u][tx+2u]);\n"
                     "      float s01 = NGU_FROM_RAW(src_u_t[ry+2u][tx+1u]);\n"
                     "      float s11 = NGU_FROM_RAW(src_u_t[ry+2u][tx+2u]);\n"
                     "      precise float p10 = (s00 + s10) * 0.5 + " + hn + ".y;\n"
                     "      precise float p01 = (s00 + s01) * 0.5 + " + hn + ".x;\n"
                     "      precise float p11 = (((s00 + s10) + s01) + s11) * 0.25 + " + hn + ".z;\n"
                     "      d[(oy + 0u) * pc.dst_s + ox + 0u] = u00;\n"
                     "      d[(oy + 0u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p10);\n"
                     "      d[(oy + 1u) * pc.dst_s + ox + 0u] = NGU_TO_RAW(p01);\n"
                     "      d[(oy + 1u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p11);\n";
            } else if (alias_bx) {
                s += "      float r0a=((ry+0u)&1u)==0u?tile[1][(ry+0u)>>1u][tx].x:tile[1][(ry+0u)>>1u][tx].z, r0b=((ry+0u)&1u)==0u?tile[1][(ry+0u)>>1u][tx].y:tile[1][(ry+0u)>>1u][tx].w;\n"
                     "      float r1a=((ry+1u)&1u)==0u?tile[1][(ry+1u)>>1u][tx].x:tile[1][(ry+1u)>>1u][tx].z, r1b=((ry+1u)&1u)==0u?tile[1][(ry+1u)>>1u][tx].y:tile[1][(ry+1u)>>1u][tx].w;\n"
                     "      float r2a=((ry+2u)&1u)==0u?tile[1][(ry+2u)>>1u][tx].x:tile[1][(ry+2u)>>1u][tx].z, r2b=((ry+2u)&1u)==0u?tile[1][(ry+2u)>>1u][tx].y:tile[1][(ry+2u)>>1u][tx].w;\n"
                     "      float r3a=((ry+3u)&1u)==0u?tile[1][(ry+3u)>>1u][tx].x:tile[1][(ry+3u)>>1u][tx].z, r3b=((ry+3u)&1u)==0u?tile[1][(ry+3u)>>1u][tx].y:tile[1][(ry+3u)>>1u][tx].w;\n"
                     "      float r4a=((ry+4u)&1u)==0u?tile[1][(ry+4u)>>1u][tx].x:tile[1][(ry+4u)>>1u][tx].z, r4b=((ry+4u)&1u)==0u?tile[1][(ry+4u)>>1u][tx].y:tile[1][(ry+4u)>>1u][tx].w;\n"
                     "      precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + " + hn + ".x;\n"
                     "      precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + " + hn + ".y;\n"
                     "      precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + " + hn + ".z;\n"
                     "      precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + " + hn + ".w;\n"
                     "      d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
                     "      d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
                     "      d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
                     "      d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
            } else {
                s += "      float r0a=bx_t[ry+0u][c0], r1a=bx_t[ry+1u][c0], r2a=bx_t[ry+2u][c0], r3a=bx_t[ry+3u][c0], r4a=bx_t[ry+4u][c0];\n"
                     "      float r0b=bx_t[ry+0u][c0+1u], r1b=bx_t[ry+1u][c0+1u], r2b=bx_t[ry+2u][c0+1u], r3b=bx_t[ry+3u][c0+1u], r4b=bx_t[ry+4u][c0+1u];\n"
                     "      precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + " + hn + ".x;\n"
                     "      precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + " + hn + ".y;\n"
                     "      precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + " + hn + ".z;\n"
                     "      precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + " + hn + ".w;\n"
                     "      d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
                     "      d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
                     "      d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
                     "      d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
            }
            s += "    }\n";
            if (p > 0) s += "    }\n";
        }
        s += "}\n";
        return s;
    }

    // 5. Pure Warp-Stripe 2-Stage (Spatial -> Col [+ Combine]) on `30 x th` (local_size=(32, wg_y)):
    // Eliminates ALL `tid < 119` intra-warp branch divergence AND 2x2 quad bank conflicts!
    std::string gen_2L_spa_col_stripe(std::span<const StageDesc> stages, std::uint32_t total_uvec4,
                                      bool aa, int wg_y, int py, bool full_unr_s1 = true) const {
        const bool scalar_in = (stages[0].c_in == 1);
        const bool tail_out = stages[1].is_head;
        const int g_in0 = stages[0].c_in / 4;
        const int g_mid = stages[0].c_out / 4;
        const int g_last = tail_out ? 1 : (stages[1].c_out / 4);
        const int g_smem = scalar_in ? g_mid : std::max(g_in0, g_mid);
        const int th = wg_y * py - 2, s0_rows = wg_y * py, rows = wg_y * py + 2;
        const char *comps[4] = {"x", "y", "z", "w"};

        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=" + std::to_string(wg_y) + ", local_size_z=1) in;\n";
        s += scalar_in
            ? "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
            : "layout(set=0, binding=0, std430) readonly buffer Source { vec4 s[]; };\n";
        if (tail_out) {
            s += aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
                    : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
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
        s += "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n";

        if (scalar_in) {
            s += "shared float src_t[" + std::to_string(rows) + "][36];\n"
                 "shared vec4 smem[" + std::to_string(g_smem) + "][" + std::to_string(s0_rows) + "][34];\n";
        } else {
            s += "shared vec4 smem[" + std::to_string(g_smem) + "][" + std::to_string(rows) + "][34];\n";
            if (tail_out && aa)
                s += "shared NGU_RAW src_u_t[" + std::to_string(th + 1) + "][32];\n";
            else if (tail_out)
                s += "shared float src_t[" + std::to_string(rows) + "][36];\n";
        }

        s += "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, yl = ty * " + std::to_string(py) + "u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 30, by = int(gl_WorkGroupID.y) * " + std::to_string(th) + ";\n"
             "    {\n"
             "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
             "        for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
             "            uint cy = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1));\n";
        if (scalar_in) {
            s += "            uint srow = cy * pc.src_s;\n"
                 "            src_t[sy][tx] = s[srow + gx];\n"
                 "            if (tx < 2u) src_t[sy][32u + tx] = s[srow + gx_r];\n";
        } else {
            s += "            uint row = cy * pc.width;\n"
                 "            for (int gi = 0; gi < " + std::to_string(g_in0) + "; ++gi) {\n"
                 "                uint ps = uint(gi) * pc.plane_stride + row;\n"
                 "                smem[gi][sy][tx] = s[ps + gx];\n"
                 "                if (tx < 2u) smem[gi][sy][32u + tx] = s[ps + gx_r];\n"
                 "            }\n";
            if (tail_out && !aa) {
                s += "            uint srow = cy * pc.src_s;\n"
                     "            src_t[sy][tx] = src_in[srow + gx];\n"
                     "            if (tx < 4u) src_t[sy][32u + tx] = src_in[srow + gx_r];\n";
            }
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

        // Stage 0: Spatial warp-stripe (100% active threads!)
        s += "    {\n";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g_mid; ++og)
                s += "        vec4 q" + std::to_string(p) + "_" + std::to_string(og) + ";\n";
        s += emit_bias_init_n(g_mid, std::to_string(stages[0].b_uoff) + "u", "q", py, "        ", 10);
        if (scalar_in) {
            const int uvec4_per_tap = (g_mid + 1) / 2;
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    s += "        {\n";
                    for (int p = 0; p < py; ++p)
                        s += "            float sv" + std::to_string(p) + " = src_t[yl+" + std::to_string(ky + p) + "u][tx+" + std::to_string(kx) + "u];\n";
                    const int tap = ky * 3 + kx;
                    for (int og = 0; og < g_mid; og += 2) {
                        const std::string i2 = std::to_string(stages[0].w_uoff + tap * uvec4_per_tap + og / 2) + "u";
                        const std::string tg = "s0_" + std::to_string(tap) + "_" + std::to_string(og);
                        if (og + 1 < g_mid) {
                            s += "            { NGU_LOAD_PAIR(" + i2 + ", uw_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og) + ", sv" + std::to_string(p) + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og + 1) + ", sv" + std::to_string(p) + ", wb_" + tg + "); ";
                            s += "}\n";
                        } else {
                            s += "            { vec4 wa_" + tg + " = NGU_P0(" + i2 + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og) + ", sv" + std::to_string(p) + ", wa_" + tg + "); ";
                            s += "}\n";
                        }
                    }
                    s += "        }\n";
                }
        } else {
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    for (int ig = 0; ig < g_in0; ++ig) {
                        s += "        {\n";
                        for (int p = 0; p < py; ++p)
                            s += "            vec4 t" + std::to_string(p) + " = smem[" + std::to_string(ig) + "][yl+" + std::to_string(ky + p) + "u][tx+" + std::to_string(kx) + "u];\n";
                        const std::uint32_t base = stages[0].w_uoff + static_cast<std::uint32_t>(((ky * 3 + kx) * g_in0 + ig) * (2 * g_mid));
                        for (int f = 0; f < 4 * g_mid; f += 2) {
                            const int c0 = f / g_mid, og0 = f % g_mid, c1 = (f + 1) / g_mid, og1 = (f + 1) % g_mid;
                            const std::string tg = "sp_" + std::to_string(ky) + "_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(f / 2);
                            s += "            { NGU_LOAD_PAIR(" + std::to_string(base + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og0) + ", t" + std::to_string(p) + "." + comps[c0] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(q" + std::to_string(p) + "_" + std::to_string(og1) + ", t" + std::to_string(p) + "." + comps[c1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                        s += "        }\n";
                    }
            s += "        barrier();\n";
        }
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g_mid; ++og) {
                const std::string qn = "q" + std::to_string(p) + "_" + std::to_string(og);
                s += "        smem[" + std::to_string(og) + "][yl+" + std::to_string(p) + "u][tx] = mix(" + qn + ", " + qn + "*0.1, lessThan(" + qn + ", vec4(0.0)));\n";
            }
        s += "        barrier();\n    }\n";

        // Stage 1: ColumnGroupRows warp-stripe
        if (!(tail_out && !aa))
            s += "    if (tx >= 30u || uint(bx) + tx >= pc.width) return;\n";
        const char *pfx = tail_out ? "h" : "p";
        for (int p = 0; p < py; ++p)
            for (int og = 0; og < g_last; ++og)
                s += "    vec4 " + std::string(pfx) + std::to_string(p) + ((g_last == 1) ? "" : ("_" + std::to_string(og))) + ";\n";
        s += emit_bias_init_n(g_last, std::to_string(stages[1].b_uoff) + "u", pfx, py, "    ", 20) +
             "    int xl = max(0, 1 - bx), xh = min(31, int(pc.width) - bx);\n"
             "    int ylo = max(0, 1 - by), yhi = min(" + std::to_string(th + 1) + ", int(pc.height) - by);\n";
        for (int r = 0; r < py + 2; ++r)
            s += "    uint ry" + std::to_string(r) + " = uint(clamp(int(yl) + " + std::to_string(r) + ", ylo, yhi));\n";
        if (full_unr_s1) {
            for (int kx = 0; kx < 3; ++kx) {
                s += "    { uint rx = uint(clamp(int(tx) + " + std::to_string(kx) + ", xl, xh));\n";
                for (int ig = 0; ig < g_mid; ++ig) {
                    s += "      {\n";
                    for (int r = 0; r < py + 2; ++r)
                        s += "        vec4 t" + std::to_string(r) + " = smem[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx];\n";
                    const std::uint32_t base = stages[1].w_uoff + static_cast<std::uint32_t>((kx * g_mid + ig) * (6 * g_last));
                    for (int ky = 0; ky < 3; ++ky)
                        for (int f = 0; f < 4 * g_last; f += 2) {
                            const int c0 = f / g_last, og0 = f % g_last, c1 = (f + 1) / g_last, og1 = (f + 1) % g_last;
                            const std::string s0 = (g_last == 1) ? "" : ("_" + std::to_string(og0));
                            const std::string s1 = (g_last == 1) ? "" : ("_" + std::to_string(og1));
                            const std::string tg = "u1_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(f / 2);
                            s += "        { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g_last + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(" + std::string(pfx) + std::to_string(p) + s0 + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                            for (int p = 0; p < py; ++p)
                                s += "NGU_MAC(" + std::string(pfx) + std::to_string(p) + s1 + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                            s += "}\n";
                        }
                    s += "      }\n";
                }
                s += "    }\n";
            }
        } else {
            s += "    [[dont_unroll]] for (uint kx = 0u; kx < pc.k3; ++kx) {\n"
                 "        uint rx = uint(clamp(int(tx + kx), xl, xh));\n"
                 "        uint kbase = " + std::to_string(stages[1].w_uoff) + "u + kx * " + std::to_string(6 * g_mid * g_last) + "u;\n";
            for (int ig = 0; ig < g_mid; ++ig) {
                s += "        {\n";
                for (int r = 0; r < py + 2; ++r)
                    s += "            vec4 t" + std::to_string(r) + " = smem[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx];\n";
                for (int ky = 0; ky < 3; ++ky)
                    for (int f = 0; f < 4 * g_last; f += 2) {
                        const int c0 = f / g_last, og0 = f % g_last, c1 = (f + 1) / g_last, og1 = (f + 1) % g_last;
                        const std::string s0 = (g_last == 1) ? "" : ("_" + std::to_string(og0));
                        const std::string s1 = (g_last == 1) ? "" : ("_" + std::to_string(og1));
                        const std::string tg = "r1_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(f / 2);
                        s += "            { NGU_LOAD_PAIR(kbase + " + std::to_string(ig * 6 * g_last + ky * 2 * g_last + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); ";
                        for (int p = 0; p < py; ++p)
                            s += "NGU_MAC(" + std::string(pfx) + std::to_string(p) + s0 + ", t" + std::to_string(ky + p) + "." + comps[c0] + ", wa_" + tg + "); ";
                        for (int p = 0; p < py; ++p)
                            s += "NGU_MAC(" + std::string(pfx) + std::to_string(p) + s1 + ", t" + std::to_string(ky + p) + "." + comps[c1] + ", wb_" + tg + "); ";
                        s += "}\n";
                    }
                s += "        }\n";
            }
            s += "    }\n";
        }

        if (!tail_out) {
            s += "    uint x = uint(bx) + tx, y0 = uint(by) + yl;\n";
            for (int p = 0; p < py; ++p) {
                s += "    if (yl + " + std::to_string(p) + "u < " + std::to_string(th) + "u && y0 + " + std::to_string(p) + "u < pc.height) {\n"
                     "        uint ob = (y0 + " + std::to_string(p) + "u) * pc.width + x;\n";
                for (int og = 0; og < g_last; ++og) {
                    const std::string v = "p" + std::to_string(p) + "_" + std::to_string(og);
                    s += "        d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(" + v + ", " + v + "*0.1, lessThan(" + v + ", vec4(0.0)));\n";
                }
                s += "    }\n";
            }
        } else {
            if (!aa) {
                // Compute horizontal 4-tap into smem[0] after head finishes (saves an extra bx_t shared array!)
                s += "    barrier();\n"
                     "    {\n"
                     "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                     "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                     "        for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += " + std::to_string(wg_y) + "u) {\n"
                     "            float s0=src_t[sy][tx+0u], s1=src_t[sy][tx+1u], s2=src_t[sy][tx+2u], s3=src_t[sy][tx+3u], s4=src_t[sy][tx+4u];\n"
                     "            smem[0][sy][tx].x = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                     "            smem[0][sy][tx].y = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                     "        }\n    }\n    barrier();\n"
                     "    if (tx >= 30u || uint(bx) + tx >= pc.width) return;\n";
            }
            s += "    uint x = uint(bx) + tx, y0 = uint(by) + yl, ox = x * 2u;\n";
            if (!aa) {
                s += "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                     "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n";
            }
            for (int p = 0; p < py; ++p) {
                const std::string hn = "h" + std::to_string(p);
                s += "    if (yl + " + std::to_string(p) + "u < " + std::to_string(th) + "u && y0 + " + std::to_string(p) + "u < pc.height) {\n"
                     "        uint ry = yl + " + std::to_string(p) + "u, oy = (y0 + " + std::to_string(p) + "u) * 2u;\n";
                if (aa) {
                    s += "        NGU_RAW u00 = src_u_t[ry+0u][tx+0u];\n"
                         "        float s00 = NGU_FROM_RAW(u00);\n"
                         "        float s10 = NGU_FROM_RAW(src_u_t[ry+0u][tx+1u]);\n"
                         "        float s01 = NGU_FROM_RAW(src_u_t[ry+1u][tx+0u]);\n"
                         "        float s11 = NGU_FROM_RAW(src_u_t[ry+1u][tx+1u]);\n"
                         "        precise float p10 = (s00 + s10) * 0.5 + " + hn + ".y;\n"
                         "        precise float p01 = (s00 + s01) * 0.5 + " + hn + ".x;\n"
                         "        precise float p11 = (((s00 + s10) + s01) + s11) * 0.25 + " + hn + ".z;\n"
                         "        d[(oy + 0u) * pc.dst_s + ox + 0u] = u00;\n"
                         "        d[(oy + 0u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p10);\n"
                         "        d[(oy + 1u) * pc.dst_s + ox + 0u] = NGU_TO_RAW(p01);\n"
                         "        d[(oy + 1u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p11);\n";
                } else {
                    s += "        float r0a=smem[0][ry+0u][tx].x, r1a=smem[0][ry+1u][tx].x, r2a=smem[0][ry+2u][tx].x, r3a=smem[0][ry+3u][tx].x, r4a=smem[0][ry+4u][tx].x;\n"
                         "        float r0b=smem[0][ry+0u][tx].y, r1b=smem[0][ry+1u][tx].y, r2b=smem[0][ry+2u][tx].y, r3b=smem[0][ry+3u][tx].y, r4b=smem[0][ry+4u][tx].y;\n"
                         "        precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + " + hn + ".x;\n"
                         "        precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + " + hn + ".y;\n"
                         "        precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + " + hn + ".z;\n"
                         "        precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + " + hn + ".w;\n"
                         "        d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
                         "        d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
                         "        d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
                         "        d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
                }
                s += "    }\n";
            }
        }
        s += "}\n";
        return s;
    }

    // 6. Legacy 2-Stage 32x12 (32.7 KB FP32) retained ONLY for VeryHigh FP32 Pass 1 on strict 32-KB devices.
    std::string gen_2L_32x12(std::span<const StageDesc> stages, std::uint32_t total_uvec4,
                             bool aa, bool full_unr_s1, bool alias_bx = false) const {
        const bool scalar_in = (stages[0].c_in == 1);
        const bool tail_out = stages[1].is_head;
        const int g_mid = stages[0].c_out / 4;
        const int g_last = tail_out ? 1 : (stages[1].c_out / 4);
        const int g_smem = scalar_in ? g_mid : std::max(stages[0].c_in / 4, g_mid);
        const char *comps[4] = {"x", "y", "z", "w"};

        std::string s = shader_preamble;
        s += "layout(local_size_x=32, local_size_y=4, local_size_z=1) in;\n";
        s += scalar_in
            ? "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
            : "layout(set=0, binding=0, std430) readonly buffer Source { vec4 s[]; };\n";
        if (tail_out) {
            s += aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
                    : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
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
        s += "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n";

        if (scalar_in) {
            s += "shared float src_t[16][36];\n"
                 "shared vec4 smem[" + std::to_string(g_smem) + "][14][34];\n";
        } else {
            s += "shared vec4 smem[" + std::to_string(g_smem) + "][16][36];\n";
            if (tail_out && aa)
                s += "shared NGU_RAW src_u_t[13][34];\n";
            else if (tail_out && !alias_bx)
                s += "shared float src_t[16][36];\nshared float bx_t[16][64];\n";
        }

        s += "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 32u + tx, yl = ty * 3u;\n"
             "    int bx = int(gl_WorkGroupID.x) * 32, by = int(gl_WorkGroupID.y) * 12;\n";

        if (scalar_in) {
            s += "    {\n"
                 "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                 "            uint row = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
                 "            src_t[sy][tx] = s[row + gx];\n"
                 "            if (tx < 4u) src_t[sy][32u + tx] = s[row + gx_r];\n"
                 "        }\n    }\n";
        } else {
            const int g_in0 = stages[0].c_in / 4;
            s += "    {\n"
                 "        uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
                 "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                 "            uint row = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.width;\n"
                 "            for (int gi = 0; gi < " + std::to_string(g_in0) + "; ++gi) {\n"
                 "                uint ps = uint(gi) * pc.plane_stride + row;\n"
                 "                smem[gi][sy][tx] = s[ps + gx];\n"
                 "                if (tx < 4u) smem[gi][sy][32u + tx] = s[ps + gx_r];\n"
                 "            }\n";
            if (tail_out && !aa && !alias_bx) {
                s += "            uint srow = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
                     "            src_t[sy][tx] = src_in[srow + gx];\n"
                     "            if (tx < 4u) src_t[sy][32u + tx] = src_in[srow + gx_r];\n";
            }
            s += "        }\n    }\n";
            if (tail_out && aa) {
                s += "    {\n"
                     "        uint gx = uint(clamp(bx + int(tx), 0, int(pc.width) - 1));\n"
                     "        uint gx_r = uint(clamp(bx + 32 + int(tx), 0, int(pc.width) - 1));\n"
                     "        for (uint sy = ty; sy < 13u; sy += 4u) {\n"
                     "            uint row = uint(clamp(by + int(sy), 0, int(pc.height) - 1)) * pc.src_s;\n"
                     "            src_u_t[sy][tx] = src_in[row + gx];\n"
                     "            if (tx < 1u) src_u_t[sy][32u + tx] = src_in[row + gx_r];\n"
                     "        }\n    }\n";
            }
        }
        s += "    barrier();\n";
        if (tail_out && !aa && !alias_bx) {
            s += "    {\n"
                 "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                 "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                 "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                 "            float s0=src_t[sy][tx+0u], s1=src_t[sy][tx+1u], s2=src_t[sy][tx+2u], s3=src_t[sy][tx+3u], s4=src_t[sy][tx+4u];\n"
                 "            bx_t[sy][tx*2u+0u] = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                 "            bx_t[sy][tx*2u+1u] = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                 "        }\n    }\n";
        }

        // Stage 0: 119 2x2 quads (Spatial)
        s += "    {\n";
        for (int p = 0; p < 4; ++p)
            for (int og = 0; og < g_mid; ++og)
                s += "        vec4 q" + std::to_string(p) + "_" + std::to_string(og) + " = vec4(0.0);\n";
        s += "        uint py0 = 0u, px0 = 0u;\n"
             "        if (tid < 119u) {\n"
             "            uint qy = tid / 17u, qx = tid - qy * 17u;\n"
             "            py0 = qy * 2u; px0 = qx * 2u;\n" +
             emit_bias_init_n(g_mid, std::to_string(stages[0].b_uoff) + "u", "q", 4, "            ", 10);
        if (scalar_in) {
            const int uvec4_per_tap = (g_mid + 1) / 2;
            for (int ky = 0; ky < 3; ++ky) {
                s += "            {\n"
                     "                float a0=src_t[py0+" + std::to_string(ky) + "u][px0+0u], a1=src_t[py0+" + std::to_string(ky) + "u][px0+1u], "
                     "a2=src_t[py0+" + std::to_string(ky) + "u][px0+2u], a3=src_t[py0+" + std::to_string(ky) + "u][px0+3u];\n"
                     "                float b0s=src_t[py0+" + std::to_string(ky + 1) + "u][px0+0u], b1s=src_t[py0+" + std::to_string(ky + 1) + "u][px0+1u], "
                     "b2s=src_t[py0+" + std::to_string(ky + 1) + "u][px0+2u], b3s=src_t[py0+" + std::to_string(ky + 1) + "u][px0+3u];\n";
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    for (int og = 0; og < g_mid; og += 2) {
                        const std::string i2 = std::to_string(stages[0].w_uoff + tap * uvec4_per_tap + og / 2) + "u";
                        const std::string tag = "s0_" + std::to_string(tap) + "_" + std::to_string(og);
                        if (og + 1 < g_mid) {
                            s += "                { NGU_LOAD_PAIR(" + i2 + ", uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                                 "NGU_MAC(q0_" + std::to_string(og) + ", a" + std::to_string(kx) + ", wa_" + tag + "); "
                                 "NGU_MAC(q1_" + std::to_string(og) + ", a" + std::to_string(kx + 1) + ", wa_" + tag + "); "
                                 "NGU_MAC(q2_" + std::to_string(og) + ", b" + std::to_string(kx) + "s, wa_" + tag + "); "
                                 "NGU_MAC(q3_" + std::to_string(og) + ", b" + std::to_string(kx + 1) + "s, wa_" + tag + "); "
                                 "NGU_MAC(q0_" + std::to_string(og + 1) + ", a" + std::to_string(kx) + ", wb_" + tag + "); "
                                 "NGU_MAC(q1_" + std::to_string(og + 1) + ", a" + std::to_string(kx + 1) + ", wb_" + tag + "); "
                                 "NGU_MAC(q2_" + std::to_string(og + 1) + ", b" + std::to_string(kx) + "s, wb_" + tag + "); "
                                 "NGU_MAC(q3_" + std::to_string(og + 1) + ", b" + std::to_string(kx + 1) + "s, wb_" + tag + "); }\n";
                        } else {
                            s += "                { vec4 wa_" + tag + " = NGU_P0(" + i2 + "); "
                                 "NGU_MAC(q0_" + std::to_string(og) + ", a" + std::to_string(kx) + ", wa_" + tag + "); "
                                 "NGU_MAC(q1_" + std::to_string(og) + ", a" + std::to_string(kx + 1) + ", wa_" + tag + "); "
                                 "NGU_MAC(q2_" + std::to_string(og) + ", b" + std::to_string(kx) + "s, wa_" + tag + "); "
                                 "NGU_MAC(q3_" + std::to_string(og) + ", b" + std::to_string(kx + 1) + "s, wa_" + tag + "); }\n";
                        }
                    }
                }
                s += "            }\n";
            }
        } else {
            const int g_in0 = stages[0].c_in / 4;
            // Keep ig rolled when FP32 G>=3 so SASS stays < 28 KB inside L1I cache!
            if (half && g_in0 <= 3) {
                s += "            uint u_idx = " + std::to_string(stages[0].w_uoff) + "u;\n"
                     "            [[dont_unroll]] for (uint ky = 0u; ky < pc.k3; ++ky) {\n";
                for (int ig = 0; ig < g_in0; ++ig)
                    for (int row = 0; row < 2; ++row)
                        for (int col = 0; col < 4; ++col)
                            s += "                vec4 t" + std::to_string(ig) + "_" + std::to_string(row) + "_" + std::to_string(col) +
                                 " = smem[" + std::to_string(ig) + "][py0+ky+" + std::to_string(row) + "u][px0+" + std::to_string(col) + "u];\n";
                for (int kx = 0; kx < 3; ++kx)
                    for (int ig = 0; ig < g_in0; ++ig)
                        for (int f = 0; f < 4 * g_mid; f += 2) {
                            const int c0 = f / g_mid, og0 = f % g_mid, c1 = (f + 1) / g_mid, og1 = (f + 1) % g_mid;
                            const int u_loc = (kx * g_in0 + ig) * (2 * g_mid) + f / 2;
                            const std::string tag = "hq_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(f / 2);
                            const std::string ta0 = "t" + std::to_string(ig) + "_0_" + std::to_string(kx);
                            const std::string tb0 = "t" + std::to_string(ig) + "_0_" + std::to_string(kx + 1);
                            const std::string ta1 = "t" + std::to_string(ig) + "_1_" + std::to_string(kx);
                            const std::string tb1 = "t" + std::to_string(ig) + "_1_" + std::to_string(kx + 1);
                            s += "                { NGU_LOAD_PAIR(u_idx + " + std::to_string(u_loc) + "u, uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                                 "NGU_MAC(q0_" + std::to_string(og0) + ", " + ta0 + "." + comps[c0] + ", wa_" + tag + "); "
                                 "NGU_MAC(q1_" + std::to_string(og0) + ", " + tb0 + "." + comps[c0] + ", wa_" + tag + "); "
                                 "NGU_MAC(q2_" + std::to_string(og0) + ", " + ta1 + "." + comps[c0] + ", wa_" + tag + "); "
                                 "NGU_MAC(q3_" + std::to_string(og0) + ", " + tb1 + "." + comps[c0] + ", wa_" + tag + "); "
                                 "NGU_MAC(q0_" + std::to_string(og1) + ", " + ta0 + "." + comps[c1] + ", wb_" + tag + "); "
                                 "NGU_MAC(q1_" + std::to_string(og1) + ", " + tb0 + "." + comps[c1] + ", wb_" + tag + "); "
                                 "NGU_MAC(q2_" + std::to_string(og1) + ", " + ta1 + "." + comps[c1] + ", wb_" + tag + "); "
                                 "NGU_MAC(q3_" + std::to_string(og1) + ", " + tb1 + "." + comps[c1] + ", wb_" + tag + "); }\n";
                        }
                s += "                u_idx += " + std::to_string(6 * g_in0 * g_mid) + "u;\n"
                     "            }\n";
            } else {
                s += "            for (uint ky = 0u; ky < pc.k3; ++ky) {\n"
                     "                [[dont_unroll]] for (uint ig = 0u; ig < " + std::to_string(g_in0) + "u; ++ig) {\n"
                     "                    vec4 a0=smem[ig][py0+ky][px0+0u], a1=smem[ig][py0+ky][px0+1u], a2=smem[ig][py0+ky][px0+2u], a3=smem[ig][py0+ky][px0+3u];\n"
                     "                    vec4 b0s=smem[ig][py0+ky+1u][px0+0u], b1s=smem[ig][py0+ky+1u][px0+1u], b2s=smem[ig][py0+ky+1u][px0+2u], b3s=smem[ig][py0+ky+1u][px0+3u];\n"
                     "                    uint base = " + std::to_string(stages[0].w_uoff) + "u + ky * " + std::to_string(6 * g_in0 * g_mid) + "u + ig * " + std::to_string(2 * g_mid) + "u;\n";
                for (int kx = 0; kx < 3; ++kx)
                    for (int f = 0; f < 4 * g_mid; f += 2) {
                        const int c0 = f / g_mid, og0 = f % g_mid, c1 = (f + 1) / g_mid, og1 = (f + 1) % g_mid;
                        const int u_loc = kx * g_in0 * 2 * g_mid + f / 2;
                        const std::string tag = "fq_" + std::to_string(kx) + "_" + std::to_string(f / 2);
                        s += "                    { NGU_LOAD_PAIR(base + " + std::to_string(u_loc) + "u, uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                             "NGU_MAC(q0_" + std::to_string(og0) + ", a" + std::to_string(kx) + "." + comps[c0] + ", wa_" + tag + "); "
                             "NGU_MAC(q1_" + std::to_string(og0) + ", a" + std::to_string(kx + 1) + "." + comps[c0] + ", wa_" + tag + "); "
                             "NGU_MAC(q2_" + std::to_string(og0) + ", b" + std::to_string(kx) + "s." + comps[c0] + ", wa_" + tag + "); "
                             "NGU_MAC(q3_" + std::to_string(og0) + ", b" + std::to_string(kx + 1) + "s." + comps[c0] + ", wa_" + tag + "); "
                             "NGU_MAC(q0_" + std::to_string(og1) + ", a" + std::to_string(kx) + "." + comps[c1] + ", wb_" + tag + "); "
                             "NGU_MAC(q1_" + std::to_string(og1) + ", " + "a" + std::to_string(kx + 1) + "." + comps[c1] + ", wb_" + tag + "); "
                             "NGU_MAC(q2_" + std::to_string(og1) + ", b" + std::to_string(kx) + "s." + comps[c1] + ", wb_" + tag + "); "
                             "NGU_MAC(q3_" + std::to_string(og1) + ", b" + std::to_string(kx + 1) + "s." + comps[c1] + ", wb_" + tag + "); }\n";
                    }
                s += "                }\n            }\n";
            }
        }
        s += "        }\n";
        if (!scalar_in)
            s += "        barrier();\n";
        s += "        if (tid < 119u) {\n";
        for (int og = 0; og < g_mid; ++og) {
            const std::string ogs = std::to_string(og);
            s += "            smem[" + ogs + "][py0+0u][px0+0u] = mix(q0_" + ogs + ", q0_" + ogs + "*0.1, lessThan(q0_" + ogs + ", vec4(0.0)));\n"
                 "            smem[" + ogs + "][py0+0u][px0+1u] = mix(q1_" + ogs + ", q1_" + ogs + "*0.1, lessThan(q1_" + ogs + ", vec4(0.0)));\n"
                 "            smem[" + ogs + "][py0+1u][px0+0u] = mix(q2_" + ogs + ", q2_" + ogs + "*0.1, lessThan(q2_" + ogs + ", vec4(0.0)));\n"
                 "            smem[" + ogs + "][py0+1u][px0+1u] = mix(q3_" + ogs + ", q3_" + ogs + "*0.1, lessThan(q3_" + ogs + ", vec4(0.0)));\n";
        }
        s += "        }\n        barrier();\n    }\n";

        // Stage 1: 32x12 3py (ColumnGroupRows)
        const char *pfx = tail_out ? "h" : "p";
        if (!alias_bx) {
            s += "    if (uint(bx) + tx >= pc.width || uint(by) + yl >= pc.height) return;\n";
        }
        s += "    {\n";
        for (int p = 0; p < 3; ++p)
            for (int og = 0; og < g_last; ++og)
                s += "        vec4 " + std::string(pfx) + std::to_string(p) + ((g_last == 1) ? "" : ("_" + std::to_string(og))) + ";\n";
        s += emit_bias_init_n(g_last, std::to_string(stages[1].b_uoff) + "u", pfx, 3, "        ", 20) +
             "        int xl = max(0, 1 - bx), xh = min(33, int(pc.width) - bx);\n"
             "        int ylo = max(0, 1 - by), yhi = min(13, int(pc.height) - by);\n"
             "        uint ry0=uint(clamp(int(yl)+0,ylo,yhi)), ry1=uint(clamp(int(yl)+1,ylo,yhi)), "
             "ry2=uint(clamp(int(yl)+2,ylo,yhi)), ry3=uint(clamp(int(yl)+3,ylo,yhi)), ry4=uint(clamp(int(yl)+4,ylo,yhi));\n";
        if (full_unr_s1) {
            for (int kx = 0; kx < 3; ++kx) {
                s += "        uint rx_" + std::to_string(kx) + " = uint(clamp(int(tx) + " + std::to_string(kx) + ", xl, xh));\n";
                for (int ig = 0; ig < g_mid; ++ig) {
                    s += "        {\n";
                    for (int r = 0; r < 5; ++r)
                        s += "            vec4 t" + std::to_string(r) + " = smem[" + std::to_string(ig) + "][ry" + std::to_string(r) + "][rx_" + std::to_string(kx) + "];\n";
                    const std::uint32_t base = stages[1].w_uoff + static_cast<std::uint32_t>((kx * g_mid + ig) * (6 * g_last));
                    for (int ky = 0; ky < 3; ++ky)
                        for (int f = 0; f < 4 * g_last; f += 2) {
                            const int c0 = f / g_last, og0 = f % g_last, c1 = (f + 1) / g_last, og1 = (f + 1) % g_last;
                            const std::string s0 = (g_last == 1) ? "" : ("_" + std::to_string(og0));
                            const std::string s1 = (g_last == 1) ? "" : ("_" + std::to_string(og1));
                            const std::string tg = "u1_" + std::to_string(kx) + "_" + std::to_string(ig) + "_" + std::to_string(ky) + "_" + std::to_string(f / 2);
                            s += "            { NGU_LOAD_PAIR(" + std::to_string(base + ky * 2 * g_last + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); "
                                 "NGU_MAC(" + pfx + "0" + s0 + ", t" + std::to_string(ky) + "." + comps[c0] + ", wa_" + tg + "); "
                                 "NGU_MAC(" + pfx + "1" + s0 + ", t" + std::to_string(ky + 1) + "." + comps[c0] + ", wa_" + tg + "); "
                                 "NGU_MAC(" + pfx + "2" + s0 + ", t" + std::to_string(ky + 2) + "." + comps[c0] + ", wa_" + tg + "); "
                                 "NGU_MAC(" + pfx + "0" + s1 + ", t" + std::to_string(ky) + "." + comps[c1] + ", wb_" + tg + "); "
                                 "NGU_MAC(" + pfx + "1" + s1 + ", t" + std::to_string(ky + 1) + "." + comps[c1] + ", wb_" + tg + "); "
                                 "NGU_MAC(" + pfx + "2" + s1 + ", t" + std::to_string(ky + 2) + "." + comps[c1] + ", wb_" + tg + "); }\n";
                        }
                    s += "        }\n";
                }
            }
        } else {
            s += "        for (uint kx = 0u; kx < pc.k3; ++kx) {\n"
                 "            uint rx = uint(clamp(int(tx + kx), xl, xh));\n"
                 "            [[dont_unroll]] for (uint ig = 0u; ig < " + std::to_string(g_mid) + "u; ++ig) {\n"
                 "                vec4 t0=smem[ig][ry0][rx], t1=smem[ig][ry1][rx], t2=smem[ig][ry2][rx], t3=smem[ig][ry3][rx], t4=smem[ig][ry4][rx];\n"
                 "                uint base = " + std::to_string(stages[1].w_uoff) + "u + (kx * " + std::to_string(g_mid) + "u + ig) * " + std::to_string(6 * g_last) + "u;\n";
            for (int ky = 0; ky < 3; ++ky)
                for (int f = 0; f < 4 * g_last; f += 2) {
                    const int c0 = f / g_last, og0 = f % g_last, c1 = (f + 1) / g_last, og1 = (f + 1) % g_last;
                    const std::string s0 = (g_last == 1) ? "" : ("_" + std::to_string(og0));
                    const std::string s1 = (g_last == 1) ? "" : ("_" + std::to_string(og1));
                    const std::string tg = "r1_" + std::to_string(ky) + "_" + std::to_string(f / 2);
                    s += "                { NGU_LOAD_PAIR(base + " + std::to_string(ky * 2 * g_last + f / 2) + "u, u_" + tg + ", wa_" + tg + ", wb_" + tg + "); "
                         "NGU_MAC(" + pfx + "0" + s0 + ", t" + std::to_string(ky) + "." + comps[c0] + ", wa_" + tg + "); "
                         "NGU_MAC(" + pfx + "1" + s0 + ", t" + std::to_string(ky + 1) + "." + comps[c0] + ", wa_" + tg + "); "
                         "NGU_MAC(" + pfx + "2" + s0 + ", t" + std::to_string(ky + 2) + "." + comps[c0] + ", wa_" + tg + "); "
                         "NGU_MAC(" + pfx + "0" + s1 + ", t" + std::to_string(ky) + "." + comps[c1] + ", wb_" + tg + "); "
                         "NGU_MAC(" + pfx + "1" + s1 + ", t" + std::to_string(ky + 1) + "." + comps[c1] + ", wb_" + tg + "); "
                         "NGU_MAC(" + pfx + "2" + s1 + ", t" + std::to_string(ky + 2) + "." + comps[c1] + ", wb_" + tg + "); }\n";
                }
            s += "            }\n        }\n";
        }

        if (!tail_out) {
            s += "        uint x = uint(bx) + tx, y0 = uint(by) + yl;\n";
            for (int p = 0; p < 3; ++p) {
                if (p > 0) s += "        if (y0 + " + std::to_string(p) + "u < pc.height) {\n";
                s += "        { uint ob = (y0 + " + std::to_string(p) + "u) * pc.width + x;\n";
                for (int og = 0; og < g_last; ++og) {
                    const std::string v = "p" + std::to_string(p) + "_" + std::to_string(og);
                    s += "          d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(" + v + ", " + v + "*0.1, lessThan(" + v + ", vec4(0.0)));\n";
                }
                s += "        }\n";
                if (p > 0) s += "        }\n";
            }
        } else {
            if (!aa && alias_bx) {
                s += "        barrier();\n"
                     "        {\n"
                     "            uint gx = uint(clamp(bx + int(tx) - 2, 0, int(pc.width) - 1));\n"
                     "            uint gx_r = uint(clamp(bx + 32 + int(tx) - 2, 0, int(pc.width) - 1));\n"
                     "            for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                     "                uint row = uint(clamp(by + int(sy) - 2, 0, int(pc.height) - 1)) * pc.src_s;\n"
                     "                smem[0][sy][tx].x = src_in[row + gx];\n"
                     "                if (tx < 4u) smem[0][sy][32u + tx].x = src_in[row + gx_r];\n"
                     "            }\n        }\n        barrier();\n"
                     "        {\n"
                     "            float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                     "            float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
                     "            for (uint sy = ty; sy < 16u; sy += 4u) {\n"
                     "                float s0=smem[0][sy][tx+0u].x, s1=smem[0][sy][tx+1u].x, s2=smem[0][sy][tx+2u].x, s3=smem[0][sy][tx+3u].x, s4=smem[0][sy][tx+4u].x;\n"
                     "                smem[1][sy][tx].x = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                     "                smem[1][sy][tx].y = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                     "            }\n        }\n        barrier();\n"
                     "        if (uint(bx) + tx >= pc.width || uint(by) + yl >= pc.height) return;\n";
            }
            s += "        uint x = uint(bx) + tx, y0 = uint(by) + yl, ox = x * 2u, c0 = tx * 2u;\n";
            if (!aa) {
                s += "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
                     "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n";
            }
            for (int p = 0; p < 3; ++p) {
                const std::string hn = "h" + std::to_string(p);
                if (p > 0) s += "        if (y0 + " + std::to_string(p) + "u < pc.height) {\n";
                s += "        { uint ry = yl + " + std::to_string(p) + "u, oy = (y0 + " + std::to_string(p) + "u) * 2u;\n";
                if (aa) {
                    s += "          NGU_RAW u00 = src_u_t[ry+0u][tx+0u];\n"
                         "          float s00 = NGU_FROM_RAW(u00);\n"
                         "          float s10 = NGU_FROM_RAW(src_u_t[ry+0u][tx+1u]);\n"
                         "          float s01 = NGU_FROM_RAW(src_u_t[ry+1u][tx+0u]);\n"
                         "          float s11 = NGU_FROM_RAW(src_u_t[ry+1u][tx+1u]);\n"
                         "          precise float p10 = (s00 + s10) * 0.5 + " + hn + ".y;\n"
                         "          precise float p01 = (s00 + s01) * 0.5 + " + hn + ".x;\n"
                         "          precise float p11 = (((s00 + s10) + s01) + s11) * 0.25 + " + hn + ".z;\n"
                         "          d[(oy + 0u) * pc.dst_s + ox + 0u] = u00;\n"
                         "          d[(oy + 0u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p10);\n"
                         "          d[(oy + 1u) * pc.dst_s + ox + 0u] = NGU_TO_RAW(p01);\n"
                         "          d[(oy + 1u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p11);\n";
                } else if (alias_bx) {
                    s += "          float r0a=smem[1][ry+0u][tx].x, r1a=smem[1][ry+1u][tx].x, r2a=smem[1][ry+2u][tx].x, r3a=smem[1][ry+3u][tx].x, r4a=smem[1][ry+4u][tx].x;\n"
                         "          float r0b=smem[1][ry+0u][tx].y, r1b=smem[1][ry+1u][tx].y, r2b=smem[1][ry+2u][tx].y, r3b=smem[1][ry+3u][tx].y, r4b=smem[1][ry+4u][tx].y;\n"
                         "          precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + " + hn + ".x;\n"
                         "          precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + " + hn + ".y;\n"
                         "          precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + " + hn + ".z;\n"
                         "          precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + " + hn + ".w;\n"
                         "          d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
                         "          d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
                         "          d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
                         "          d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
                } else {
                    s += "          float r0a=bx_t[ry+0u][c0], r1a=bx_t[ry+1u][c0], r2a=bx_t[ry+2u][c0], r3a=bx_t[ry+3u][c0], r4a=bx_t[ry+4u][c0];\n"
                         "          float r0b=bx_t[ry+0u][c0+1u], r1b=bx_t[ry+1u][c0+1u], r2b=bx_t[ry+2u][c0+1u], r3b=bx_t[ry+3u][c0+1u], r4b=bx_t[ry+4u][c0+1u];\n"
                         "          precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + " + hn + ".x;\n"
                         "          precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + " + hn + ".y;\n"
                         "          precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + " + hn + ".z;\n"
                         "          precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + " + hn + ".w;\n"
                         "          d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
                         "          d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
                         "          d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
                         "          d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
                }
                s += "        }\n";
                if (p > 0) s += "        }\n";
            }
        }
        s += "    }\n}\n";
        return s;
    }

    // Compact 32-KB fallback for Low FP32 on strict 32-KB devices (< 41.8 KB smem)
    std::string gen_low_32kb(std::span<const StageDesc> stages, std::uint32_t total_uvec4, bool aa) const {
        const char *comps[4] = {"x", "y", "z", "w"};
        std::string s = shader_preamble;
        s += "layout(local_size_x=16, local_size_y=16, local_size_z=1) in;\n";
        s += aa ? "layout(set=0, binding=0, std430) readonly buffer Source { NGU_RAW s[]; };\n"
                  "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
                : "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
                  "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
        s += "NGU_UBO_DECL(" + std::to_string(total_uvec4 * 2) + ", " + std::to_string(total_uvec4) + ")\n"
             "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
             "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n";
        s += aa ? "shared NGU_RAW src_t[22][22];\n" : "shared float src_t[22][22];\n";
        s += "shared vec4 tile[2][20][20];\n"
             "void main() {\n"
             "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 16u + tx;\n"
             "    int bx = int(gl_WorkGroupID.x) * 14, by = int(gl_WorkGroupID.y) * 14;\n"
             "    for (uint idx = tid; idx < 484u; idx += 256u) {\n"
             "        uint ly = idx / 22u, lx = idx - ly * 22u;\n"
             "        src_t[ly][lx] = s[uint(clamp(by + int(ly) - 4, 0, int(pc.height) - 1)) * pc.src_s + uint(clamp(bx + int(lx) - 4, 0, int(pc.width) - 1))];\n"
             "    }\n    barrier();\n"
             "    uint py0 = ty * 2u, px0 = tx * 2u;\n";
        const std::string rd_pre = aa ? "NGU_FROM_RAW(src_t" : "src_t";
        const std::string rd_post = aa ? ")" : "";
        s += "    if (tx < 10u && ty < 10u) {\n"
             "        vec4 q0_0, q0_1, q1_0, q1_1, q2_0, q2_1, q3_0, q3_1;\n" +
             emit_bias_init_n(2, std::to_string(stages[0].b_uoff) + "u", "q", 4, "        ", 0);
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                const std::string i2 = std::to_string(stages[0].w_uoff + ky * 3 + kx) + "u";
                const std::string tag = "c0_" + std::to_string(ky * 3 + kx);
                s += "        { NGU_LOAD_PAIR(" + i2 + ", uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                     "float a0=" + rd_pre + "[py0+" + std::to_string(ky) + "u][px0+" + std::to_string(kx) + "u]" + rd_post + ", "
                     "a1=" + rd_pre + "[py0+" + std::to_string(ky) + "u][px0+" + std::to_string(kx + 1) + "u]" + rd_post + ", "
                     "b0s=" + rd_pre + "[py0+" + std::to_string(ky + 1) + "u][px0+" + std::to_string(kx) + "u]" + rd_post + ", "
                     "b1s=" + rd_pre + "[py0+" + std::to_string(ky + 1) + "u][px0+" + std::to_string(kx + 1) + "u]" + rd_post + "; "
                     "NGU_MAC(q0_0, a0, wa_" + tag + "); NGU_MAC(q1_0, a1, wa_" + tag + "); NGU_MAC(q2_0, b0s, wa_" + tag + "); NGU_MAC(q3_0, b1s, wa_" + tag + "); "
                     "NGU_MAC(q0_1, a0, wb_" + tag + "); NGU_MAC(q1_1, a1, wb_" + tag + "); NGU_MAC(q2_1, b0s, wb_" + tag + "); NGU_MAC(q3_1, b1s, wb_" + tag + "); }\n";
            }
        for (int og = 0; og < 2; ++og) {
            const std::string ogs = std::to_string(og);
            s += "        tile[" + ogs + "][py0+0u][px0+0u]=mix(q0_" + ogs + ",q0_" + ogs + "*0.1,lessThan(q0_" + ogs + ",vec4(0.0)));\n"
                 "        tile[" + ogs + "][py0+0u][px0+1u]=mix(q1_" + ogs + ",q1_" + ogs + "*0.1,lessThan(q1_" + ogs + ",vec4(0.0)));\n"
                 "        tile[" + ogs + "][py0+1u][px0+0u]=mix(q2_" + ogs + ",q2_" + ogs + "*0.1,lessThan(q2_" + ogs + ",vec4(0.0)));\n"
                 "        tile[" + ogs + "][py0+1u][px0+1u]=mix(q3_" + ogs + ",q3_" + ogs + "*0.1,lessThan(q3_" + ogs + ",vec4(0.0)));\n";
        }
        s += "    }\n    barrier();\n";
        const std::uint32_t w_stride = stages[2].w_uoff - stages[1].w_uoff;
        const std::uint32_t b_stride = stages[2].b_uoff - stages[1].b_uoff;
        s += "    [[dont_unroll]] for (uint stg = 0u; stg < 2u; ++stg) {\n"
             "        uint q_dim = 9u - stg;\n"
             "        int pad = 3 - int(stg), max_in = 19 - int(stg) * 2;\n"
             "        vec4 q0_0=vec4(0.0),q0_1=vec4(0.0),q1_0=vec4(0.0),q1_1=vec4(0.0),q2_0=vec4(0.0),q2_1=vec4(0.0),q3_0=vec4(0.0),q3_1=vec4(0.0);\n"
             "        if (tx < q_dim && ty < q_dim) {\n"
             "            int xl=max(0,pad-bx),xh=min(max_in,int(pc.width)-1+pad-bx),ylo=max(0,pad-by),yhi=min(max_in,int(pc.height)-1+pad-by);\n" +
             emit_bias_init_n(2, std::to_string(stages[1].b_uoff) + "u + stg * " + std::to_string(b_stride) + "u", "q", 4, "            ", 1) +
             "            uint u_idx = " + std::to_string(stages[1].w_uoff) + "u + stg * " + std::to_string(w_stride) + "u;\n"
             "            for (uint ky=0u; ky<pc.k3; ++ky) for (uint kx=0u; kx<pc.k3; ++kx) for (uint ig=0u; ig<2u; ++ig) {\n"
             "                uint ry0=uint(clamp(int(py0+ky)+0,ylo,yhi)),ry1=uint(clamp(int(py0+ky)+1,ylo,yhi));\n"
             "                uint rx0=uint(clamp(int(px0+kx)+0,xl,xh)),rx1=uint(clamp(int(px0+kx)+1,xl,xh));\n"
             "                vec4 a0=tile[ig][ry0][rx0],b0s=tile[ig][ry0][rx1],a1=tile[ig][ry1][rx0],b1s=tile[ig][ry1][rx1];\n";
        for (int c = 0; c < 4; ++c) {
            const std::string tag = "cl_" + std::to_string(c);
            s += "                { NGU_LOAD_PAIR(u_idx + " + std::to_string(c) + "u, uw_" + tag + ", wa_" + tag + ", wb_" + tag + "); "
                 "NGU_MAC(q0_0,a0." + comps[c] + ",wa_" + tag + "); NGU_MAC(q1_0,b0s." + comps[c] + ",wa_" + tag + "); NGU_MAC(q2_0,a1." + comps[c] + ",wa_" + tag + "); NGU_MAC(q3_0,b1s." + comps[c] + ",wa_" + tag + "); "
                 "NGU_MAC(q0_1,a0." + comps[c] + ",wb_" + tag + "); NGU_MAC(q1_1,b0s." + comps[c] + ",wb_" + tag + "); NGU_MAC(q2_1,a1." + comps[c] + ",wb_" + tag + "); NGU_MAC(q3_1,b1s." + comps[c] + ",wb_" + tag + "); }\n";
        }
        s += "                u_idx += 4u;\n            }\n        }\n        barrier();\n"
             "        if (tx < q_dim && ty < q_dim) {\n";
        for (int og = 0; og < 2; ++og) {
            const std::string ogs = std::to_string(og);
            s += "            tile[" + ogs + "][py0+0u][px0+0u]=mix(q0_" + ogs + ",q0_" + ogs + "*0.1,lessThan(q0_" + ogs + ",vec4(0.0)));\n"
                 "            tile[" + ogs + "][py0+0u][px0+1u]=mix(q1_" + ogs + ",q1_" + ogs + "*0.1,lessThan(q1_" + ogs + ",vec4(0.0)));\n"
                 "            tile[" + ogs + "][py0+1u][px0+0u]=mix(q2_" + ogs + ",q2_" + ogs + "*0.1,lessThan(q2_" + ogs + ",vec4(0.0)));\n"
                 "            tile[" + ogs + "][py0+1u][px0+1u]=mix(q3_" + ogs + ",q3_" + ogs + "*0.1,lessThan(q3_" + ogs + ",vec4(0.0)));\n";
        }
        s += "        }\n        barrier();\n    }\n"
             "    vec4 hsum = vec4(0.0);\n"
             "    if (tx < 14u && ty < 14u) {\n"
             "        int xl=max(0,1-bx),xh=min(15,int(pc.width)-bx),ylo=max(0,1-by),yhi=min(15,int(pc.height)-by);\n" +
             emit_bias_init_n(1, std::to_string(stages[3].b_uoff) + "u", "hsum_tmp", 1, "        ", 9) +
             "        hsum = hsum_tmp0;\n"
             "        uint u_idx = " + std::to_string(stages[3].w_uoff) + "u;\n"
             "        for (uint ky=0u; ky<pc.k3; ++ky) for (uint kx=0u; kx<pc.k3; ++kx) for (uint ig=0u; ig<2u; ++ig) {\n"
             "            vec4 v = tile[ig][uint(clamp(int(ty+ky),ylo,yhi))][uint(clamp(int(tx+kx),xl,xh))];\n"
             "            { NGU_LOAD_PAIR(u_idx+0u, u0, wa0, wb0); NGU_MAC(hsum,v.x,wa0); NGU_MAC(hsum,v.y,wb0); }\n"
             "            { NGU_LOAD_PAIR(u_idx+1u, u1, wa1, wb1); NGU_MAC(hsum,v.z,wa1); NGU_MAC(hsum,v.w,wb1); }\n"
             "            u_idx += 2u;\n        }\n    }\n";
        if (!aa) {
            s += "    barrier();\n    if (tid < 252u) {\n"
                 "        float ca=uintBitsToFloat(0x3f5f3333u),cb=uintBitsToFloat(0x3e766666u),cc=uintBitsToFloat(0xbdaccccdu),cd=uintBitsToFloat(0xbce66666u);\n"
                 "        uint r=tid/14u, lx=tid-r*14u;\n"
                 "        float s0=src_t[r+2u][lx+2u],s1=src_t[r+2u][lx+3u],s2=src_t[r+2u][lx+4u],s3=src_t[r+2u][lx+5u],s4=src_t[r+2u][lx+6u];\n"
                 "        tile[0][r][lx].x=(((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
                 "        tile[0][r][lx].y=(((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
                 "    }\n    barrier();\n";
        }
        s += "    if (tx>=14u||ty>=14u) return;\n    uint x=uint(bx)+tx,y=uint(by)+ty;\n    if (x>=pc.width||y>=pc.height) return;\n    uint ox=x*2u,oy=y*2u;\n";
        if (aa) {
            s += "    NGU_RAW u00=src_t[ty+4u][tx+4u]; float s00=NGU_FROM_RAW(u00),s10=NGU_FROM_RAW(src_t[ty+4u][tx+5u]),s01=NGU_FROM_RAW(src_t[ty+5u][tx+4u]),s11=NGU_FROM_RAW(src_t[ty+5u][tx+5u]);\n"
                 "    precise float p10=(s00+s10)*0.5+hsum.y, p01=(s00+s01)*0.5+hsum.x, p11=(((s00+s10)+s01)+s11)*0.25+hsum.z;\n"
                 "    d[(oy+0u)*pc.dst_s+ox+0u]=u00; d[(oy+0u)*pc.dst_s+ox+1u]=NGU_TO_RAW(p10); d[(oy+1u)*pc.dst_s+ox+0u]=NGU_TO_RAW(p01); d[(oy+1u)*pc.dst_s+ox+1u]=NGU_TO_RAW(p11);\n";
        } else {
            s += "    float ca=uintBitsToFloat(0x3f5f3333u),cb=uintBitsToFloat(0x3e766666u),cc=uintBitsToFloat(0xbdaccccdu),cd=uintBitsToFloat(0xbce66666u);\n"
                 "    float r0a=tile[0][ty+0u][tx].x,r1a=tile[0][ty+1u][tx].x,r2a=tile[0][ty+2u][tx].x,r3a=tile[0][ty+3u][tx].x,r4a=tile[0][ty+4u][tx].x;\n"
                 "    float r0b=tile[0][ty+0u][tx].y,r1b=tile[0][ty+1u][tx].y,r2b=tile[0][ty+2u][tx].y,r3b=tile[0][ty+3u][tx].y,r4b=tile[0][ty+4u][tx].y;\n"
                 "    precise float p00=((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a))+hsum.x;\n"
                 "    precise float p10=((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b))+hsum.y;\n"
                 "    precise float p01=((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a))+hsum.z;\n"
                 "    precise float p11=((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b))+hsum.w;\n"
                 "    d[(oy+0u)*pc.dst_s+ox+0u]=p00; d[(oy+0u)*pc.dst_s+ox+1u]=p10; d[(oy+1u)*pc.dst_s+ox+0u]=p01; d[(oy+1u)*pc.dst_s+ox+1u]=p11;\n";
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
        min_smem = half ? 20880 : 14736;
        max_ubo = half ? 3104 : 6208;
    } else if (b.size() == 5 && is_conv3(b[0], 1, 12) && is_conv3(b[1], 12, 8) &&
               is_conv3(b[2], 8, 8) && is_conv3(b[3], 8, 8) && is_conv3(b[4], 8, 8) &&
               h.input_channels() == 8) {
        topo = Topology::Medium;
        min_smem = half ? 21504 : 29184;
        max_ubo = half ? 2368 : 4736;
    } else if (b.size() == 6 && is_conv3(b[0], 1, 12) && is_conv3(b[1], 12, 12) &&
               is_conv3(b[2], 12, 12) && is_conv3(b[3], 12, 12) && is_conv3(b[4], 12, 12) &&
               is_conv3(b[5], 12, 12) && h.input_channels() == 12) {
        topo = Topology::High;
        min_smem = half ? 26624 : 29696;
        max_ubo = half ? 5312 : 10624;
    } else if (b.size() == 8 && is_conv3(b[0], 1, 16) && is_conv3(b[1], 16, 16) &&
               is_conv3(b[2], 16, 16) && is_conv3(b[3], 16, 16) && is_conv3(b[4], 16, 16) &&
               is_conv3(b[5], 16, 16) && is_conv3(b[6], 16, 16) && is_conv3(b[7], 16, 16) &&
               h.input_channels() == 16) {
        topo = Topology::VeryHigh;
        min_smem = half ? 28288 : 30720;
        max_ubo = half ? 9600 : 9920;
    }
    if (topo == Topology::Fallback)
        return Topology::Fallback;
    if (limits.maxComputeSharedMemorySize < min_smem || limits.maxUniformBufferRange < max_ubo ||
        limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 32 ||
        limits.maxComputeWorkGroupSize[1] < 16)
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
    VkDeviceSize workspace_size = 0;

    mutable std::once_flag ws_once;
    mutable Buffer ws_buf{nullptr, BufferDeleter{nullptr}};
    mutable VSVulkanBufferInfo ws_info{};

    VkPipeline fallback_pipeline = VK_NULL_HANDLE;
    std::vector<LayerInfo> layers;
    std::array<VkDescriptorBufferInfo, 5> fallback_slices{};

    ~Impl() {
        if (gpu && pool) {
            gpu->freeGPUExecPool(pool);
            pool = nullptr;
        }
        ws_buf.reset();
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

    VkPipeline compile_pipeline(const char *source) {
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
            vk->vkCreateComputePipelines(handles.device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipe);
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

        std::vector<float> ubo_data;
        const VkDeviceSize ubo_align = std::max<VkDeviceSize>(16, limits.minUniformBufferOffsetAlignment);
        const VkDeviceSize elem_bytes = half_compute ? 2 : 4;

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
        if (topo == Topology::Low) {
            std::vector<float> blk;
            std::array<StageDesc, 4> st{{
                make_body(blk, 0, false),
                make_body(blk, 1, false),
                make_body(blk, 2, false),
                make_head(blk, false),
            }};
            VkDeviceSize off = 0, sz = 0;
            const auto total_uvec4 = append_ubo(blk, off, sz);
            if (half_compute || limits.maxComputeSharedMemorySize >= 41760) {
                VkPipeline pipe = compile_pipeline(shader.gen_low_26x26(st, total_uvec4, aa).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 26, 26, -1, -2});
            } else {
                VkPipeline pipe = compile_pipeline(shader.gen_low_32kb(st, total_uvec4, aa).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 14, 14, -1, -2});
            }
        } else if (topo == Topology::Medium) {
            // 3-Pass Pure Warp-Stripe (2 + 2 + 2): Spatial(0)->Col(1), Spatial(2)->Col(3), Spatial(4)->Col(head).
            slice_channels = 8;
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 0, false), make_body(blk, 1, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const bool use_3py = half_compute || (limits.maxComputeSharedMemorySize >= 42912);
                const int py = use_3py ? 3 : 2;
                const auto th = static_cast<std::uint32_t>(8 * py - 2);
                VkPipeline pipe = compile_pipeline(shader.gen_2L_spa_col_stripe(st, nu, aa, 8, py, true).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 30, th, -1, 0});
            }
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 2, false), make_body(blk, 3, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const int py = half_compute ? 4 : 3;
                const auto th = static_cast<std::uint32_t>(8 * py - 2);
                VkPipeline pipe = compile_pipeline(shader.gen_2L_spa_col_stripe(st, nu, aa, 8, py, true).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 30, th, 0, 1});
            }
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 4, false), make_head(blk, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const bool use_3py = half_compute || (limits.maxComputeSharedMemorySize >= 32032);
                const int py = use_3py ? 3 : 2;
                const auto th = static_cast<std::uint32_t>(8 * py - 2);
                VkPipeline pipe = compile_pipeline(shader.gen_2L_spa_col_stripe(st, nu, aa, 8, py, true).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 30, th, 1, -2});
            }
        } else if (topo == Topology::High) {
            // Schedule: Spatial(0), Col(1..4), Spatial(5), Col(head).
            slice_channels = 12;
            // Pass 1: b0 (1->12 Spatial) + b1 (12->12 Col) Pure Warp-Stripe
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 0, false), make_body(blk, 1, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const int py = half_compute ? 3 : 2;
                const auto th = static_cast<std::uint32_t>(8 * py - 2);
                VkPipeline pipe = compile_pipeline(shader.gen_2L_spa_col_stripe(st, nu, aa, 8, py, true).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 30, th, -1, 0});
            }
            int cur_slice = 0;
            // Middle layers b2, b3, b4:
            // - FP16: fuse b2+b3 via 30x30 (8w,4py) FULL (cuts 100 MB DRAM!) + b4 via 1L 32x12 (4w,3py) FULL
            // - FP32: 3 x 1L 32x12 (6w,2py) FULL (272 us/layer, 24 warps/SM, smem=22.8 KB)
            if (half_compute && limits.maxComputeSharedMemorySize >= 27744) {
                {
                    std::vector<float> blk;
                    std::array<StageDesc, 2> st{{make_body(blk, 2, true), make_body(blk, 3, true)}};
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    VkPipeline pipe = compile_pipeline(shader.gen_2L_col_stripe(st, nu, 8, 4, true).c_str());
                    fast_passes.push_back(FastPass{pipe, off, sz, 30, 30, 0, 1});
                }
                {
                    std::vector<float> blk;
                    StageDesc st = make_body(blk, 4, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    VkPipeline pipe = compile_pipeline(shader.gen_1L_stripe(st, nu, 4, 3).c_str());
                    fast_passes.push_back(FastPass{pipe, off, sz, 32, 12, 1, 0});
                }
                cur_slice = 0;
            } else {
                const int wg_y = half_compute ? 4 : 6;
                const int py = half_compute ? 3 : 2;
                VkPipeline p_1l = VK_NULL_HANDLE;
                for (std::size_t i = 2; i <= 4; ++i) {
                    std::vector<float> blk;
                    StageDesc st = make_body(blk, i, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    if (p_1l == VK_NULL_HANDLE)
                        p_1l = compile_pipeline(shader.gen_1L_stripe(st, nu, wg_y, py).c_str());
                    const int next_slice = cur_slice ^ 1;
                    fast_passes.push_back(FastPass{p_1l, off, sz, 32, 12, cur_slice, next_slice});
                    cur_slice = next_slice;
                }
            }
            // Final:
            // - FP16: Fused Tail b5 (Spatial) + Head (Col) + Combine on 30x22 (330 us, 23.1 KB smem, zero intermediate DRAM!)
            // - FP32: Split b5 (1L 6w,2py Spatial FULL: 382 us) + Head (32x12 4w,3py: 199 us, 29.5 KB smem < 32 KB)
            if (half_compute) {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 5, false), make_head(blk, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                VkPipeline pipe = compile_pipeline(shader.gen_2L_spa_col_stripe(st, nu, aa, 8, 3, true).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 30, 22, cur_slice, -2});
            } else {
                const int next_slice = cur_slice ^ 1;
                {
                    std::vector<float> blk;
                    StageDesc st = make_body(blk, 5, false);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    VkPipeline pipe = compile_pipeline(shader.gen_1L_stripe(st, nu, 6, 2, true).c_str());
                    fast_passes.push_back(FastPass{pipe, off, sz, 32, 12, cur_slice, next_slice});
                }
                {
                    std::vector<float> blk;
                    StageDesc st = make_head(blk, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    VkPipeline pipe = compile_pipeline(shader.gen_head_32x12(st, nu, aa, false).c_str());
                    fast_passes.push_back(FastPass{pipe, off, sz, 32, 12, next_slice, -2});
                }
            }
        } else {
            // VeryHigh (C=16): Spatial(0), Col(1..7), head(Col if aa else Spatial).
            slice_channels = 16;
            // Pass 1: b0 (1->16 Spatial) + b1 (16->16 Col) Pure Warp-Stripe
            {
                std::vector<float> blk;
                std::array<StageDesc, 2> st{{make_body(blk, 0, false), make_body(blk, 1, true)}};
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                if (half_compute || limits.maxComputeSharedMemorySize >= 37408) {
                    const int py = half_compute ? 3 : 2;
                    const auto th = static_cast<std::uint32_t>(8 * py - 2);
                    VkPipeline pipe = compile_pipeline(shader.gen_2L_spa_col_stripe(st, nu, aa, 8, py, true).c_str());
                    fast_passes.push_back(FastPass{pipe, off, sz, 30, th, -1, 0});
                } else {
                    VkPipeline pipe = compile_pipeline(shader.gen_2L_32x12(st, nu, aa, false).c_str());
                    fast_passes.push_back(FastPass{pipe, off, sz, 32, 12, -1, 0});
                }
            }
            int cur_slice = 0;
            // Body b2..b7 (6 layers):
            // - FP16: 1L 32x16 (8w, 2py FULL: regs=36, smem=19.6 KB -> 40 warps/SM, 354 us/layer, -0.35 ms & same W!)
            // - FP32: 1L 32x12 (6w, 2py FULL: regs=81, smem=30.5 KB -> 18 warps/SM, 544 us/layer)
            {
                const int wg_y = half_compute ? 8 : 6;
                const int py = 2;
                const auto th = static_cast<std::uint32_t>(wg_y * py);
                VkPipeline p_1l = VK_NULL_HANDLE;
                for (std::size_t i = 2; i < model.body.size(); ++i) {
                    std::vector<float> blk;
                    StageDesc st = make_body(blk, i, true);
                    VkDeviceSize off = 0, sz = 0;
                    const auto nu = append_ubo(blk, off, sz);
                    if (p_1l == VK_NULL_HANDLE)
                        p_1l = compile_pipeline(shader.gen_1L_stripe(st, nu, wg_y, py).c_str());
                    const int next_slice = cur_slice ^ 1;
                    fast_passes.push_back(FastPass{p_1l, off, sz, 32, th, cur_slice, next_slice});
                    cur_slice = next_slice;
                }
            }
            // Final Head + Combine: unified 32x12 3py immediate-constant kernel
            {
                std::vector<float> blk;
                StageDesc st = make_head(blk, aa);
                VkDeviceSize off = 0, sz = 0;
                const auto nu = append_ubo(blk, off, sz);
                const bool alias_bx = !half_compute && (limits.maxComputeSharedMemorySize < 37120);
                VkPipeline pipe = compile_pipeline(shader.gen_head_32x12(st, nu, aa, alias_bx).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 32, 12, cur_slice, -2});
            }
        }

        if (slice_channels > 0) {
            const auto g_count = static_cast<std::uint32_t>(slice_channels / 4);
            if (static_cast<std::uint64_t>(width) * height >
                std::numeric_limits<std::uint32_t>::max() / g_count)
                throw std::invalid_argument("Vulkan feature plane stride overflow");
            int max_slice_idx = -1;
            for (const auto &p : fast_passes)
                max_slice_idx = std::max(max_slice_idx, std::max(p.src_slice, p.dst_slice));
            VkDeviceSize cursor = 0;
            const VkDeviceSize slice_bytes =
                static_cast<VkDeviceSize>(width) * height * slice_channels * (half_compute ? 2 : 4);
            if (max_slice_idx >= 0) {
                const VkDeviceSize s0 = add_slice(width, height, slice_channels, cursor);
                fast_slices[0] = {VK_NULL_HANDLE, s0, slice_bytes};
            }
            if (max_slice_idx >= 1) {
                const VkDeviceSize s1 = add_slice(width, height, slice_channels, cursor);
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
        memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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

    VkBuffer ensure_workspace() const {
        if (workspace_size == 0)
            return VK_NULL_HANDLE;
        std::call_once(ws_once, [&] {
            ws_buf = buffer(workspace_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, ws_info);
        });
        return ws_info.buffer;
    }

    void process(const VSFrame *source, VSFrame *output) const {
        std::uint32_t source_stride = 0, output_stride = 0;
        auto input = plane(source, width, height, source_stride);
        auto destination = plane(output, width * 2, height * 2, output_stride);
        const VkBuffer workspace_buffer = ensure_workspace();

        char error[1024]{};
        Exec exec{gpu, gpu->gpuExecAcquire(pool, error, sizeof(error))};
        if (!exec.context)
            throw std::runtime_error(std::string("GPU frame acquire: ") + error);
        gpu->gpuExecReadsFrame(exec.context, source);
        gpu->gpuExecWritesPlane(exec.context, output, 0);
        const VkCommandBuffer cmd = gpu->gpuExecCommandBuffer(exec.context);

        if (workspace_size > 0) {
            barrier(cmd, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        const auto w = static_cast<std::uint32_t>(width);
        const auto h = static_cast<std::uint32_t>(height);

        if (fast_path) {
            auto local = fast_slices;
            for (auto &slice : local)
                slice.buffer = workspace_buffer;
            const FastPush push{w, h, w * h, source_stride, output_stride, w * 2, h * 2, 3u};
            push_constants(cmd, push);
            VkPipeline bound = VK_NULL_HANDLE;
            for (std::size_t i = 0; i < fast_passes.size(); ++i) {
                const FastPass &pass = fast_passes[i];
                if (pass.pipeline != bound) {
                    vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
                    bound = pass.pipeline;
                }
                const auto &src = (pass.src_slice < 0) ? input : local[pass.src_slice];
                const auto &dst = (pass.dst_slice < 0) ? destination : local[pass.dst_slice];
                const VkDescriptorBufferInfo ubo{parameter_info.buffer, pass.ubo_offset, pass.ubo_size};
                const std::uint32_t gx = (w + pass.tile_w - 1) / pass.tile_w;
                const std::uint32_t gy = (h + pass.tile_h - 1) / pass.tile_h;
                push_bindings(cmd, src, dst, ubo, input);
                vk->vkCmdDispatch(cmd, gx, gy, 1);
                if (i + 1 < fast_passes.size())
                    barrier(cmd);
            }
        } else {
            auto local = fallback_slices;
            for (auto &slice : local)
                slice.buffer = workspace_buffer;
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
