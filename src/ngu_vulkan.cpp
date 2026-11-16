#define VK_NO_PROTOTYPES
#include <VSVulkan4.h>
#include "ngu_vulkan.hpp"
#include "ngu_half.hpp"
#include "ngu_half_shader.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
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
float parameter_at(uint i) { return p[i]; }
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
        // Source is the packed (w+2,h+2,4) residual, auxiliary is baseline Y.
        uint lane=3-(x&1)-2*(y&1);
        precise float v=a[y*pc.aux_stride+x]+s[((y+1)/2)*pc.src_stride+((x+1)/2)*4+lane];
        d[out_index+x]=v;
    } else {
        // Source is the original pitched plane; auxiliary is the AA XYZ head.
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
        // submit consumes the recording on BOTH success and failure.
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

void append_floats(std::vector<float> &dst, std::span<const float> src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

std::vector<float> pack_1toC(const Layer &layer) {
    const int co = layer.output_channels();
    std::vector<float> out(static_cast<std::size_t>(9) * co, 0.0f);
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx)
            for (int o = 0; o < co; ++o)
                out[(ky * 3 + kx) * co + o] = layer.weight_at(o * 9 + ky * 3 + kx);
    return out;
}

std::vector<float> pack_nc4(const Layer &layer) {
    const int ci = layer.input_channels();
    const int co = layer.output_channels();
    const int igs = (ci + 3) / 4;
    const int ogs = (co + 3) / 4;
    std::vector<float> out(static_cast<std::size_t>(9) * igs * 4 * ogs * 4, 0.0f);
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx)
            for (int c = 0; c < ci; ++c)
                for (int o = 0; o < co; ++o)
                    out[(((ky * 3 + kx) * igs + (c >> 2)) * 4 + (c & 3)) * ogs * 4 + (o >> 2) * 4 +
                        (o & 3)] = layer.weight_at(((o * ci + c) * 3 + ky) * 3 + kx);
    return out;
}

std::vector<float> pack_aa_head_3x3(const Layer &head) {
    const int ci = head.input_channels();
    const int igs = (ci + 3) / 4;
    std::vector<float> out(static_cast<std::size_t>(9) * igs * 16, 0.0f);
    for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx)
            for (int ic = 0; ic < ci; ++ic)
                for (int oc = 0; oc < 4; ++oc)
                    out[((ky * 3 + kx) * igs * 4 + (ic >> 2) * 4 + (ic & 3)) * 4 + oc] =
                        head.weight_at(((oc * ci + ic) * 3 + ky) * 3 + kx);
    return out;
}

std::vector<float> pack_k4_head_3x3(const Layer &head) {
    const int ci = head.input_channels();
    const int igs = (ci + 3) / 4;
    std::vector<float> out(static_cast<std::size_t>(9) * igs * 16, 0.0f);
    for (int sub = 0; sub < 4; ++sub) {
        const int sx = sub & 1, sy = sub >> 1;
        const int lane = 3 - sx - 2 * sy;
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int kx = (sx - dx) + 1;
                const int ky = (sy - dy) + 1;
                for (int ic = 0; ic < ci; ++ic)
                    out[((ky * 3 + kx) * igs * 16) + (ic >> 2) * 16 + (ic & 3) * 4 + sub] =
                        head.weight_at(ic * 16 + (lane / 2 + dy * 2) * 4 + lane % 2 + dx * 2);
            }
    }
    return out;
}

std::vector<float> pack_bias(const Layer &layer) {
    std::vector<float> out(layer.output_channels());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = layer.bias_at(i);
    return out;
}

// One generator context per filter construction. FP16 uses explicit half FMA
// updates; FP32 retains its original contraction and traversal strategy.
struct ShaderGenerator {
    bool half;

std::string emit_nc4_loop_1px(const char *wname, const char *tile_name, int igs, int ogs,
                              const std::string &ry_expr, const std::string &rx_expr, bool clamp_coords) const {
    std::string r = "    uint wi = 0u;\n    for (int ky = 0; ky < 3; ++ky) {\n";
    if (clamp_coords)
        r += "        uint ry = " + ry_expr + ";\n";
    r += "        for (int kx = 0; kx < 3; ++kx) {\n";
    if (clamp_coords)
        r += "            uint rx = " + rx_expr + ";\n";
    const std::string y_s = clamp_coords ? "ry" : ry_expr;
    const std::string x_s = clamp_coords ? "rx" : rx_expr;
    r += "            for (uint ig = 0u; ig < " + std::to_string(igs) + "u; ++ig) {\n";
    r += "                vec4 v = " + std::string(tile_name) + "[ig][" + y_s + "][" + x_s + "];\n";
    const char *comps[4] = {"x", "y", "z", "w"};
    for (int og = 0; og < ogs; ++og) {
        for (int c = 0; c < 4; ++c) {
            if (c == 0 || half)
                r += "                o" + std::to_string(og) + " += ";
            else
                r += " + ";
            r += std::string("v.") + comps[c] + "*" + std::string(wname) +
                 "[wi+" + std::to_string(c * ogs + og) + "u]";
            if (half || c == 3)
                r += ";\n";
        }
    }
    r += "                wi += " + std::to_string(4 * ogs) + "u;\n";
    r += "            }\n        }\n    }\n";
    return r;
}

std::string emit_head_compute_and_store(int C, bool aa, const std::string &src_tile_name, int ox_shift,
                                        int oy_shift, bool clamp_tile) const {
    const int G = C / 4;
    std::string r = "    vec4 hsum = bh[0]; uint wi = 0u;\n";
    r += "    for (int ky = 0; ky < 3; ++ky) {\n";
    if (clamp_tile)
        r += "        uint ry = uint(clamp(int(ty) + ky, yl_h, yh_h));\n";
    r += "        for (int kx = 0; kx < 3; ++kx) {\n";
    if (clamp_tile)
        r += "            uint rx = uint(clamp(int(tx) + kx, xl_h, xh_h));\n";
    const std::string ry_s = clamp_tile ? "ry" : "ty + uint(ky)";
    const std::string rx_s = clamp_tile ? "rx" : "tx + uint(kx)";
    r += "            for (uint ig = 0u; ig < " + std::to_string(G) + "u; ++ig) {\n"
         "                vec4 v = tile[ig][" + ry_s + "][" + rx_s + "];\n"
         "                hsum += v.x*wh[wi+0u] + v.y*wh[wi+1u] + v.z*wh[wi+2u] + v.w*wh[wi+3u]; wi += 4u;\n"
         "            }\n        }\n    }\n    uint ox = x * 2u, oy = y * 2u;\n";
    if (aa) {
        const std::string sx0 = "tx + " + std::to_string(ox_shift) + "u";
        const std::string sx1 = "tx + " + std::to_string(ox_shift + 1) + "u";
        const std::string sy0 = "ty + " + std::to_string(oy_shift) + "u";
        const std::string sy1 = "ty + " + std::to_string(oy_shift + 1) + "u";
        r += "    NGU_RAW u00 = " + src_tile_name + "[" + sy0 + "][" + sx0 + "];\n"
             "    float s00 = NGU_FROM_RAW(u00);\n"
             "    float s10 = NGU_FROM_RAW(" + src_tile_name + "[" + sy0 + "][" + sx1 + "]);\n"
             "    float s01 = NGU_FROM_RAW(" + src_tile_name + "[" + sy1 + "][" + sx0 + "]);\n"
             "    float s11 = NGU_FROM_RAW(" + src_tile_name + "[" + sy1 + "][" + sx1 + "]);\n"
             "    precise float v10 = (s00 + s10) * 0.5 + hsum.y;\n"
             "    precise float v01 = (s00 + s01) * 0.5 + hsum.x;\n"
             "    precise float v11 = (((s00 + s10) + s01) + s11) * 0.25 + hsum.z;\n"
             "    d[(oy + 0u) * pc.dst_s + ox + 0u] = u00;\n"
             "    d[(oy + 0u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(v10);\n"
             "    d[(oy + 1u) * pc.dst_s + ox + 0u] = NGU_TO_RAW(v01);\n"
             "    d[(oy + 1u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(v11);\n";
    } else {
        r += "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
             "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
             "    uint c0 = tx * 2u;\n"
             "    float r0a=bx_t[ty+0u][c0+0u], r1a=bx_t[ty+1u][c0+0u], r2a=bx_t[ty+2u][c0+0u], r3a=bx_t[ty+3u][c0+0u], r4a=bx_t[ty+4u][c0+0u];\n"
             "    float r0b=bx_t[ty+0u][c0+1u], r1b=bx_t[ty+1u][c0+1u], r2b=bx_t[ty+2u][c0+1u], r3b=bx_t[ty+3u][c0+1u], r4b=bx_t[ty+4u][c0+1u];\n"
             "    precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + hsum.x;\n"
             "    precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + hsum.y;\n"
             "    precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + hsum.z;\n"
             "    precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + hsum.w;\n"
             "    d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
             "    d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
             "    d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
             "    d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
    }
    if (half)
        replace_shader_text(r,
            "hsum += v.x*wh[wi+0u] + v.y*wh[wi+1u] + v.z*wh[wi+2u] + v.w*wh[wi+3u];",
            "hsum += v.x*wh[wi+0u]; hsum += v.y*wh[wi+1u]; "
            "hsum += v.z*wh[wi+2u]; hsum += v.w*wh[wi+3u];");
    return r;
}

// Low quality 1-pass fused 14x14 kernel (1->8, 8->8, 8->8, 8->head), 0 B workspace.
std::string gen_low_fused(bool aa) const {
    std::string s = shader_preamble;
    s += "layout(local_size_x=16, local_size_y=16, local_size_z=1) in;\n";
    s += aa ? "layout(set=0, binding=0, std430) readonly buffer Source { NGU_RAW s[]; };\n"
              "layout(set=0, binding=1, std430) writeonly buffer Dest { NGU_RAW d[]; };\n"
            : "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
              "layout(set=0, binding=1, std430) writeonly buffer Dest { float d[]; };\n";
    s += "layout(set=0, binding=2, std140) uniform Params {\n"
         "    vec4 w0[18]; vec4 b0[2]; vec4 w1[144]; vec4 b1[2];\n"
         "    vec4 w2[144]; vec4 b2[2]; vec4 wh[72]; vec4 bh[1];\n};\n"
         "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
         "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h; } pc;\n";
    s += aa ? "shared NGU_RAW src_t[22][22];\n" : "shared float src_t[22][22];\n";
    s += "shared vec4 tile[2][20][20];\n";
    if (!aa)
        s += "shared float bx_t[18][28];\n";
    s += "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 16u + tx;\n"
         "    int bx = int(gl_WorkGroupID.x) * 14, by = int(gl_WorkGroupID.y) * 14;\n"
         "    for (uint idx = tid; idx < 484u; idx += 256u) {\n"
         "        uint ly = idx / 22u; uint lx = idx - ly * 22u;\n"
         "        uint gx = uint(clamp(bx + int(lx) - 4, 0, int(pc.width) - 1));\n"
         "        uint gy = uint(clamp(by + int(ly) - 4, 0, int(pc.height) - 1));\n"
         "        src_t[ly][lx] = s[gy * pc.src_s + gx];\n"
         "    }\n    barrier();\n";
    if (!aa) {
        s += "    if (tid < 252u) {\n"
             "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
             "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
             "        uint r = tid / 14u; uint lx = tid - r * 14u;\n"
             "        float s0=src_t[r+2u][lx+2u], s1=src_t[r+2u][lx+3u], s2=src_t[r+2u][lx+4u], s3=src_t[r+2u][lx+5u], s4=src_t[r+2u][lx+6u];\n"
             "        bx_t[r][lx*2u+0u] = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
             "        bx_t[r][lx*2u+1u] = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
             "    }\n";
    }
    const std::string src_read = aa ? "NGU_FROM_RAW(src_t" : "src_t";
    const std::string src_end = aa ? ")" : "";
    auto emit_b0_px = [&](const char *py, const char *px, const char *sfx) {
        return "        vec4 o0=b0[0], o1=b0[1]; uint wi=0u;\n"
               "        for (int ky=0; ky<3; ++ky) for (int kx=0; kx<3; ++kx) {\n"
               "            float v = " + src_read + "[" + py + "+uint(ky)][" + px + "+uint(kx)]" + src_end + ";\n"
               "            o0+=v*w0[wi+0u]; o1+=v*w0[wi+1u]; wi+=2u;\n"
               "        }\n"
               "        o0_" + std::string(sfx) + "=mix(o0,o0*0.1,lessThan(o0,vec4(0.0)));\n"
               "        o1_" + std::string(sfx) + "=mix(o1,o1*0.1,lessThan(o1,vec4(0.0)));\n";
    };
    s += "    {\n        uint py0 = tid / 20u, px0 = tid - py0 * 20u;\n"
         "        vec4 o0_0, o1_0, o0_1, o1_1;\n        {\n" + emit_b0_px("py0", "px0", "0") +
         "        }\n        uint py1 = (tid + 256u) / 20u, px1 = (tid + 256u) - py1 * 20u;\n"
         "        if (tid + 256u < 400u) {\n" + emit_b0_px("py1", "px1", "1") +
         "        }\n        barrier();\n"
         "        tile[0][py0][px0]=o0_0; tile[1][py0][px0]=o1_0;\n"
         "        if (tid+256u<400u) { tile[0][py1][px1]=o0_1; tile[1][py1][px1]=o1_1; }\n"
         "        barrier();\n    }\n";
    // b1: 18x18 (M_out=2, reads M_in=3)
    s += "    {\n        int xl_19=max(0,3-bx), yl_19=max(0,3-by), xh_19=min(19,int(pc.width)+2-bx), yh_19=min(19,int(pc.height)+2-by);\n"
         "        uint py0 = tid / 18u, px0 = tid - py0 * 18u;\n"
         "        vec4 o0_0, o1_0, o0_1, o1_1;\n        {\n"
         "            vec4 o0=b1[0], o1=b1[1];\n" +
         emit_nc4_loop_1px("w1", "tile", 2, 2, "uint(clamp(int(py0)+ky,yl_19,yh_19))",
                           "uint(clamp(int(px0)+kx,xl_19,xh_19))", true) +
         "            o0_0=mix(o0,o0*0.1,lessThan(o0,vec4(0.0))); o1_0=mix(o1,o1*0.1,lessThan(o1,vec4(0.0)));\n"
         "        }\n        uint py1 = (tid + 256u) / 18u, px1 = (tid + 256u) - py1 * 18u;\n"
         "        if (tid + 256u < 324u) {\n"
         "            vec4 o0=b1[0], o1=b1[1];\n" +
         emit_nc4_loop_1px("w1", "tile", 2, 2, "uint(clamp(int(py1)+ky,yl_19,yh_19))",
                           "uint(clamp(int(px1)+kx,xl_19,xh_19))", true) +
         "            o0_1=mix(o0,o0*0.1,lessThan(o0,vec4(0.0))); o1_1=mix(o1,o1*0.1,lessThan(o1,vec4(0.0)));\n"
         "        }\n        barrier();\n"
         "        tile[0][py0][px0]=o0_0; tile[1][py0][px0]=o1_0;\n"
         "        if (tid+256u<324u) { tile[0][py1][px1]=o0_1; tile[1][py1][px1]=o1_1; }\n"
         "        barrier();\n    }\n";
    // b2: 16x16 (M_out=1, reads M_in=2)
    s += "    {\n        int xl_17=max(0,2-bx), yl_17=max(0,2-by), xh_17=min(17,int(pc.width)+1-bx), yh_17=min(17,int(pc.height)+1-by);\n"
         "        vec4 o0=b2[0], o1=b2[1];\n" +
         emit_nc4_loop_1px("w2", "tile", 2, 2, "uint(clamp(int(ty)+ky,yl_17,yh_17))",
                           "uint(clamp(int(tx)+kx,xl_17,xh_17))", true) +
         "        barrier();\n"
         "        tile[0][ty][tx]=mix(o0,o0*0.1,lessThan(o0,vec4(0.0)));\n"
         "        tile[1][ty][tx]=mix(o1,o1*0.1,lessThan(o1,vec4(0.0)));\n"
         "        barrier();\n    }\n";
    // Head: 14x14 (M_out=0, reads M_in=1)
    s += "    if (tx >= 14u || ty >= 14u) return;\n"
         "    uint x = uint(bx) + tx, y = uint(by) + ty;\n"
         "    if (x >= pc.width || y >= pc.height) return;\n"
         "    int xl_h=max(0,1-bx), yl_h=max(0,1-by), xh_h=min(15,int(pc.width)-bx), yh_h=min(15,int(pc.height)-by);\n" +
         emit_head_compute_and_store(8, aa, "src_t", 4, 4, true) +
         "}\n";
    return s;
}

// Coalesced halo loader for 32x12 (local_size=(32,4)) tiles; g_count=0 loads scalar plane, >=1 loads vec4 groups.
std::string emit_32x12_halo_load(const char *dst_arr, const char *src_buf, const char *row_stride,
                                 int g_count, int pad, int extra_x, int rows) const {
    std::string r =
        "    { uint gx = uint(clamp(bx + int(tx) - " + std::to_string(pad) + ", 0, int(pc.width) - 1));\n"
        "      uint gx_r = uint(clamp(bx + 32 + int(tx) - " + std::to_string(pad) + ", 0, int(pc.width) - 1));\n"
        "      for (uint sy = ty; sy < " + std::to_string(rows) + "u; sy += 4u) {\n"
        "          uint row = uint(clamp(by + int(sy) - " + std::to_string(pad) + ", 0, int(pc.height) - 1)) * " +
        std::string(row_stride) + ";\n";
    if (g_count == 0) {
        r += "          " + std::string(dst_arr) + "[sy][tx] = " + std::string(src_buf) + "[row + gx];\n"
             "          if (tx < " + std::to_string(extra_x) + "u) " + std::string(dst_arr) +
             "[sy][32u + tx] = " + std::string(src_buf) + "[row + gx_r];\n";
    } else {
        r += "          for (int g = 0; g < " + std::to_string(g_count) + "; ++g) {\n"
             "              uint ps = uint(g) * pc.plane_stride + row;\n"
             "              " + std::string(dst_arr) + "[g][sy][tx] = " + std::string(src_buf) + "[ps + gx];\n"
             "              if (tx < " + std::to_string(extra_x) + "u) " + std::string(dst_arr) +
             "[g][sy][32u + tx] = " + std::string(src_buf) + "[ps + gx_r];\n"
             "          }\n";
    }
    return r + "      }\n    }\n";
}

// 3-row-per-thread (3py) 3x3 NC4 inner accumulation loop with runtime pc.k3 over kx to control register pressure.
std::string emit_nc4_3py_loop(const char *wname, const char *tname, const char *acc_prefix, int c_in,
                              int og0, int og1, int g_out_total, bool du, bool clamp_coords) const {
    const int igs = c_in / 4;
    const char *comps[4] = {"x", "y", "z", "w"};
    std::string r =
        "        for (uint kx = 0u; kx < pc.k3; ++kx) {\n"
        "            uint kx_off = kx * " + std::to_string(c_in * g_out_total) + "u;\n";
    if (clamp_coords)
        r += "            uint rx = uint(clamp(int(tx + kx), xl, xh));\n";
    r += "            " + std::string(du ? "[[dont_unroll]] " : "") +
         "for (uint ig = 0u; ig < " + std::to_string(igs) + "u; ++ig) {\n";
    for (int row = 0; row < 5; ++row) {
        const std::string y_idx = clamp_coords ? ("ry" + std::to_string(row)) : ("yl + " + std::to_string(row) + "u");
        const std::string x_idx = clamp_coords ? "rx" : "tx + kx";
        r += "                vec4 t" + std::to_string(row) + " = " + std::string(tname) + "[ig][" + y_idx + "][" +
             x_idx + "];\n";
    }
    r += "                uint base = kx_off + ig * " + std::to_string(4 * g_out_total) + "u;\n";
    for (int ky = 0; ky < 3; ++ky)
        for (int c = 0; c < 4; ++c)
            for (int og = og0; og < og1; ++og) {
                const std::string suffix = (g_out_total == 1) ? "" : ("_" + std::to_string(og));
                r += "                { vec4 w = " + std::string(wname) + "[base + " +
                     std::to_string((ky * 3 * c_in + c) * g_out_total + og) + "u]; ";
                for (int p = 0; p < 3; ++p)
                    r += std::string(acc_prefix) + std::to_string(p) + suffix + "+=t" + std::to_string(ky + p) + "." +
                         comps[c] + "*w; ";
                r += "}\n";
            }
    return r + "            }\n        }\n";
}

// 32x12 128-thread single-pass 119-quad + 3py 1->c_mid->c_out fused stage.
std::string gen_in_fused2_3py(int c_mid, int c_out, bool du_l1) const {
    const int gm = c_mid / 4, go = c_out / 4;
    std::string s = shader_preamble;
    s += "layout(local_size_x=32, local_size_y=4, local_size_z=1) in;\n"
         "layout(set=0, binding=0, std430) readonly buffer Source { float s[]; };\n"
         "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n"
         "layout(set=0, binding=2, std140) uniform Params { vec4 w0[" + std::to_string(9 * gm) +
         "]; vec4 b0[" + std::to_string(gm) + "]; vec4 w1[" + std::to_string(9 * c_mid * go) +
         "]; vec4 b1[" + std::to_string(go) + "]; };\n"
         "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
         "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n"
         "shared float src_t[16][36];\n"
         "shared vec4 tile[" + std::to_string(gm) + "][14][34];\n"
         "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 32u + tx;\n"
         "    int bx = int(gl_WorkGroupID.x) * 32, by = int(gl_WorkGroupID.y) * 12;\n" +
         emit_32x12_halo_load("src_t", "s", "pc.src_s", 0, 2, 4, 16) +
         "    barrier();\n"
         "    if (tid < 119u) {\n"
         "        uint qy = tid / 17u, qx = tid - qy * 17u, py0 = qy * 2u, px0 = qx * 2u;\n";
    for (int og = 0; og < gm; ++og)
        s += "        vec4 q00_" + std::to_string(og) + "=b0[" + std::to_string(og) + "], q01_" +
             std::to_string(og) + "=q00_" + std::to_string(og) + ", q10_" + std::to_string(og) + "=q00_" +
             std::to_string(og) + ", q11_" + std::to_string(og) + "=q00_" + std::to_string(og) + ";\n";
    for (int ky = 0; ky < 3; ++ky) {
        s += "        {\n"
             "            float a0=src_t[py0+" + std::to_string(ky) + "u][px0+0u], a1=src_t[py0+" +
             std::to_string(ky) + "u][px0+1u], a2=src_t[py0+" + std::to_string(ky) +
             "u][px0+2u], a3=src_t[py0+" + std::to_string(ky) + "u][px0+3u];\n"
             "            float b0s=src_t[py0+" + std::to_string(ky + 1) + "u][px0+0u], b1s=src_t[py0+" +
             std::to_string(ky + 1) + "u][px0+1u], b2s=src_t[py0+" + std::to_string(ky + 1) +
             "u][px0+2u], b3s=src_t[py0+" + std::to_string(ky + 1) + "u][px0+3u];\n";
        for (int kx = 0; kx < 3; ++kx)
            for (int og = 0; og < gm; ++og)
                s += "            { vec4 w=w0[" + std::to_string((ky * 3 + kx) * gm + og) + "u]; q00_" +
                     std::to_string(og) + "+=a" + std::to_string(kx) + "*w; q01_" + std::to_string(og) +
                     "+=a" + std::to_string(kx + 1) + "*w; q10_" + std::to_string(og) + "+=b" +
                     std::to_string(kx) + "s*w; q11_" + std::to_string(og) + "+=b" +
                     std::to_string(kx + 1) + "s*w; }\n";
        s += "        }\n";
    }
    for (int og = 0; og < gm; ++og)
        for (int py = 0; py < 2; ++py)
            for (int px = 0; px < 2; ++px) {
                const std::string qn = "q" + std::to_string(py) + std::to_string(px) + "_" + std::to_string(og);
                s += "        tile[" + std::to_string(og) + "][py0+" + std::to_string(py) + "u][px0+" +
                     std::to_string(px) + "u]=mix(" + qn + "," + qn + "*0.1,lessThan(" + qn + ",vec4(0.0)));\n";
            }
    s += "    }\n    barrier();\n"
         "    uint x = uint(bx) + tx, y0 = uint(by) + ty * 3u;\n"
         "    if (x >= pc.width || y0 >= pc.height) return;\n"
         "    int xl = max(0, 1 - bx), yl_b = max(0, 1 - by), xh = min(33, int(pc.width) - bx), yh_b = min(13, int(pc.height) - by);\n";
    for (int r = 0; r < 5; ++r)
        s += "    uint ry" + std::to_string(r) + " = uint(clamp(int(ty * 3u) + " + std::to_string(r) + ", yl_b, yh_b));\n";
    s += "    {\n";
    for (int p = 0; p < 3; ++p)
        for (int og = 0; og < go; ++og)
            s += "        vec4 p" + std::to_string(p) + "_" + std::to_string(og) + " = b1[" + std::to_string(og) + "];\n";
    s += emit_nc4_3py_loop("w1", "tile", "p", c_mid, 0, go, go, du_l1, true);
    for (int p = 0; p < 3; ++p) {
        if (p > 0)
            s += "        if (y0 + " + std::to_string(p) + "u < pc.height) {\n";
        s += "        uint ob = (y0 + " + std::to_string(p) + "u) * pc.width + x;\n";
        for (int og = 0; og < go; ++og) {
            const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
            s += "        d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(" + pn + ", " + pn +
                 "*0.1, lessThan(" + pn + ", vec4(0.0)));\n";
        }
        if (p > 0)
            s += "        }\n";
    }
    return s + "    }\n}\n";
}

// 16x16 C->C->C 2-layer fused stage (optimal for C=8 medium Pass 2).
std::string gen_fused2_16x16(int C) const {
    const int G = C / 4;
    const int W4 = 9 * C * G;
    std::string s = shader_preamble;
    s += "layout(local_size_x=16, local_size_y=16, local_size_z=1) in;\n"
         "layout(set=0, binding=0, std430) readonly buffer Source { vec4 s[]; };\n"
         "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n"
         "layout(set=0, binding=2, std140) uniform Params { vec4 w0[" + std::to_string(W4) + "]; vec4 b0[" +
         std::to_string(G) + "]; vec4 w1[" + std::to_string(W4) + "]; vec4 b1[" + std::to_string(G) + "]; };\n"
         "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
         "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h; } pc;\n"
         "shared vec4 tile[" + std::to_string(G) + "][20][20];\n"
         "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 16u + tx;\n"
         "    int bx = int(gl_WorkGroupID.x) * 16, by = int(gl_WorkGroupID.y) * 16;\n"
         "    for (uint idx = tid; idx < " + std::to_string(G * 400) + "u; idx += 256u) {\n"
         "        uint g = idx / 400u; uint rem = idx - g * 400u;\n"
         "        uint ly = rem / 20u; uint lx = rem - ly * 20u;\n"
         "        uint gx = uint(clamp(bx + int(lx) - 2, 0, int(pc.width) - 1));\n"
         "        uint gy = uint(clamp(by + int(ly) - 2, 0, int(pc.height) - 1));\n"
         "        tile[g][ly][lx] = s[g * pc.plane_stride + gy * pc.width + gx];\n"
         "    }\n    barrier();\n";
    for (int og = 0; og < G; ++og)
        s += "    vec4 o" + std::to_string(og) + "_a, o" + std::to_string(og) + "_b;\n";
    s += "    uint py0 = tid / 18u, px0 = tid - py0 * 18u;\n    {\n";
    for (int og = 0; og < G; ++og)
        s += "        vec4 o" + std::to_string(og) + " = b0[" + std::to_string(og) + "];\n";
    s += emit_nc4_loop_1px("w0", "tile", G, G, "py0+uint(ky)", "px0+uint(kx)", false);
    for (int og = 0; og < G; ++og)
        s += "        o" + std::to_string(og) + "_a = mix(o" + std::to_string(og) + ", o" +
             std::to_string(og) + "*0.1, lessThan(o" + std::to_string(og) + ", vec4(0.0)));\n";
    s += "    }\n    uint py1 = (tid + 256u) / 18u, px1 = (tid + 256u) - py1 * 18u;\n"
         "    if (tid + 256u < 324u) {\n";
    for (int og = 0; og < G; ++og)
        s += "        vec4 o" + std::to_string(og) + " = b0[" + std::to_string(og) + "];\n";
    s += emit_nc4_loop_1px("w0", "tile", G, G, "py1+uint(ky)", "px1+uint(kx)", false);
    for (int og = 0; og < G; ++og)
        s += "        o" + std::to_string(og) + "_b = mix(o" + std::to_string(og) + ", o" +
             std::to_string(og) + "*0.1, lessThan(o" + std::to_string(og) + ", vec4(0.0)));\n";
    s += "    }\n    barrier();\n";
    for (int og = 0; og < G; ++og)
        s += "    tile[" + std::to_string(og) + "][py0][px0] = o" + std::to_string(og) + "_a;\n";
    s += "    if (tid + 256u < 324u) {\n";
    for (int og = 0; og < G; ++og)
        s += "        tile[" + std::to_string(og) + "][py1][px1] = o" + std::to_string(og) + "_b;\n";
    s += "    }\n    barrier();\n"
         "    uint x = uint(bx) + tx, y = uint(by) + ty;\n"
         "    if (x >= pc.width || y >= pc.height) return;\n"
         "    int xl = max(0, 1 - bx), yl = max(0, 1 - by), xh = min(17, int(pc.width) - bx), yh = min(17, int(pc.height) - by);\n";
    for (int og = 0; og < G; ++og)
        s += "    vec4 o" + std::to_string(og) + " = b1[" + std::to_string(og) + "];\n";
    s += emit_nc4_loop_1px("w1", "tile", G, G, "uint(clamp(int(ty)+ky,yl,yh))",
                           "uint(clamp(int(tx)+kx,xl,xh))", true);
    s += "    uint ob = y * pc.width + x;\n";
    for (int og = 0; og < G; ++og)
        s += "    d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(o" + std::to_string(og) +
             ", o" + std::to_string(og) + "*0.1, lessThan(o" + std::to_string(og) + ", vec4(0.0)));\n";
    s += "}\n";
    return s;
}

// 32x12 128-thread 3px/thread (3py) 1-layer C->C stage (optimal for high C=12 & very_high C=16).
// The five loaded rows feed all three outputs. Clamp indices are hoisted and
// du controls the ig unroll hint for both precisions (FP32 Ada: 128/168 registers).
std::string gen_1L_3py(int C) const {
    const int G = C / 4;
    std::string s = shader_preamble;
    s += "layout(local_size_x=32, local_size_y=4, local_size_z=1) in;\n"
         "layout(set=0, binding=0, std430) readonly buffer Source { vec4 s[]; };\n"
         "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n"
         "layout(set=0, binding=2, std140) uniform Params { vec4 w0[" + std::to_string(9 * C * G) +
         "]; vec4 b0[" + std::to_string(G) + "]; };\n"
         "layout(set=0, binding=3, std430) readonly buffer Aux { float aux_unused[]; };\n"
         "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n"
         "shared vec4 tile[" + std::to_string(G) + "][14][34];\n"
         "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y;\n"
         "    int bx = int(gl_WorkGroupID.x) * 32, by = int(gl_WorkGroupID.y) * 12;\n" +
         emit_32x12_halo_load("tile", "s", "pc.width", G, 1, 2, 14) +
         "    barrier();\n"
         "    uint x = uint(bx) + tx, y0 = uint(by) + ty * 3u;\n"
         "    if (x >= pc.width || y0 >= pc.height) return;\n"
         "    uint yl = ty * 3u;\n    {\n";
    for (int p = 0; p < 3; ++p)
        for (int og = 0; og < G; ++og)
            s += "        vec4 p" + std::to_string(p) + "_" + std::to_string(og) + " = b0[" + std::to_string(og) + "];\n";
    s += emit_nc4_3py_loop("w0", "tile", "p", C, 0, G, G, true, false);
    for (int p = 0; p < 3; ++p) {
        if (p > 0)
            s += "        if (y0 + " + std::to_string(p) + "u < pc.height) {\n";
        s += "        uint ob = (y0 + " + std::to_string(p) + "u) * pc.width + x;\n";
        for (int og = 0; og < G; ++og) {
            const std::string pn = "p" + std::to_string(p) + "_" + std::to_string(og);
            s += "        d[" + std::to_string(og) + "u * pc.plane_stride + ob] = mix(" + pn + ", " + pn +
                 "*0.1, lessThan(" + pn + ", vec4(0.0)));\n";
        }
        if (p > 0)
            s += "        }\n";
    }
    return s + "    }\n}\n";
}

// 14x14 fused last C->C body layer + head + combine (used for medium Pass 3 & high aa).
std::string gen_tail_14x14(int C, bool aa) const {
    const int G = C / 4;
    const int W4 = 9 * C * G;
    const int WH = 9 * C;
    std::string s = shader_preamble;
    s += "layout(local_size_x=16, local_size_y=16, local_size_z=1) in;\n"
         "layout(set=0, binding=0, std430) readonly buffer Features { vec4 s[]; };\n";
    s += aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
            : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
    s += "layout(set=0, binding=2, std140) uniform Params { vec4 w0[" + std::to_string(W4) + "]; vec4 b0[" +
         std::to_string(G) + "]; vec4 wh[" + std::to_string(WH) + "]; vec4 bh[1]; };\n";
    s += aa ? "layout(set=0, binding=3, std430) readonly buffer OrigSrc { NGU_RAW src_in[]; };\n"
            : "layout(set=0, binding=3, std430) readonly buffer OrigSrc { float src_in[]; };\n";
    s += "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h; } pc;\n"
         "shared vec4 tile[" + std::to_string(G) + "][18][18];\n";
    if (aa)
        s += "shared NGU_RAW src_u_t[16][16];\n";
    else
        s += "shared float src_t[18][18];\nshared float bx_t[18][28];\n";
    s += "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 16u + tx;\n"
         "    int bx = int(gl_WorkGroupID.x) * 14, by = int(gl_WorkGroupID.y) * 14;\n"
         "    for (uint idx = tid; idx < " + std::to_string(G * 324) + "u; idx += 256u) {\n"
         "        uint g = idx / 324u; uint rem = idx - g * 324u;\n"
         "        uint ly = rem / 18u; uint lx = rem - ly * 18u;\n"
         "        uint gx = uint(clamp(bx + int(lx) - 2, 0, int(pc.width) - 1));\n"
         "        uint gy = uint(clamp(by + int(ly) - 2, 0, int(pc.height) - 1));\n"
         "        tile[g][ly][lx] = s[g * pc.plane_stride + gy * pc.width + gx];\n"
         "    }\n";
    if (aa) {
        s += "    {\n"
             "        uint gx = uint(clamp(bx + int(tx), 0, int(pc.width) - 1));\n"
             "        uint gy = uint(clamp(by + int(ty), 0, int(pc.height) - 1));\n"
             "        src_u_t[ty][tx] = src_in[gy * pc.src_s + gx];\n"
             "    }\n    barrier();\n";
    } else {
        s += "    for (uint idx = tid; idx < 324u; idx += 256u) {\n"
             "        uint ly = idx / 18u; uint lx = idx - ly * 18u;\n"
             "        uint gx = uint(clamp(bx + int(lx) - 2, 0, int(pc.width) - 1));\n"
             "        uint gy = uint(clamp(by + int(ly) - 2, 0, int(pc.height) - 1));\n"
             "        src_t[ly][lx] = src_in[gy * pc.src_s + gx];\n"
             "    }\n    barrier();\n"
             "    if (tid < 252u) {\n"
             "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
             "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
             "        uint r = tid / 14u; uint lx = tid - r * 14u;\n"
             "        float s0=src_t[r][lx+0u], s1=src_t[r][lx+1u], s2=src_t[r][lx+2u], s3=src_t[r][lx+3u], s4=src_t[r][lx+4u];\n"
             "        bx_t[r][lx*2u+0u] = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
             "        bx_t[r][lx*2u+1u] = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
             "    }\n";
    }
    s += "    {\n";
    for (int og = 0; og < G; ++og)
        s += "        vec4 o" + std::to_string(og) + " = b0[" + std::to_string(og) + "];\n";
    s += emit_nc4_loop_1px("w0", "tile", G, G, "ty+uint(ky)", "tx+uint(kx)", false);
    s += "        barrier();\n";
    for (int og = 0; og < G; ++og)
        s += "        tile[" + std::to_string(og) + "][ty][tx] = mix(o" + std::to_string(og) + ", o" +
             std::to_string(og) + "*0.1, lessThan(o" + std::to_string(og) + ", vec4(0.0)));\n";
    s += "        barrier();\n    }\n"
         "    if (tx >= 14u || ty >= 14u) return;\n"
         "    uint x = uint(bx) + tx, y = uint(by) + ty;\n"
         "    if (x >= pc.width || y >= pc.height) return;\n"
         "    int xl_h=max(0,1-bx), yl_h=max(0,1-by), xh_h=min(15,int(pc.width)-bx), yh_h=min(15,int(pc.height)-by);\n" +
         emit_head_compute_and_store(C, aa, "src_u_t", 0, 0, true) +
         "}\n";
    return s;
}

// 16x16 standalone head + baseline/AA combine (used for high C=12 & very_high C=16).
std::string gen_head_end(int C, bool aa) const {
    const int G = C / 4;
    const int WH = 9 * C;
    std::string s = shader_preamble;
    s += "layout(local_size_x=16, local_size_y=16, local_size_z=1) in;\n"
         "layout(set=0, binding=0, std430) readonly buffer Features { vec4 feat[]; };\n";
    s += aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
            : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
    s += "layout(set=0, binding=2, std140) uniform Params { vec4 wh[" + std::to_string(WH) +
         "]; vec4 bh[1]; };\n";
    s += aa ? "layout(set=0, binding=3, std430) readonly buffer OrigSrc { NGU_RAW src_in[]; };\n"
            : "layout(set=0, binding=3, std430) readonly buffer OrigSrc { float src_in[]; };\n";
    s += "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h; } pc;\n"
         "shared vec4 tile[" + std::to_string(G) + "][18][18];\n";
    if (aa)
        s += "shared NGU_RAW src_u_t[18][18];\n";
    else
        s += "shared float src_t[20][20];\nshared float bx_t[20][32];\n";
    s += "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 16u + tx;\n"
         "    int bx = int(gl_WorkGroupID.x) * 16, by = int(gl_WorkGroupID.y) * 16;\n"
         "    for (uint idx = tid; idx < " + std::to_string(G * 324) + "u; idx += 256u) {\n"
         "        uint g = idx / 324u; uint rem = idx - g * 324u;\n"
         "        uint ly = rem / 18u; uint lx = rem - ly * 18u;\n"
         "        uint gx = uint(clamp(bx + int(lx) - 1, 0, int(pc.width) - 1));\n"
         "        uint gy = uint(clamp(by + int(ly) - 1, 0, int(pc.height) - 1));\n"
         "        tile[g][ly][lx] = feat[g * pc.plane_stride + gy * pc.width + gx];\n"
         "    }\n";
    if (aa) {
        s += "    for (uint idx = tid; idx < 324u; idx += 256u) {\n"
             "        uint ly = idx / 18u; uint lx = idx - ly * 18u;\n"
             "        uint gx = uint(clamp(bx + int(lx) - 1, 0, int(pc.width) - 1));\n"
             "        uint gy = uint(clamp(by + int(ly) - 1, 0, int(pc.height) - 1));\n"
             "        src_u_t[ly][lx] = src_in[gy * pc.src_s + gx];\n"
             "    }\n    barrier();\n";
    } else {
        s += "    for (uint idx = tid; idx < 400u; idx += 256u) {\n"
             "        uint ly = idx / 20u; uint lx = idx - ly * 20u;\n"
             "        uint gx = uint(clamp(bx + int(lx) - 2, 0, int(pc.width) - 1));\n"
             "        uint gy = uint(clamp(by + int(ly) - 2, 0, int(pc.height) - 1));\n"
             "        src_t[ly][lx] = src_in[gy * pc.src_s + gx];\n"
             "    }\n    barrier();\n"
             "    {\n"
             "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
             "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
             "        for (uint idx = tid; idx < 320u; idx += 256u) {\n"
             "            uint r = idx >> 4; uint lx = idx & 15u;\n"
             "            float s0=src_t[r][lx+0u], s1=src_t[r][lx+1u], s2=src_t[r][lx+2u], s3=src_t[r][lx+3u], s4=src_t[r][lx+4u];\n"
             "            bx_t[r][lx*2u+0u] = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
             "            bx_t[r][lx*2u+1u] = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
             "        }\n"
             "    }\n    barrier();\n";
    }
    s += "    uint x = uint(bx) + tx, y = uint(by) + ty;\n"
         "    if (x >= pc.width || y >= pc.height) return;\n" +
         emit_head_compute_and_store(C, aa, "src_u_t", 1, 1, false) +
         "}\n";
    return s;
}

// Emits 3-row (3py) head upsampling combine & global store for 32x12 tiles, preserving FP32 order and AA -0.0.
std::string emit_3py_head_combine_and_store(bool aa, int src_pad) const {
    auto off = [](const char *var, int add) {
        return add == 0 ? std::string(var) : (std::string(var) + "+" + std::to_string(add) + "u");
    };
    std::string r = "    uint ox = x * 2u;\n";
    if (!aa)
        r += "    float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
             "    float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
             "    uint c0 = tx * 2u;\n";
    for (int p = 0; p < 3; ++p) {
        if (p > 0)
            r += "    if (y0 + " + std::to_string(p) + "u < pc.height) {\n";
        r += "    uint ry = ty * 3u + " + std::to_string(p) + "u, oy = (y0 + " + std::to_string(p) + "u) * 2u;\n";
        const std::string hn = "h" + std::to_string(p);
        if (aa) {
            const std::string sy0 = off("ry", src_pad), sy1 = off("ry", src_pad + 1);
            const std::string sx0 = off("tx", src_pad), sx1 = off("tx", src_pad + 1);
            r += "    NGU_RAW u00 = src_u_t[" + sy0 + "][" + sx0 + "];\n"
                 "    float s00 = NGU_FROM_RAW(u00);\n"
                 "    float s10 = NGU_FROM_RAW(src_u_t[" + sy0 + "][" + sx1 + "]);\n"
                 "    float s01 = NGU_FROM_RAW(src_u_t[" + sy1 + "][" + sx0 + "]);\n"
                 "    float s11 = NGU_FROM_RAW(src_u_t[" + sy1 + "][" + sx1 + "]);\n"
                 "    precise float p10 = (s00 + s10) * 0.5 + " + hn + ".y;\n"
                 "    precise float p01 = (s00 + s01) * 0.5 + " + hn + ".x;\n"
                 "    precise float p11 = (((s00 + s10) + s01) + s11) * 0.25 + " + hn + ".z;\n"
                 "    d[(oy + 0u) * pc.dst_s + ox + 0u] = u00;\n"
                 "    d[(oy + 0u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p10);\n"
                 "    d[(oy + 1u) * pc.dst_s + ox + 0u] = NGU_TO_RAW(p01);\n"
                 "    d[(oy + 1u) * pc.dst_s + ox + 1u] = NGU_TO_RAW(p11);\n";
        } else {
            r += "    float r0a=bx_t[ry+0u][c0], r1a=bx_t[ry+1u][c0], r2a=bx_t[ry+2u][c0], r3a=bx_t[ry+3u][c0], r4a=bx_t[ry+4u][c0];\n"
                 "    float r0b=bx_t[ry+0u][c0+1u], r1b=bx_t[ry+1u][c0+1u], r2b=bx_t[ry+2u][c0+1u], r3b=bx_t[ry+3u][c0+1u], r4b=bx_t[ry+4u][c0+1u];\n"
                 "    precise float p00 = ((((cd*r0a)+(cb*r1a))+(ca*r2a))+(cc*r3a)) + " + hn + ".x;\n"
                 "    precise float p10 = ((((cd*r0b)+(cb*r1b))+(ca*r2b))+(cc*r3b)) + " + hn + ".y;\n"
                 "    precise float p01 = ((((cc*r1a)+(ca*r2a))+(cb*r3a))+(cd*r4a)) + " + hn + ".z;\n"
                 "    precise float p11 = ((((cc*r1b)+(ca*r2b))+(cb*r3b))+(cd*r4b)) + " + hn + ".w;\n"
                 "    d[(oy + 0u) * pc.dst_s + ox + 0u] = p00;\n"
                 "    d[(oy + 0u) * pc.dst_s + ox + 1u] = p10;\n"
                 "    d[(oy + 1u) * pc.dst_s + ox + 0u] = p01;\n"
                 "    d[(oy + 1u) * pc.dst_s + ox + 1u] = p11;\n";
        }
        if (p > 0)
            r += "    }\n";
    }
    return r;
}

// 32x12 128-thread 3py head + AA combine (optimal for very_high aa C=16).
std::string gen_head_end_3py(int C) const {
    const int G = C / 4;
    std::string s = shader_preamble;
    s += "layout(local_size_x=32, local_size_y=4, local_size_z=1) in;\n"
         "layout(set=0, binding=0, std430) readonly buffer Features { vec4 feat[]; };\n"
         "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
         "layout(set=0, binding=2, std140) uniform Params { vec4 wh[" + std::to_string(9 * C) +
         "]; vec4 bh[1]; };\n"
         "layout(set=0, binding=3, std430) readonly buffer OrigSrc { NGU_RAW src_in[]; };\n"
         "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n"
         "shared vec4 tile[" + std::to_string(G) + "][14][34];\n"
         "shared NGU_RAW src_u_t[14][34];\n"
         "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y;\n"
         "    int bx = int(gl_WorkGroupID.x) * 32, by = int(gl_WorkGroupID.y) * 12;\n" +
         emit_32x12_halo_load("tile", "feat", "pc.width", G, 1, 2, 14) +
         emit_32x12_halo_load("src_u_t", "src_in", "pc.src_s", 0, 1, 2, 14) +
         "    barrier();\n"
         "    uint x = uint(bx) + tx, y0 = uint(by) + ty * 3u;\n"
         "    if (x >= pc.width || y0 >= pc.height) return;\n"
         "    {\n"
         "        uint yl = ty * 3u;\n"
         "        vec4 h0 = bh[0], h1 = h0, h2 = h0;\n" +
         emit_nc4_3py_loop("wh", "tile", "h", C, 0, 1, 1, false, false) +
         emit_3py_head_combine_and_store(true, 1) +
         "    }\n}\n";
    return s;
}

// 32x12 119-quad + 3py: body + head/combine, or two body layers.
// FP16 shares each weight across four pixels without changing the tap order.
std::string gen_tail_32x12(int C, bool aa, bool body_only = false) const {
    const int G = C / 4;
    std::string s = shader_preamble;
    s += "layout(local_size_x=32, local_size_y=4, local_size_z=1) in;\n"
         "layout(set=0, binding=0, std430) readonly buffer Features { vec4 s[]; };\n";
    s += body_only ? "layout(set=0, binding=1, std430) writeonly buffer Dest { vec4 d[]; };\n" : aa ? "layout(set=0, binding=1, std430) writeonly buffer Dest4K { NGU_RAW d[]; };\n"
            : "layout(set=0, binding=1, std430) writeonly buffer Dest4K { float d[]; };\n";
    s += "layout(set=0, binding=2, std140) uniform Params { vec4 w0[" + std::to_string(9 * C * G) +
         "]; vec4 b0[" + std::to_string(G) + "]; vec4 wh[" + std::to_string(body_only ? 9*C*G : 9*C) + "]; vec4 bh[" + std::to_string(body_only ? G : 1) + "]; };\n";
    s += aa ? "layout(set=0, binding=3, std430) readonly buffer OrigSrc { NGU_RAW src_in[]; };\n"
            : "layout(set=0, binding=3, std430) readonly buffer OrigSrc { float src_in[]; };\n";
    s += "layout(push_constant, std430) uniform Push { uint width, height, plane_stride, src_s, dst_s, dst_w, dst_h, k3; } pc;\n"
         "shared vec4 smem[" + std::to_string(G) + "][16][36];\n";
    if (!body_only && aa)
        s += "shared NGU_RAW src_u_t[13][33];\n";
    else if (!body_only)
        s += "shared float src_t[16][36];\nshared float bx_t[16][64];\n";
    s += "void main() {\n"
         "    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y, tid = ty * 32u + tx;\n"
         "    int bx = int(gl_WorkGroupID.x) * 32, by = int(gl_WorkGroupID.y) * 12;\n" +
         emit_32x12_halo_load("smem", "s", "pc.width", G, 2, 4, 16) +
         (body_only ? "" : aa ? emit_32x12_halo_load("src_u_t", "src_in", "pc.src_s", 0, 0, 1, 13)
             : emit_32x12_halo_load("src_t", "src_in", "pc.src_s", 0, 2, 4, 16)) +
         "    barrier();\n";
    if (!body_only && !aa) {
        s += "    {\n"
             "        float ca = uintBitsToFloat(0x3f5f3333u), cb = uintBitsToFloat(0x3e766666u);\n"
             "        float cc = uintBitsToFloat(0xbdaccccdu), cd = uintBitsToFloat(0xbce66666u);\n"
             "        for (uint sy = ty; sy < 16u; sy += 4u) {\n"
             "            float s0=src_t[sy][tx+0u], s1=src_t[sy][tx+1u], s2=src_t[sy][tx+2u], s3=src_t[sy][tx+3u], s4=src_t[sy][tx+4u];\n"
             "            bx_t[sy][tx*2u+0u] = (((cd*s0)+(cb*s1))+(ca*s2))+(cc*s3);\n"
             "            bx_t[sy][tx*2u+1u] = (((cc*s1)+(ca*s2))+(cb*s3))+(cd*s4);\n"
             "        }\n    }\n";
    }
    const char *comps[4] = {"x", "y", "z", "w"};
    for (int og = 0; og < G; ++og)
        s += "    vec4 q00_" + std::to_string(og) + ", q01_" + std::to_string(og) + ", q10_" +
             std::to_string(og) + ", q11_" + std::to_string(og) + ";\n";
    s += "    uint qy = 0u, qx = 0u, py0 = 0u, px0 = 0u;\n"
         "    if (tid < 119u) {\n"
         "        qy = tid / 17u; qx = tid - qy * 17u; py0 = qy * 2u; px0 = qx * 2u;\n";
    for (int og = 0; og < G; ++og)
        s += "        q00_" + std::to_string(og) + "=b0[" + std::to_string(og) + "]; q01_" +
             std::to_string(og) + "=q00_" + std::to_string(og) + "; q10_" + std::to_string(og) + "=q00_" +
             std::to_string(og) + "; q11_" + std::to_string(og) + "=q00_" + std::to_string(og) + ";\n";
    if (half) {
        // Reuse a 2x4 patch across four output pixels. kx/ic are emitted in
        // scalar-oracle order; ky stays rolled to bound the instruction cache.
        s += "        for (uint ky=0u; ky<pc.k3; ++ky) {\n";
        for (int ig = 0; ig < G; ++ig) {
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 4; ++col)
                    s += "            vec4 t" + std::to_string(ig) + "_" + std::to_string(row) + "_" + std::to_string(col) +
                         "=smem[" + std::to_string(ig) + "][py0+ky+" + std::to_string(row) + "u][px0+" + std::to_string(col) + "u];\n";
        }
        for (int kx = 0; kx < 3; ++kx)
            for (int ig = 0; ig < G; ++ig)
                for (int c = 0; c < 4; ++c)
                    for (int og = 0; og < G; ++og) {
                        s += "            { vec4 w=w0[ky*" + std::to_string(3*C*G) + "u+" +
                             std::to_string((kx*C+ig*4+c)*G+og) + "u]; ";
                        for (int row = 0; row < 2; ++row)
                            for (int col = 0; col < 2; ++col)
                                s += "q" + std::to_string(row) + std::to_string(col) + "_" + std::to_string(og) +
                                     "+=t" + std::to_string(ig) + "_" + std::to_string(row) + "_" + std::to_string(kx+col) +
                                     "." + comps[c] + "*w; ";
                        s += "}\n";
                    }
        s += "        }\n";
    } else {
        s += "        for (uint ky = 0u; ky < pc.k3; ++ky) {\n"
             "            uint ky_off = ky * " + std::to_string(3 * C * G) + "u;\n"
             "            [[dont_unroll]] for (uint ig = 0u; ig < " + std::to_string(G) + "u; ++ig) {\n"
             "                vec4 a0=smem[ig][py0+ky][px0+0u], a1=smem[ig][py0+ky][px0+1u], a2=smem[ig][py0+ky][px0+2u], a3=smem[ig][py0+ky][px0+3u];\n"
             "                vec4 b0s=smem[ig][py0+ky+1u][px0+0u], b1s=smem[ig][py0+ky+1u][px0+1u], b2s=smem[ig][py0+ky+1u][px0+2u], b3s=smem[ig][py0+ky+1u][px0+3u];\n"
             "                uint base = ky_off + ig * " + std::to_string(4 * G) + "u;\n";
        for (int kx = 0; kx < 3; ++kx)
            for (int c = 0; c < 4; ++c)
                for (int og = 0; og < G; ++og)
                    s += "                { vec4 w=w0[base+" + std::to_string((kx * C + c) * G + og) + "u]; q00_" +
                         std::to_string(og) + "+=a" + std::to_string(kx) + "." + comps[c] + "*w; q01_" +
                         std::to_string(og) + "+=a" + std::to_string(kx + 1) + "." + comps[c] + "*w; q10_" +
                         std::to_string(og) + "+=b" + std::to_string(kx) + "s." + comps[c] + "*w; q11_" +
                         std::to_string(og) + "+=b" + std::to_string(kx + 1) + "s." + comps[c] + "*w; }\n";
        s += "            }\n        }\n";
    }
    s += "    }\n    barrier();\n    if (tid < 119u) {\n";
    for (int og = 0; og < G; ++og)
        for (int py = 0; py < 2; ++py)
            for (int px = 0; px < 2; ++px) {
                const std::string qn = "q" + std::to_string(py) + std::to_string(px) + "_" + std::to_string(og);
                s += "        smem[" + std::to_string(og) + "][py0+" + std::to_string(py) + "u][px0+" +
                     std::to_string(px) + "u]=mix(" + qn + "," + qn + "*0.1,lessThan(" + qn + ",vec4(0.0)));\n";
            }
    s += "    }\n    barrier();\n"
         "    uint x = uint(bx) + tx, y0 = uint(by) + ty * 3u;\n"
         "    if (x >= pc.width || y0 >= pc.height) return;\n"
         "    int xl = max(0, 1 - bx), yl_b = max(0, 1 - by), xh = min(33, int(pc.width) - bx), yh_b = min(13, int(pc.height) - by);\n";
    for (int r = 0; r < 5; ++r)
        s += "    uint ry" + std::to_string(r) + " = uint(clamp(int(ty * 3u) + " + std::to_string(r) + ", yl_b, yh_b));\n";
    if (body_only) {
        s += "    {\n";
        for (int p = 0; p < 3; ++p)
            for (int og = 0; og < G; ++og)
                s += "        vec4 p" + std::to_string(p) + "_" + std::to_string(og) + "=bh[" + std::to_string(og) + "];\n";
        s += emit_nc4_3py_loop("wh", "smem", "p", C, 0, G, G, true, true);
        for (int p = 0; p < 3; ++p) {
            const auto n = std::to_string(p);
            s += "        if (y0+" + n + "u<pc.height) {\n";
            for (int og = 0; og < G; ++og) {
                const auto v = "p" + n + "_" + std::to_string(og);
                s += "            d[" + std::to_string(og) + "u*pc.plane_stride+(y0+" + n + "u)*pc.width+x]=mix(" +
                     v + "," + v + "*0.1,lessThan(" + v + ",vec4(0.0)));\n";
            }
            s += "        }\n";
        }
        s += "    }\n}\n";
    } else {
        s += "    {\n        vec4 h0 = bh[0], h1 = h0, h2 = h0;\n" +
             emit_nc4_3py_loop("wh", "smem", "h", C, 0, 1, 1, true, true) +
             emit_3py_head_combine_and_store(aa, 0) +
             "    }\n}\n";
    }
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
        min_smem = 16752;
        max_ubo = 385 * 16;
    } else if (b.size() == 5 && is_conv3(b[0], 1, 12) && is_conv3(b[1], 12, 8) &&
               is_conv3(b[2], 8, 8) && is_conv3(b[3], 8, 8) && is_conv3(b[4], 8, 8) &&
               h.input_channels() == 8) {
        topo = Topology::Medium;
        min_smem = 25152;
        max_ubo = 292 * 16;
    } else if (b.size() == 6 && is_conv3(b[0], 1, 12) && is_conv3(b[1], 12, 12) &&
               is_conv3(b[2], 12, 12) && is_conv3(b[3], 12, 12) && is_conv3(b[4], 12, 12) &&
               is_conv3(b[5], 12, 12) && h.input_channels() == 12) {
        topo = Topology::High;
        min_smem = 25152;
        max_ubo = 436 * 16;
    } else if (b.size() == 8 && is_conv3(b[0], 1, 16) && is_conv3(b[1], 16, 16) &&
               is_conv3(b[2], 16, 16) && is_conv3(b[3], 16, 16) && is_conv3(b[4], 16, 16) &&
               is_conv3(b[5], 16, 16) && is_conv3(b[6], 16, 16) && is_conv3(b[7], 16, 16) &&
               h.input_channels() == 16) {
        topo = Topology::VeryHigh;
        min_smem = 32768;
        max_ubo = 620 * 16;
    }
    if (topo == Topology::Fallback)
        return Topology::Fallback;
    if (half) {
        min_smem /= 2;
        if (topo == Topology::High)
            min_smem = 17024; // C=12 32x12 tail: features + baseline X + original plane
    }
    const auto parameter_limit = limits.maxUniformBufferRange;
    if (limits.maxComputeSharedMemorySize < min_smem || parameter_limit < max_ubo ||
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
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkPipeline> owned_pipelines;
    Buffer parameters{nullptr, BufferDeleter{nullptr}};
    VSVulkanBufferInfo parameter_info{};
    int width = 0, height = 0;
    bool aa = false;
    bool half_compute = false;
    VSVideoFormat format{};

    // Fast path state
    bool fast_path = false;
    std::vector<FastPass> fast_passes;
    std::array<VkDescriptorBufferInfo, 2> fast_slices{};
    VkDeviceSize workspace_size = 0;

    struct WorkspaceSlot {
        Buffer buf{nullptr, BufferDeleter{nullptr}};
        VSVulkanBufferInfo info{};
        std::uint64_t signaled = 0;
        bool busy = false;
    };
    struct WaitNode {
        std::condition_variable cv;
        WorkspaceSlot *slot = nullptr;
        WaitNode *next = nullptr;
    };
    static constexpr std::size_t kWorkspaceSlots = 2;
    mutable std::mutex ws_mutex;
    mutable WaitNode *ws_wait_head = nullptr;
    mutable WaitNode *ws_wait_tail = nullptr;
    mutable std::vector<WorkspaceSlot> ws_slots;

    // Fallback state
    VkPipeline fallback_pipeline = VK_NULL_HANDLE;
    std::vector<LayerInfo> layers;
    std::array<VkDescriptorBufferInfo, 5> fallback_slices{}; // ping, pong, head, baseline X/Y

    ~Impl() {
        // Drain all in-flight GPU submissions before destroying workspace buffers, pipelines, or pools.
        if (gpu && pool) {
            gpu->freeGPUExecPool(pool);
            pool = nullptr;
        }
        ws_slots.clear();
        if (vk && handles.device != VK_NULL_HANDLE) {
            for (VkPipeline p : owned_pipelines)
                if (p != VK_NULL_HANDLE)
                    vk->vkDestroyPipeline(handles.device, p, nullptr);
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
            // Developer audit only, compiled out of the timed/default build.
            // Capture actual compiler output, including scalar fallback modules.
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
    }

    void init_fast(const Model &model, Topology topo) {
        fast_path = true;
        setup_descriptors_and_layout(true, sizeof(FastPush));
        const ShaderGenerator shader{half_compute};

        std::vector<float> ubo_data;
        const VkDeviceSize ubo_align = std::max<VkDeviceSize>(16, limits.minUniformBufferOffsetAlignment);
        const auto append_ubo_block = [&](const std::vector<float> &block, VkDeviceSize &out_off,
                                          VkDeviceSize &out_size) {
            const VkDeviceSize element_bytes = 4;
            const VkDeviceSize cur_bytes = static_cast<VkDeviceSize>(ubo_data.size()) * element_bytes;
            out_off = align_up(cur_bytes, ubo_align);
            if (out_off > cur_bytes)
                ubo_data.resize(static_cast<std::size_t>(out_off / element_bytes), 0.0f);
            out_size = static_cast<VkDeviceSize>(block.size()) * element_bytes;
            if (out_size > limits.maxUniformBufferRange)
                throw std::runtime_error("fused parameters exceed Vulkan buffer range");
            ubo_data.insert(ubo_data.end(), block.begin(), block.end());
        };
        const auto append_head = [&](std::vector<float> &dst) {
            if (aa) {
                append_floats(dst, pack_aa_head_3x3(model.head));
                append_floats(dst, pack_bias(model.head));
            } else {
                append_floats(dst, pack_k4_head_3x3(model.head));
                const float bh = model.head.bias_at(0);
                dst.insert(dst.end(), 4, bh);
            }
        };

        int slice_channels = 0;
        if (topo == Topology::Low) {
            std::vector<float> blk;
            append_floats(blk, pack_1toC(model.body[0]));
            append_floats(blk, pack_bias(model.body[0]));
            append_floats(blk, pack_nc4(model.body[1]));
            append_floats(blk, pack_bias(model.body[1]));
            append_floats(blk, pack_nc4(model.body[2]));
            append_floats(blk, pack_bias(model.body[2]));
            append_head(blk);
            VkDeviceSize off = 0, sz = 0;
            append_ubo_block(blk, off, sz);
            VkPipeline pipe = compile_pipeline(shader.gen_low_fused(aa).c_str());
            fast_passes.push_back(FastPass{pipe, off, sz, 14, 14, -1, -2});
        } else if (topo == Topology::Medium) {
            slice_channels = 8;
            // Pass 1: 1->12->8 (32x12)
            {
                std::vector<float> blk;
                append_floats(blk, pack_1toC(model.body[0]));
                append_floats(blk, pack_bias(model.body[0]));
                append_floats(blk, pack_nc4(model.body[1]));
                append_floats(blk, pack_bias(model.body[1]));
                VkDeviceSize off = 0, sz = 0;
                append_ubo_block(blk, off, sz);
                VkPipeline pipe = compile_pipeline(shader.gen_in_fused2_3py(12, 8, false).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, 32, 12, -1, 0});
            }
            // Pass 2: 8->8->8 (FP16 32x12 / FP32 16x16)
            {
                std::vector<float> blk;
                append_floats(blk, pack_nc4(model.body[2]));
                append_floats(blk, pack_bias(model.body[2]));
                append_floats(blk, pack_nc4(model.body[3]));
                append_floats(blk, pack_bias(model.body[3]));
                VkDeviceSize off = 0, sz = 0;
                append_ubo_block(blk, off, sz);
                VkPipeline pipe = compile_pipeline((half_compute ? shader.gen_tail_32x12(8, true, true) :
                    shader.gen_fused2_16x16(8)).c_str());
                fast_passes.push_back(FastPass{pipe, off, sz, half_compute ? 32u : 16u,
                    half_compute ? 12u : 16u, 0, 1});
            }
            // Pass 3: 8->8 + head (32x12)
            {
                std::vector<float> blk;
                append_floats(blk, pack_nc4(model.body[4]));
                append_floats(blk, pack_bias(model.body[4]));
                append_head(blk);
                VkDeviceSize off = 0, sz = 0;
                append_ubo_block(blk, off, sz);
                const std::string tail = shader.gen_tail_32x12(8, aa);
                VkPipeline pipe = compile_pipeline(tail.c_str());
                fast_passes.push_back(FastPass{pipe, off, sz,
                    32, 12, 1, -2});
            }
        } else {
            const bool is_high = (topo == Topology::High);
            const int C = is_high ? 12 : 16;
            const std::uint32_t mid_tw = 32;
            const std::uint32_t mid_th = 12;
            slice_channels = C;
            // Pass 1: 1->C->C fused (32x12)
            {
                std::vector<float> blk;
                append_floats(blk, pack_1toC(model.body[0]));
                append_floats(blk, pack_bias(model.body[0]));
                append_floats(blk, pack_nc4(model.body[1]));
                append_floats(blk, pack_bias(model.body[1]));
                VkDeviceSize off = 0, sz = 0;
                append_ubo_block(blk, off, sz);
                const std::string src_in = shader.gen_in_fused2_3py(C, C, true);
                VkPipeline p_in = compile_pipeline(src_in.c_str());
                fast_passes.push_back(FastPass{p_in, off, sz, 32, 12, -1, 0});
            }
            // High: FP16 32x12 / FP32 14x14 tail. VeryHigh: 3py body + head.
            const std::size_t tail_fused = is_high ? 1 : 0;
            const std::size_t mid_end = model.body.size() - tail_fused;
            const VkPipeline p_1l = compile_pipeline(shader.gen_1L_3py(C).c_str());
            int cur_slice = 0;
            for (std::size_t i = 2; i < mid_end; ++i) {
                std::vector<float> blk;
                append_floats(blk, pack_nc4(model.body[i]));
                append_floats(blk, pack_bias(model.body[i]));
                VkDeviceSize off = 0, sz = 0;
                append_ubo_block(blk, off, sz);
                const int next_slice = cur_slice ^ 1;
                fast_passes.push_back(FastPass{p_1l, off, sz, mid_tw, mid_th, cur_slice, next_slice});
                cur_slice = next_slice;
            }
            // Final pass: High fused tail, or VeryHigh standalone head.
            {
                std::vector<float> blk;
                if (tail_fused) {
                    append_floats(blk, pack_nc4(model.body.back()));
                    append_floats(blk, pack_bias(model.body.back()));
                }
                append_head(blk);
                VkDeviceSize off = 0, sz = 0;
                append_ubo_block(blk, off, sz);
                const bool head_3py = !tail_fused && aa;
                const std::string src_end = tail_fused
                    ? (half_compute ? shader.gen_tail_32x12(12, aa) : shader.gen_tail_14x14(12, aa))
                    : (head_3py ? shader.gen_head_end_3py(C) : shader.gen_head_end(C, false));
                const std::uint32_t end_tw = tail_fused ? (half_compute ? 32 : 14) : (head_3py ? 32 : 16);
                const std::uint32_t end_th = tail_fused ? (half_compute ? 12 : 14) : (head_3py ? 12 : 16);
                VkPipeline p_end = compile_pipeline(src_end.c_str());
                fast_passes.push_back(FastPass{p_end, off, sz, end_tw, end_th, cur_slice, -2});
            }
        }

        if (slice_channels > 0) {
            const std::uint32_t g_count = static_cast<std::uint32_t>(slice_channels / 4);
            if (static_cast<std::uint64_t>(width) * height >
                std::numeric_limits<std::uint32_t>::max() / g_count)
                throw std::invalid_argument("Vulkan feature plane stride overflow");
            VkDeviceSize cursor = 0;
            const VkDeviceSize s0 = add_slice(width, height, slice_channels, cursor);
            const VkDeviceSize s1 = add_slice(width, height, slice_channels, cursor);
            const VkDeviceSize slice_bytes = static_cast<VkDeviceSize>(width) * height * slice_channels * (half_compute ? 2 : 4);
            fast_slices[0] = {VK_NULL_HANDLE, s0, slice_bytes};
            fast_slices[1] = {VK_NULL_HANDLE, s1, slice_bytes};
            workspace_size = cursor;
        }

        upload_parameters(ubo_data, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    void upload_parameters(const std::vector<float> &values, VkBufferUsageFlags usage) {
        const bool half_uniform = half_compute && (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        const VkDeviceSize bytes = align_up(
            static_cast<VkDeviceSize>(values.size()) * (half_compute && !half_uniform ? 2 : 4), 4);
        const auto max_range = (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ?
            limits.maxUniformBufferRange : limits.maxStorageBufferRange;
        if (!fast_path && bytes > max_range)
            throw std::runtime_error("parameters exceed Vulkan buffer range");
        parameters = buffer(bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, parameter_info);
        if (half_uniform) {
            std::vector<std::uint32_t> packed(values.size(), 0);
            for (std::size_t i = 0; i < values.size(); ++i)
                packed[(i / 4) * 4 + (i % 4) / 2] |=
                    std::uint32_t(float_to_half(values[i])) << ((i % 2) * 16);
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
            // R82 enables reported shaderFloat16, and requires storageBuffer16BitAccess.
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
            init_fallback(model); // Same precision as the fused path, never a promotion.
        if (workspace_size > 0)
            ws_slots.resize(kWorkspaceSlots);
    }

    void barrier(VkCommandBuffer cmd) const {
        VkMemoryBarrier2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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
                       VkDescriptorType param_type, const VkDescriptorBufferInfo &aux) const {
        const std::array<VkDescriptorBufferInfo, 4> buffers{src, dst, params, aux};
        std::array<VkWriteDescriptorSet, 4> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = (i == 2) ? param_type : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vk->vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
                                   static_cast<std::uint32_t>(writes.size()), writes.data());
    }

    template <typename PushT>
    void dispatch_kernel(VkCommandBuffer cmd, const PushT &push, const VkDescriptorBufferInfo &src,
                         const VkDescriptorBufferInfo &dst, const VkDescriptorBufferInfo &params,
                         VkDescriptorType param_type, const VkDescriptorBufferInfo &aux,
                         std::uint32_t gx, std::uint32_t gy, std::uint32_t gz, bool need_barrier) const {
        push_bindings(cmd, src, dst, params, param_type, aux);
        VkPushConstantsInfo constants{};
        constants.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO;
        constants.layout = layout;
        constants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        constants.size = sizeof(push);
        constants.pValues = &push;
        vk->vkCmdPushConstants2(cmd, &constants);
        vk->vkCmdDispatch(cmd, gx, gy, gz);
        if (need_barrier)
            barrier(cmd);
    }

    struct SlotLease {
        const Impl &impl;
        WorkspaceSlot *slot = nullptr;

        explicit SlotLease(const Impl &i) : impl(i) {}
        SlotLease(const SlotLease &) = delete;
        SlotLease &operator=(const SlotLease &) = delete;

        void handoff_locked() {
            if (impl.ws_wait_head) {
                WaitNode *next = impl.ws_wait_head;
                impl.ws_wait_head = next->next;
                if (!impl.ws_wait_head)
                    impl.ws_wait_tail = nullptr;
                next->next = nullptr;
                next->slot = slot;
                slot = nullptr;
                next->cv.notify_one();
                return;
            }
            slot->busy = false;
            slot = nullptr;
        }

        ~SlotLease() {
            if (slot) {
                std::lock_guard lock(impl.ws_mutex);
                handoff_locked();
            }
        }

        VSVulkanBufferInfo acquire() {
            if (impl.workspace_size == 0)
                return {};
            std::uint64_t wait_val = 0;
            {
                std::unique_lock lock(impl.ws_mutex);
                if (!impl.ws_wait_head) {
                    std::uint64_t best_sig = std::numeric_limits<std::uint64_t>::max();
                    for (auto &candidate : impl.ws_slots) {
                        if (!candidate.busy && candidate.signaled <= best_sig) {
                            best_sig = candidate.signaled;
                            slot = &candidate;
                        }
                    }
                }
                if (slot) {
                    slot->busy = true;
                } else {
                    WaitNode node;
                    if (impl.ws_wait_tail) {
                        impl.ws_wait_tail->next = &node;
                    } else {
                        impl.ws_wait_head = &node;
                    }
                    impl.ws_wait_tail = &node;
                    node.cv.wait(lock, [&] { return node.slot != nullptr; });
                    slot = node.slot;
                }
                wait_val = slot->signaled;
            }
            if (!slot->buf) {
                slot->buf = impl.buffer(impl.workspace_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, slot->info);
            } else if (wait_val != 0) {
                char error[1024]{};
                if (impl.gpu->gpuExecWaitValue(impl.pool, wait_val, error, sizeof(error)) != gdDrained)
                    throw std::runtime_error(std::string("GPU workspace wait: ") + error);
            }
            return slot->info;
        }

        void release(std::uint64_t new_val) {
            if (!slot)
                return;
            std::lock_guard lock(impl.ws_mutex);
            slot->signaled = new_val;
            handoff_locked();
        }
    };

    void process(const VSFrame *source, VSFrame *output) const {
        std::uint32_t source_stride = 0, output_stride = 0;
        auto input = plane(source, width, height, source_stride);
        auto destination = plane(output, width * 2, height * 2, output_stride);
        SlotLease lease(*this);
        const VSVulkanBufferInfo workspace_info = lease.acquire();

        char error[1024]{};
        Exec exec{gpu, gpu->gpuExecAcquire(pool, error, sizeof(error))};
        if (!exec.context)
            throw std::runtime_error(std::string("GPU frame acquire: ") + error);
        gpu->gpuExecReadsFrame(exec.context, source);
        gpu->gpuExecWritesPlane(exec.context, output, 0);
        const VkCommandBuffer cmd = gpu->gpuExecCommandBuffer(exec.context);

        const auto w = static_cast<std::uint32_t>(width);
        const auto h = static_cast<std::uint32_t>(height);

        if (fast_path) {
            auto local = fast_slices;
            for (auto &slice : local)
                slice.buffer = workspace_info.buffer;
            const FastPush push{w, h, w * h, source_stride, output_stride, w * 2, h * 2, 3u};
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
                dispatch_kernel(cmd, push, src, dst, ubo,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, input,
                                gx, gy, 1, i + 1 < fast_passes.size());
            }
        } else {
            auto local = fallback_slices;
            for (auto &slice : local)
                slice.buffer = workspace_info.buffer;
            const VkDescriptorBufferInfo ssbo{parameter_info.buffer, 0, parameter_info.size};
            vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fallback_pipeline);
            const auto run_fb = [&](FallbackPush push, const VkDescriptorBufferInfo &src,
                                    const VkDescriptorBufferInfo &dst, const VkDescriptorBufferInfo &aux,
                                    int ow, int oh, std::uint32_t groups, bool bar) {
                dispatch_kernel(cmd, push, src, dst, ssbo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, aux,
                                (static_cast<std::uint32_t>(ow) + 7) / 8,
                                (static_cast<std::uint32_t>(oh) + 7) / 8, groups, bar);
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
        std::uint64_t signaled = 0;
        exec.submit(&signaled);
        lease.release(signaled);
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
