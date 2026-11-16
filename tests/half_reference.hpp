#pragma once
#include "ngu_cpu.hpp"
#include "ngu_half.hpp"

namespace half_reference {

// Test-only scalar oracle: binary16 fused multiply-add for convolutions;
// coefficients, activations, tensors and separate baseline/combine operations
// are rounded to half. Wider host arithmetic only emulates half, never GPU inference.
inline float h(float x) { return ngu::round_half(x); }
inline float add(float a, float b) { return h(a + b); }
inline float mul(float a, float b) { return h(a * b); }

// Round directly from binary64: rounding through float can double-round a half
// FMA at a midpoint. Half operands/products are exact in double; the only
// rounding relevant to a finite half result is this explicit ties-to-even step.
inline float from_double(double value) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 48) & 0x8000u);
    const int e = static_cast<int>((bits >> 52) & 0x7ffu) - 1023;
    if (e == 1024) return h(static_cast<float>(value)); // NaN / infinity
    if ((bits & 0x7fffffffffffffffull) == 0 || e < -25)
        return ngu::half_to_float(sign);
    if (e > 15)
        return ngu::half_to_float(static_cast<std::uint16_t>(sign | 0x7c00u));
    const auto mantissa = (bits & 0xfffffffffffffull) | (1ull << 52);
    const int shift = e < -14 ? 28 - e : 42;
    auto rounded = mantissa >> shift;
    const auto remainder = mantissa & ((1ull << shift) - 1);
    const auto midpoint = 1ull << (shift - 1);
    rounded += remainder > midpoint || (remainder == midpoint && (rounded & 1));
    const auto magnitude = static_cast<std::uint16_t>(e < -14 ? rounded : (e + 14) * 1024 + rounded);
    return ngu::half_to_float(static_cast<std::uint16_t>(sign | magnitude));
}
inline float fma(float a, float b, float c) {
    return from_double(static_cast<double>(a) * static_cast<double>(b) + static_cast<double>(c));
}
enum class Order { Spatial, ColumnGroupRows };

inline ngu::Tensor conv(const ngu::Tensor &in, const ngu::Layer &l, bool activation, Order order) {
    ngu::Tensor out(in.width, in.height, l.output_channels());
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x)
            for (int oc = 0; oc < l.output_channels(); ++oc) {
                float v = h(l.bias_at(oc));
                const auto tap = [&](int ky, int kx, int ic) {
                    v = fma(in.at(x+kx-1, y+ky-1, ic),
                        h(l.weight_at((oc*in.channels+ic)*9+ky*3+kx)), v);
                };
                if (order == Order::ColumnGroupRows) {
                    for (int kx = 0; kx < 3; ++kx)
                        for (int group = 0; group < in.channels; group += 4)
                            for (int ky = 0; ky < 3; ++ky)
                                for (int c = 0; c < 4; ++c) tap(ky, kx, group+c);
                } else {
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx)
                            for (int ic = 0; ic < in.channels; ++ic) tap(ky, kx, ic);
                }
                if (activation && v < 0) v = mul(v, h(0.1f));
                out.values[(y*in.width+x)*out.channels+oc] = v;
            }
    return out;
}

inline ngu::Tensor baseline(const ngu::Tensor &in, bool horizontal) {
    ngu::Tensor out(in.width*(horizontal?2:1), in.height*(horizontal?1:2), 1);
    const float ca=h(0.871875f), cb=h(0.240625f), cc=h(-0.084375f), cd=h(-0.028125f);
    for (int y=0; y<out.height; ++y)
        for (int x=0; x<out.width; ++x) {
            const int pos=horizontal?x:y, first=pos/2-((pos&1)?1:2);
            const float coeffs[2][4]={{cd,cb,ca,cc},{cc,ca,cb,cd}};
            float v=0;
            for (int k=0; k<4; ++k)
                v=add(v,mul(coeffs[pos&1][k],in.at(horizontal?first+k:x,horizontal?y:first+k,0)));
            out.values[y*out.width+x]=v;
        }
    return out;
}

inline ngu::Tensor run(const ngu::Model &model, ngu::Tensor source, bool fused = true) {
    for (float &v:source.values) v=h(v);
    auto features=source;
    // Numerical traversal used by the current fused schedules, independently
    // accessing the original scalar model (no GPU weight packing/tile indices).
    const auto column = Order::ColumnGroupRows, spatial = Order::Spatial;
    const auto body_order = [&](std::size_t i) {
        if (!fused || model.body.size() == 3 || i == 0) return spatial;
        if (model.body.size() == 5) return i % 2 ? column : spatial;
        if (model.body.size() == 6 && i == 5) return spatial;
        return column;
    };
    for (std::size_t i=0; i<model.body.size(); ++i)
        features=conv(features,model.body[i],true,body_order(i));
    const auto head_order = fused && (model.body.size()==5 || model.body.size()==6 ||
        (model.body.size()==8 && model.aa)) ? column : spatial;
    ngu::Tensor out(source.width*2,source.height*2,1);
    if (model.aa) {
        const auto heads=conv(features,model.head,false,head_order);
        for (int y=0;y<out.height;++y)
            for (int x=0;x<out.width;++x) {
                const int ix=x/2,iy=y/2;
                float v=source.at(ix,iy,0);
                if ((x&1)&&(y&1)) {
                    v=add(add(add(v,source.at(ix+1,iy,0)),source.at(ix,iy+1,0)),source.at(ix+1,iy+1,0));
                    v=add(mul(v,0.25f),heads.at(ix,iy,2));
                } else if (x&1) {
                    v=add(mul(add(v,source.at(ix+1,iy,0)),0.5f),heads.at(ix,iy,1));
                } else if (y&1) {
                    v=add(mul(add(v,source.at(ix,iy+1,0)),0.5f),heads.at(ix,iy,0));
                }
                out.values[y*out.width+x]=v;
            }
    } else {
        const auto base=baseline(baseline(source,true),false);
        for (int y=0;y<out.height;++y)
            for (int x=0;x<out.width;++x) {
                float v=h(model.head.bias_at(0));
                if (fused) {
                    // Gather transpose taps directly from output/source geometry,
                    // independent of the GPU's packed 3x3 NC4 addressing.
                    const auto tap = [&](int dy,int dx,int ic) {
                        const int ky=(y&1)+1-2*dy, kx=(x&1)+1-2*dx;
                        const float w=ky>=0&&ky<4&&kx>=0&&kx<4 ?
                            h(model.head.weight_at(ic*16+ky*4+kx)) : 0;
                        v=fma(features.at(x/2+dx,y/2+dy,ic),w,v);
                    };
                    if (head_order == column) {
                        for (int dx=-1;dx<=1;++dx)
                            for (int group=0;group<features.channels;group+=4)
                                for (int dy=-1;dy<=1;++dy)
                                    for (int c=0;c<4;++c) tap(dy,dx,group+c);
                    } else {
                        for (int dy=-1;dy<=1;++dy)
                            for (int dx=-1;dx<=1;++dx)
                                for (int ic=0;ic<features.channels;++ic) tap(dy,dx,ic);
                    }
                } else {
                    // Scalar fallback preserves its original ic,ky,kx order.
                    for (int ic=0;ic<features.channels;++ic)
                        for (int ky=(y+1)%2;ky<4;ky+=2)
                            for (int kx=(x+1)%2;kx<4;kx+=2)
                                v=fma(features.at((x+1-kx)/2,(y+1-ky)/2,ic),
                                    h(model.head.weight_at(ic*16+ky*4+kx)),v);
                }
                out.values[y*out.width+x]=add(base.values[y*out.width+x],v);
            }
    }
    return out;
}
} // namespace half_reference
