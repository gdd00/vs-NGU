#include "ngu_cpu.hpp"
#include "ngu_half.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef __FAST_MATH__
#error NGU requires IEEE FP32, not fast-math
#endif
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(FLT_EVAL_METHOD == 0, "NGU requires FP32 evaluation without excess precision");

namespace ngu {
namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::invalid_argument(message);
}

void check_tensor(const Tensor &t) {
    require(t.values.size() == checked_elements(t.width, t.height, t.channels), "invalid tensor storage");
}

void check_stride(std::ptrdiff_t stride, int width, int height, int bits) {
    require(bits == 16 || bits == 32, "plane must be float16 or float32");
    const auto row_bytes = static_cast<std::size_t>(width) * (bits / 8);
    const auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
    require(stride != std::numeric_limits<std::ptrdiff_t>::min(), "invalid plane stride");
    const auto magnitude = stride < 0 ? -stride : stride;
    require(row_bytes <= static_cast<std::size_t>(maximum) &&
            static_cast<std::size_t>(magnitude) >= row_bytes, "plane stride is too small");
    require(height == 1 || magnitude <= maximum / (height - 1), "plane row offset overflow");
}

void emit(const Observer &observe, std::string_view name, const Tensor &value) {
    value.require_finite(name);
    if (observe)
        observe(name, value);
}
} // namespace

std::size_t checked_elements(int width, int height, int channels) {
    require(width > 0 && height > 0 && channels >= 1 && channels <= 16, "invalid tensor dimensions");
    const auto maximum = std::min(std::vector<float>().max_size(),
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float));
    std::size_t count = static_cast<std::size_t>(width);
    for (const int factor : {height, channels}) {
        require(count <= maximum / static_cast<std::size_t>(factor), "tensor size overflow");
        count *= static_cast<std::size_t>(factor);
    }
    return count;
}

void validate_2x_dimensions(int width, int height) {
    constexpr int maximum = (std::numeric_limits<int>::max() - 2) / 2;
    require(width > 0 && height > 0 && width <= maximum && height <= maximum, "2x dimension overflow");
    (void)checked_elements(width, height, 16);
    (void)checked_elements(width + 2, height + 2, 4);
    (void)checked_elements(width * 2, height * 2, 1);
}

Tensor::Tensor(int w, int h, int c)
    : width(w), height(h), channels(c), values(checked_elements(w, h, c), 0.0f) {}

float Tensor::at(int x, int y, int channel) const noexcept {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return values[(static_cast<std::size_t>(y) * width + x) * channels + channel];
}

void Tensor::require_finite(std::string_view stage) const {
    check_tensor(*this);
    for (float value : values)
        if (!std::isfinite(value))
            throw std::runtime_error(std::string(stage) + ": NaN/Inf (non-finite input or FP32 overflow)");
}

Tensor read_plane(const std::uint8_t *data, std::ptrdiff_t stride, int width, int height, int bits) {
    require(data != nullptr, "CPU float plane required; GPU frames are unsupported");
    (void)checked_elements(width, height, 1);
    check_stride(stride, width, height, bits);
    Tensor result(width, height, 1);
    for (int y = 0; y < height; ++y) {
        const auto *row = data + static_cast<std::ptrdiff_t>(y) * stride;
        auto *out = result.values.data() + static_cast<std::size_t>(y) * width;
        if (bits == 32) {
            std::memcpy(out, row, static_cast<std::size_t>(width) * sizeof(float));
        } else {
            for (int x = 0; x < width; ++x) {
                std::uint16_t sample;
                std::memcpy(&sample, row + static_cast<std::size_t>(x) * 2, 2);
                out[x] = half_to_float(sample);
            }
        }
    }
    result.require_finite("source");
    return result;
}

void write_plane(const Tensor &source, std::uint8_t *data, std::ptrdiff_t stride, int bits) {
    check_tensor(source);
    require(data != nullptr && source.channels == 1, "CPU float destination plane required");
    check_stride(stride, source.width, source.height, bits);
    for (int y = 0; y < source.height; ++y) {
        auto *row = data + static_cast<std::ptrdiff_t>(y) * stride;
        const auto *in = source.values.data() + static_cast<std::size_t>(y) * source.width;
        if (bits == 32) {
            std::memcpy(row, in, static_cast<std::size_t>(source.width) * sizeof(float));
        } else {
            for (int x = 0; x < source.width; ++x) {
                const auto sample = float_to_half(in[x]);
                if ((sample & 0x7c00u) == 0x7c00u)
                    throw std::runtime_error("output: NaN/Inf or FP16 overflow");
                std::memcpy(row + static_cast<std::size_t>(x) * 2, &sample, 2);
            }
        }
    }
}

Layer::Layer(int ci, int co, int k, std::vector<float> weights, std::vector<float> bias,
             WeightPrecision precision)
    : ci_(ci), co_(co), kernel_(k), precision_(precision) {
    require(ci >= 1 && ci <= 16 && co >= 1 && co <= 16 && (k == 3 || k == 4), "invalid layer shape");
    require(weights.size() == static_cast<std::size_t>(ci * co * k * k) &&
            bias.size() == static_cast<std::size_t>(co), "invalid layer parameter extent");
    require(precision == WeightPrecision::FP32 || precision == WeightPrecision::FP16,
            "invalid weight precision");
    for (float v : weights) require(std::isfinite(v), "non-finite weight");
    for (float &v : bias) {
        require(std::isfinite(v), "non-finite bias");
        v /= 255.0f;
    }
    if (precision == WeightPrecision::FP16) {
        for (float v : weights) {
            const auto h = float_to_half(v);
            require((h & 0x7c00u) != 0x7c00u, "FP16 weight overflow");
            half_weights_.push_back(h);
        }
        for (float v : bias) {
            const auto h = float_to_half(v);
            require((h & 0x7c00u) != 0x7c00u, "FP16 scaled bias overflow");
            half_bias_.push_back(h);
        }
    } else {
        weights_ = std::move(weights);
        bias_ = std::move(bias);
    }
    build_interleaved();
}

Layer::Layer(int ci, int co, int k, std::vector<std::uint16_t> weights,
             std::vector<std::uint16_t> bias)
    : ci_(ci), co_(co), kernel_(k), precision_(WeightPrecision::FP16),
      half_weights_(std::move(weights)), half_bias_(std::move(bias)) {
    require(ci >= 1 && ci <= 16 && co >= 1 && co <= 16 && (k == 3 || k == 4), "invalid layer shape");
    require(half_weights_.size() == static_cast<std::size_t>(ci * co * k * k) &&
            half_bias_.size() == static_cast<std::size_t>(co), "invalid FP16 layer parameter extent");
    for (auto h : half_weights_)
        require((h & 0x7c00u) != 0x7c00u, "non-finite FP16 weight");
    for (auto h : half_bias_)
        require((h & 0x7c00u) != 0x7c00u, "non-finite FP16 bias");
    build_interleaved();
}

void Layer::build_interleaved() {
    if (kernel_ != 3) return;
    if (precision_ == WeightPrecision::FP16)
        half_interleaved_.reserve(half_weights_.size());
    else
        interleaved_.reserve(weights_.size());
    for (int tap = 0; tap < 9; ++tap)
        for (int ic = 0; ic < ci_; ++ic)
            for (int oc = 0; oc < co_; ++oc) {
                const auto i = static_cast<std::size_t>((oc * ci_ + ic) * 9 + tap);
                if (precision_ == WeightPrecision::FP16)
                    half_interleaved_.push_back(half_weights_[i]);
                else
                    interleaved_.push_back(weights_[i]);
            }
}

Layer make_aa_head(std::span<const Layer> heads) {
    require(heads.size() == 3, "AA requires three parallel heads");
    const int ci = heads[0].input_channels();
    const auto precision = heads[0].precision();
    for (const auto &head : heads)
        require(head.precision() == precision && head.input_channels() == ci &&
            head.output_channels() == 1 && head.kernel() == 3, "invalid parallel AA head");
    if (precision == WeightPrecision::FP16) {
        std::vector<std::uint16_t> weights(static_cast<std::size_t>(4 * ci * 9), 0u), bias(4, 0u);
        constexpr int order[] = {1, 0, 2};
        for (int lane = 0; lane < 3; ++lane) {
            const auto &head = heads[order[lane]];
            std::copy(head.half_weights().begin(), head.half_weights().end(), weights.begin() + lane * ci * 9);
            bias[lane] = heads[lane].half_bias()[0];
        }
        return Layer(ci, 4, 3, std::move(weights), std::move(bias));
    }
    std::vector<float> weights(static_cast<std::size_t>(4 * ci * 9), 0.0f);
    std::vector<float> raw_bias(4, 0.0f);
    constexpr int order[] = {1, 0, 2};
    for (int lane = 0; lane < 3; ++lane) {
        const Layer &head = heads[order[lane]];
        require(head.input_channels() == ci && head.output_channels() == 1 && head.kernel() == 3,
                "invalid parallel AA head");
        std::copy(head.weights().begin(), head.weights().end(), weights.begin() + lane * ci * 9);
    }
    Layer result(ci, 4, 3, std::move(weights), std::move(raw_bias));
    // These are already raw/255 in FP32. Copy without an inverse transform,
    // a second division, or swapping the biases along with the weights.
    for (int lane = 0; lane < 3; ++lane)
        result.bias_[lane] = heads[lane].bias()[0];
    return result;
}

Tensor correlate(const Tensor &source, const Layer &layer, bool activation) {
    check_tensor(source);
    require(layer.kernel() == 3 && source.channels == layer.input_channels(), "correlation shape mismatch");
    const int ci = source.channels, co = layer.output_channels();
    Tensor result(source.width, source.height, co);
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            std::array<float, 16> sum{};
            for (int oc = 0; oc < co; ++oc) sum[oc] = layer.bias_at(oc);
            const float *weight = layer.interleaved().data();
            const std::uint16_t *half_weight = layer.half_interleaved().data();
            for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx)
                    for (int ic = 0; ic < ci; ++ic) {
                        const float sample = source.at(x + kx - 1, y + ky - 1, ic);
                        // Independent binary32 accumulators, SIMD across OC.
                        // No reassociation over taps and no FP64 accumulation.
                        if (layer.precision() == WeightPrecision::FP16) {
                            for (int oc = 0; oc < co; ++oc)
                                sum[oc] += sample * half_to_float(half_weight[oc]);
                            half_weight += co;
                        } else {
                            for (int oc = 0; oc < co; ++oc)
                                sum[oc] += sample * weight[oc];
                            weight += co;
                        }
                    }
            float *out = result.values.data() + (static_cast<std::size_t>(y) * source.width + x) * co;
            for (int oc = 0; oc < co; ++oc)
                out[oc] = activation && sum[oc] < 0.0f ? sum[oc] * 0.1f : sum[oc];
        }
    }
    result.require_finite("correlation3x3");
    return result;
}

Tensor baseline_pass(const Tensor &source, bool horizontal) {
    check_tensor(source);
    require(source.channels == 1, "baseline requires one plane");
    require((horizontal ? source.width : source.height) <= (std::numeric_limits<int>::max() - 2) / 2,
            "baseline dimension overflow");
    const float a = std::bit_cast<float>(0x3f5f3333U);
    const float b = std::bit_cast<float>(0x3e766666U);
    const float c = std::bit_cast<float>(0xbdaccccdU);
    const float d = std::bit_cast<float>(0xbce66666U);
    const float coefficients[2][4] = {{d, b, a, c}, {c, a, b, d}};
    Tensor result(source.width * (horizontal ? 2 : 1), source.height * (horizontal ? 1 : 2), 1);
    for (int y = 0; y < result.height; ++y)
        for (int x = 0; x < result.width; ++x) {
            const int pos = horizontal ? x : y;
            const int parity = pos & 1;
            const int first = pos / 2 - (parity ? 1 : 2);
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
                sum += coefficients[parity][k] * (horizontal ?
                    source.at(first + k, y, 0) : source.at(x, first + k, 0));
            result.values[static_cast<std::size_t>(y) * result.width + x] = sum;
        }
    result.require_finite("baseline");
    return result;
}

Tensor packed_residual(const Tensor &source, const Layer &head) {
    check_tensor(source);
    require(head.kernel() == 4 && head.output_channels() == 1 &&
            source.channels == head.input_channels(), "transposed output shape mismatch");
    validate_2x_dimensions(source.width, source.height);
    Tensor result(source.width + 2, source.height + 2, 4);
    for (int y = 0; y < result.height; ++y)
        for (int x = 0; x < result.width; ++x) {
            std::array<float, 4> sum;
            sum.fill(head.bias_at(0));
            for (int ic = 0; ic < source.channels; ++ic)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const float v = source.at(x - dx, y - dy, ic);
                        for (int lane = 0; lane < 4; ++lane)
                            sum[lane] += v * head.weight_at(ic * 16 + (lane / 2 + dy * 2) * 4 + lane % 2 + dx * 2);
                    }
            std::copy(sum.begin(), sum.end(),
                result.values.begin() + (static_cast<std::size_t>(y) * result.width + x) * 4);
        }
    result.require_finite("packed_residual");
    return result;
}

Tensor residual_combine(const Tensor &packed, const Tensor &baseline) {
    check_tensor(packed);
    check_tensor(baseline);
    require(packed.width >= 3 && packed.height >= 3, "packed dimensions too small");
    validate_2x_dimensions(packed.width - 2, packed.height - 2);
    require(packed.channels == 4 && baseline.channels == 1 &&
            baseline.width == (packed.width - 2) * 2 && baseline.height == (packed.height - 2) * 2,
            "residual combine shape mismatch");
    Tensor result(baseline.width, baseline.height, 1);
    for (int y = 0; y < result.height; ++y)
        for (int x = 0; x < result.width; ++x) {
            // Cubicd20: even-even W, odd-even Z, even-odd Y, odd-odd X.
            const int lane = 3 - (x & 1) - 2 * (y & 1);
            const std::size_t i = static_cast<std::size_t>(y) * result.width + x;
            result.values[i] = baseline.values[i] + packed.at((x + 1) / 2, (y + 1) / 2, lane);
        }
    result.require_finite("output");
    return result;
}

Tensor aa_combine(const Tensor &source, const Tensor &heads) {
    check_tensor(source);
    check_tensor(heads);
    require(source.channels == 1 && heads.channels == 4 &&
            source.width == heads.width && source.height == heads.height, "AA combine shape mismatch");
    validate_2x_dimensions(source.width, source.height);
    Tensor result(source.width * 2, source.height * 2, 1);
    for (int y = 0; y < result.height; ++y)
        for (int x = 0; x < result.width; ++x) {
            const int ix = x / 2, iy = y / 2;
            float v = source.at(ix, iy, 0);
            if ((x & 1) && (y & 1)) {
                v = ((v + source.at(ix + 1, iy, 0)) + source.at(ix, iy + 1, 0)) +
                    source.at(ix + 1, iy + 1, 0);
                v = v * 0.25f + heads.at(ix, iy, 2);
            } else if (x & 1) {
                v = (v + source.at(ix + 1, iy, 0)) * 0.5f + heads.at(ix, iy, 1);
            } else if (y & 1) {
                v = (v + source.at(ix, iy + 1, 0)) * 0.5f + heads.at(ix, iy, 0);
            }
            result.values[static_cast<std::size_t>(y) * result.width + x] = v;
        }
    result.require_finite("output");
    return result;
}

Tensor run(const Model &model, const Tensor &source, const Observer &observe) {
    check_tensor(source);
    require(source.channels == 1 && !model.body.empty(), "invalid native-2x graph");
    validate_2x_dimensions(source.width, source.height);
    source.require_finite("source");
    Tensor features = source;
    std::size_t index = 0;
    for (const Layer &layer : model.body) {
        features = correlate(features, layer);
        emit(observe, "features" + std::to_string(index++), features);
    }
    if (model.aa) {
        const Tensor heads = correlate(features, model.head, false);
        emit(observe, "packed_head", heads);
        Tensor output = aa_combine(source, heads);
        emit(observe, "output", output);
        return output;
    }
    Tensor base = [&] {
        const Tensor bx = baseline_pass(source, true);
        emit(observe, "baseline_x", bx);
        return baseline_pass(bx, false);
    }();
    emit(observe, "baseline_y", base);
    const Tensor packed = packed_residual(features, model.head);
    emit(observe, "packed_head", packed);
    Tensor output = residual_combine(packed, base);
    emit(observe, "output", output);
    return output;
}

} // namespace ngu
