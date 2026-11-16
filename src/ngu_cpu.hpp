#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "ngu_half.hpp"

namespace ngu {

enum class WeightPrecision { FP32, FP16 };

// CPU reference tensors and accumulators are binary32. FP16 coefficients stay
// binary16 in the model and are decoded only when consumed by arithmetic.
struct Tensor {
    int width;
    int height;
    int channels;
    std::vector<float> values;

    Tensor(int width, int height, int channels);
    float at(int x, int y, int channel) const noexcept; // point + clamp
    void require_finite(std::string_view stage) const;
};

std::size_t checked_elements(int width, int height, int channels);
void validate_2x_dimensions(int width, int height);
Tensor read_plane(const std::uint8_t *data, std::ptrdiff_t stride, int width, int height, int bits = 32);
void write_plane(const Tensor &source, std::uint8_t *data, std::ptrdiff_t stride, int bits = 32);

// Raw weights remain in output,input,y,x order; bias is uploaded as raw/255.
// Correlation weights are additionally interleaved for vectorisation over output
// channels. Optional FP16 rounding applies to weights and the scaled bias;
// CPU accumulators, activations and baseline coefficients remain binary32.
class Layer {
public:
    Layer(int input_channels, int output_channels, int kernel,
          std::vector<float> raw_weights, std::vector<float> raw_bias,
          WeightPrecision precision = WeightPrecision::FP32);
    // Direct binary16 archive input; biases are already scaled raw/255.
    Layer(int input_channels, int output_channels, int kernel,
          std::vector<std::uint16_t> weights, std::vector<std::uint16_t> scaled_bias);
    int input_channels() const noexcept { return ci_; }
    int output_channels() const noexcept { return co_; }
    int kernel() const noexcept { return kernel_; }
    const std::vector<float> &weights() const noexcept { return weights_; }
    const std::vector<float> &bias() const noexcept { return bias_; }
    const std::vector<float> &interleaved() const noexcept { return interleaved_; }
    WeightPrecision precision() const noexcept { return precision_; }
    const std::vector<std::uint16_t> &half_weights() const noexcept { return half_weights_; }
    const std::vector<std::uint16_t> &half_bias() const noexcept { return half_bias_; }
    const std::vector<std::uint16_t> &half_interleaved() const noexcept { return half_interleaved_; }
    float weight_at(std::size_t i) const noexcept {
        return precision_ == WeightPrecision::FP16 ? half_to_float(half_weights_[i]) : weights_[i];
    }
    float bias_at(std::size_t i) const noexcept {
        return precision_ == WeightPrecision::FP16 ? half_to_float(half_bias_[i]) : bias_[i];
    }
    std::size_t storage_bytes() const noexcept {
        return (weights_.size() + bias_.size() + interleaved_.size()) * sizeof(float) +
            (half_weights_.size() + half_bias_.size() + half_interleaved_.size()) * sizeof(std::uint16_t);
    }

private:
    friend Layer make_aa_head(std::span<const Layer> ordered_heads);
    int ci_;
    int co_;
    int kernel_;
    WeightPrecision precision_;
    void build_interleaved();
    std::vector<std::uint16_t> half_weights_, half_bias_, half_interleaved_;
    std::vector<float> weights_;
    std::vector<float> bias_;
    std::vector<float> interleaved_;
};

Layer make_aa_head(std::span<const Layer> ordered_heads);
Tensor correlate(const Tensor &source, const Layer &layer, bool activation = true);
Tensor baseline_pass(const Tensor &source, bool horizontal);
Tensor packed_residual(const Tensor &source, const Layer &head);
Tensor residual_combine(const Tensor &packed, const Tensor &baseline);
Tensor aa_combine(const Tensor &original, const Tensor &heads);

struct Model {
    std::string id;
    bool aa;
    std::vector<Layer> body;
    Layer head; // AA: XYZ + zero W, already crossed weights/uncrossed bias
    WeightPrecision weight_precision = WeightPrecision::FP32;
};

// Optional, synchronous read-only observation for native mathematical tests.
// No tensor is shared between requests; Model is immutable during execution.
using Observer = std::function<void(std::string_view, const Tensor &)>;
Tensor run(const Model &model, const Tensor &source, const Observer &observe = {});

} // namespace ngu
