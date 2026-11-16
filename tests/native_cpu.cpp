#include "ngu_model.hpp"
#include "ngu_half.hpp"
#include "half_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "expected an exception");
}
void close(float a, float b, float tolerance = 2e-5f) {
    check(std::isfinite(a) && std::isfinite(b) &&
        std::abs(a-b) <= tolerance + tolerance * std::abs(b), "numerical mismatch");
}

void test_half() {
    check(ngu::half_to_float(1) == 0x1p-24f, "half subnormal decode");
    check(ngu::half_to_float(0x400) == 0x1p-14f, "half minimum normal decode");
    check(ngu::half_to_float(0x3c00) == 1.0f && ngu::half_to_float(0x7bff) == 65504.0f,
          "half normal decode");
    for (unsigned h = 0; h <= 0xffff; ++h) {
        const float value = ngu::half_to_float(static_cast<std::uint16_t>(h));
        if ((h & 0x7c00u) != 0x7c00u)
            check(ngu::float_to_half(value) == h, "finite half round trip");
        else
            check((h & 0x3ffu) ? std::isnan(value) : std::isinf(value), "half NaN/Inf decode");
    }
    const auto inf = std::numeric_limits<float>::infinity();
    for (unsigned h = 0; h < 0x7bff; ++h) {
        const float lo = ngu::half_to_float(static_cast<std::uint16_t>(h));
        const float hi = ngu::half_to_float(static_cast<std::uint16_t>(h + 1));
        const float midpoint = (lo + hi) * 0.5f;
        const auto even = h + (h & 1u);
        check(ngu::float_to_half(midpoint) == even, "half positive tie must round to even");
        check(ngu::float_to_half(-midpoint) == (even | 0x8000u), "half negative tie must round to even");
        check(ngu::float_to_half(std::nextafter(midpoint, -inf)) == h, "half below midpoint");
        check(ngu::float_to_half(std::nextafter(midpoint, inf)) == h + 1, "half above midpoint");
    }
    check(ngu::float_to_half(-0.0f) == 0x8000 && std::signbit(ngu::round_half(-0.0f)), "half signed zero");
    check(ngu::float_to_half(65520.0f) == 0x7c00 &&
        ngu::float_to_half(std::nextafter(65520.0f, 0.0f)) == 0x7bff, "half overflow rounding");
    check(ngu::float_to_half(inf) == 0x7c00 && ngu::float_to_half(-inf) == 0xfc00, "half infinity");
    check(std::isnan(ngu::round_half(std::numeric_limits<float>::quiet_NaN())), "half NaN");

    std::vector<std::uint16_t> plane{0x8000, 1, 0x3c01, 0x7e00, 0x7e00,
                                    0x7bff, 0xbc00, 0x400, 0x7e00, 0x7e00};
    const auto before = plane;
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(plane.data());
    const auto input = ngu::read_plane(bytes, 10, 3, 2, 16);
    check(input.values[1] == 0x1p-24f && input.values[3] == 65504.0f, "half input values");
    std::vector<std::uint16_t> out(10, 0xdead);
    ngu::write_plane(input, reinterpret_cast<std::uint8_t *>(out.data()), 10, 16);
    check(out[0] == 0x8000 && out[1] == 1 && out[2] == 0x3c01 && out[3] == 0xdead &&
        out[5] == 0x7bff && out[6] == 0xbc00 && out[8] == 0xdead, "half stride/padding");
    const auto reverse = ngu::read_plane(bytes + 10, -10, 3, 2, 16);
    ngu::write_plane(reverse, reinterpret_cast<std::uint8_t *>(out.data() + 5), -10, 16);
    check(out[0] == 0x8000 && out[5] == 0x7bff && plane == before, "half negative stride/source mutation");
    rejects([&]{ (void)ngu::read_plane(bytes, 4, 3, 2, 16); });
    rejects([&]{ (void)ngu::read_plane(bytes, 10, 3, 2, 8); });
    rejects([&]{ (void)ngu::read_plane(bytes + 6, 10, 1, 1, 16); }); // NaN
    plane[3] = 0x7c00;
    rejects([&]{ (void)ngu::read_plane(bytes + 6, 10, 1, 1, 16); }); // Inf
    auto overflow = input;
    overflow.values[0] = 65520.0f;
    rejects([&]{ ngu::write_plane(overflow, reinterpret_cast<std::uint8_t *>(out.data()), 10, 16); });

    std::vector<float> weights(9, 1.0007f);
    ngu::Layer quantized(1, 1, 3, weights, {0.12345679f}, ngu::WeightPrecision::FP16);
    check(quantized.weight_at(0) == ngu::round_half(weights[0]), "half weight quantization");
    check(quantized.bias_at(0) == ngu::round_half(0.12345679f / 255.0f), "scale bias before half rounding");
    weights[0] = 65520.0f;
    rejects([&]{ ngu::Layer bad(1, 1, 3, weights, {0}, ngu::WeightPrecision::FP16); });
    weights[0] = 0;
    rejects([&]{ ngu::Layer bad(1, 1, 3, weights, {65520.0f * 255.0f}, ngu::WeightPrecision::FP16); });
}

void test_storage() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> padded{-.125f, 1.25f, nan, nan, .33333334f, .12345679f, nan, nan};
    const auto before = padded;
    auto input = ngu::read_plane(reinterpret_cast<const std::uint8_t *>(padded.data()), 16, 2, 2);
    check(input.values == std::vector<float>({-.125f, 1.25f, .33333334f, .12345679f}), "input conversion/clamp");
    std::vector<float> output(8, -999.0f);
    ngu::write_plane(input, reinterpret_cast<std::uint8_t *>(output.data()), 16);
    check(output[0] == -.125f && output[1] == 1.25f && output[2] == -999.0f &&
        output[3] == -999.0f && output[4] == .33333334f && output[6] == -999.0f, "output stride/padding");
    check(std::memcmp(padded.data(), before.data(), padded.size()*4) == 0, "mutated input");
    const auto reverse = ngu::read_plane(reinterpret_cast<const std::uint8_t *>(padded.data()+4), -16, 2, 2);
    check(reverse.values[0] == input.values[2] && reverse.values[2] == input.values[0], "negative stride");
    rejects([&]{ (void)ngu::read_plane(nullptr, 16, 2, 2); });
    rejects([&]{ (void)ngu::read_plane(reinterpret_cast<const std::uint8_t *>(padded.data()), 4, 2, 2); });
    rejects([&]{ (void)ngu::read_plane(reinterpret_cast<const std::uint8_t *>(padded.data()), 16, 3, 2); });
    rejects([&]{ ngu::write_plane(input, nullptr, 16); });
    rejects([]{ ngu::validate_2x_dimensions(0, 2); });
    rejects([]{ ngu::validate_2x_dimensions(std::numeric_limits<int>::max(), 2); });
    rejects([]{ (void)ngu::checked_elements(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 16); });
    ngu::validate_2x_dimensions(1920, 1080); // validation only, no allocation
}

void test_correlation_and_aa() {
    ngu::Tensor source(3, 1, 1);
    source.values = {1, 2, 3};
    std::vector<float> weights(9, 0);
    weights[3] = -1;
    ngu::Layer ordinary(1, 1, 3, weights, {0});
    const auto hidden = ngu::correlate(source, ordinary);
    close(hidden.values[1], -.1f, 1e-7f);
    close(hidden.values[2], -.2f, 1e-7f);
    const auto linear = ngu::correlate(source, ordinary, false);
    check(linear.values[2] == -2, "output head must not activate");

    ngu::Tensor features(1, 1, 8);
    features.values = {.125f, .25f, .375f, 0, 0, 0, 0, 0};
    std::vector<ngu::Layer> heads;
    for (int h = 0; h < 3; ++h) {
        std::vector<float> w(72, 0);
        w[h*9+4] = static_cast<float>(h+1);
        heads.emplace_back(8, 1, 3, w, std::vector<float>{255.0f * (h+1)});
    }
    const auto packed = ngu::correlate(features, ngu::make_aa_head(heads), false);
    check(packed.values == std::vector<float>({1.5f, 2.125f, 4.125f, 0}), "AA crossed weight/uncrossed bias");
    ngu::Tensor pixel(1, 1, 1);
    pixel.values = {.5f};
    check(ngu::aa_combine(pixel, packed).values == std::vector<float>({.5f, 2.625f, 2, 4.625f}),
        "AA phase combine");

    ngu::Tensor extended(2, 1, 1), zeros(2, 1, 4);
    extended.values = {-.12345679f, 1.2345679f};
    const auto result = ngu::aa_combine(extended, zeros);
    check(result.values[0] == extended.values[0] && result.values[2] == extended.values[1],
        "AA even-even must preserve extended-range FP32 source");
    weights[0] = std::numeric_limits<float>::infinity();
    rejects([&]{ ngu::Layer invalid(1, 1, 3, weights, {0}); });
    rejects([&]{ (void)ngu::make_aa_head({}); });
}

void test_baseline() {
    ngu::Tensor ramp(5, 1, 1);
    ramp.values = {0, .25f, .5f, .75f, 1};
    const auto x = ngu::baseline_pass(ramp, true);
    close(x.values[0], -.02109375f, 1e-7f);
    close(x.values[1], .04609375f, 1e-7f);
    check(x.values.front() < 0 && x.values.back() > 1, "baseline was clamped");
    ngu::Tensor constant(1, 1, 1);
    constant.values = {.375f};
    for (const bool horizontal : {false, true})
        for (float v : ngu::baseline_pass(constant, horizontal).values)
            close(v, .375f, 1e-7f);
}

void test_packed_transpose() {
    for (const auto size : {std::pair{1,1}, {1,5}, {5,1}, {5,3}, {6,4}}) {
        const auto [w,h] = size;
        ngu::Tensor features(w, h, 8);
        for (std::size_t i = 0; i < features.values.size(); ++i)
            features.values[i] = static_cast<float>(static_cast<int>((i*37)%101)-30) / 100.0f;
        std::vector<float> weights(8*16);
        for (std::size_t i = 0; i < weights.size(); ++i)
            weights[i] = static_cast<float>(static_cast<int>((i*13)%79)-40) / 127.0f;
        ngu::Layer head(8, 1, 4, weights, {31.0f});
        const auto packed = ngu::packed_residual(features, head);
        ngu::Tensor zeros(w*2, h*2, 1);
        const auto output = ngu::residual_combine(packed, zeros);
        for (int y = 0; y < h*2; ++y)
            for (int x = 0; x < w*2; ++x) {
                // Independent direct stride-2 formula, not packed addressing.
                double expected = head.bias()[0];
                for (int ic = 0; ic < 8; ++ic)
                    for (int ky = (y+1)%2; ky < 4; ky += 2)
                        for (int kx = (x+1)%2; kx < 4; kx += 2)
                            expected += static_cast<double>(features.at((x+1-kx)/2, (y+1-ky)/2, ic)) *
                                weights[ic*16+ky*4+kx];
                close(output.at(x,y,0), static_cast<float>(expected));
            }
    }
}

void test_archive(const char *root) {
    check(ngu::model_specs().size() == 72, "native-2x catalog must cover exactly 72 models");
    for (const auto &spec : ngu::model_specs()) {
        check(std::string(spec.id).find("_4x_") == std::string::npos, "runtime depends on 4x");
        const auto model = ngu::load_model(root, spec);
        ngu::Tensor source(3, 2, 1);
        source.values = {-.5f, 0, .12345679f, .75f, 1, 1.5f};
        int observed = 0;
        const auto before = source.values;
        const auto out = ngu::run(model, source, [&](auto, const auto &t) {
            t.require_finite("observed");
            ++observed;
        });
        check(out.width == 6 && out.height == 4 && out.channels == 1, "output dimensions");
        check(observed == static_cast<int>(model.body.size()) + (model.aa ? 2 : 4), "missing FP32 storage stage");
        check(source.values == before, "executor mutated source");
        if (model.aa) {
            for (int y = 0; y < source.height; ++y)
                for (int x = 0; x < source.width; ++x)
                    check(out.at(x*2,y*2,0) == source.at(x,y,0), "AA native phase");
        }
        const auto half = ngu::load_model(root, spec, ngu::WeightPrecision::FP16);
        check(half.weight_precision == ngu::WeightPrecision::FP16, "half model precision");
        const auto check_layer = [](const ngu::Layer &full, const ngu::Layer &reduced) {
            check(reduced.weights().empty() && reduced.bias().empty() && reduced.interleaved().empty(),
                  "FP16 model must not retain FP32 parameter arrays");
            check(reduced.storage_bytes() * 2 == full.storage_bytes(), "FP16 model payload must be half size");
            for (std::size_t i = 0; i < full.weights().size(); ++i)
                check(reduced.half_weights()[i] == ngu::float_to_half(full.weights()[i]), "archive half weight bits");
            for (std::size_t i = 0; i < full.bias().size(); ++i)
                check(reduced.half_bias()[i] == ngu::float_to_half(full.bias()[i]), "archive half scaled bias bits");
        };
        for (std::size_t i = 0; i < model.body.size(); ++i)
            check_layer(model.body[i], half.body[i]);
        check_layer(model.head, half.head);
        const auto half_out = ngu::run(half, source);
        check(half_out.width == out.width && half_out.height == out.height, "half output dimensions");
        if (half.aa)
            check(half_out.at(4, 2, 0) == source.at(2, 1, 0), "half weights changed AA source phase");

        source.values[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&]{ (void)ngu::run(model, source); });
        rejects([&]{ (void)ngu::run(half, source); });
    }
    check(&ngu::select_model("aa", "low", 14) == &ngu::select_model("aa", "low", 0), "option aliases");
    check(ngu::select_model("sharp", "very_high", 14).option == 14, "Sharp option14");
    rejects([]{ (void)ngu::select_model("sharp", "low", -1); });
    rejects([]{ (void)ngu::select_model("sharp", "low", std::numeric_limits<std::int64_t>::max()); });
    rejects([]{ (void)ngu::select_model("invalid", "low", 0); });
    rejects([]{ (void)ngu::select_model("sharp", "invalid", 0); });
}
} // namespace

int main(int argc, char **argv) {
    try {
        check(argc == 1 || argc == 2, "usage: ngu_native_tests [weights-root | --half-fma]");
        if (argc == 2 && std::strcmp(argv[1], "--half-fma") == 0) {
            unsigned a, b, c;
            while (std::cin >> a >> b >> c)
                std::cout << ngu::float_to_half(half_reference::fma(
                    ngu::half_to_float(a), ngu::half_to_float(b), ngu::half_to_float(c))) << '\n';
            return 0;
        }
        test_half();
        test_storage();
        test_correlation_and_aa();
        test_baseline();
        test_packed_transpose();
        if (argc == 2) test_archive(argv[1]);
        std::cout << "native FP32 / FP16 storage and weight tests passed" << (argc == 2 ? " (all 72 FP32 / native FP16 archived 2x models)" : "") << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "native test: " << e.what() << "\n";
        return 1;
    }
}
