// File-based adapter for the production CPU math used by mpv_shaders.py.
#include "ngu_model.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) try {
    if (argc != 9)
        throw std::invalid_argument(
            "usage: mpv_reference root model-id fp32|fp16 width height bits input.u16 output.f32");
    const ngu::ModelSpec *spec = nullptr;
    for (const auto &candidate : ngu::model_specs())
        if (std::string_view(candidate.id) == argv[2])
            spec = &candidate;
    if (!spec)
        throw std::invalid_argument("unknown native model");
    const std::string precision = argv[3];
    if (precision != "fp16" && precision != "fp32")
        throw std::invalid_argument("unknown weight precision");

    ngu::Tensor src(std::stoi(argv[4]), std::stoi(argv[5]), 1);
    const unsigned bits = std::stoul(argv[6]);
    if (bits != 8 && bits != 10 && bits != 12)
        throw std::invalid_argument("invalid input bit depth");
    const unsigned peak = (1u << bits) - 1u;

    std::ifstream in(argv[7], std::ios::binary);
    for (float &v : src.values) {
        unsigned char bytes[2];
        if (!in.read(reinterpret_cast<char *>(bytes), 2))
            throw std::runtime_error("truncated input plane");
        const unsigned word = bytes[0] | (unsigned(bytes[1]) << 8);
        if (word > peak)
            throw std::runtime_error("out-of-range input sample");
        v = float(word) / float(peak);
    }
    if (in.peek() != std::char_traits<char>::eof() || in.bad())
        throw std::runtime_error("extra input data or I/O error");

    const auto model = ngu::load_model(argv[1], *spec, precision == "fp16" ?
        ngu::WeightPrecision::FP16 : ngu::WeightPrecision::FP32);
    const auto dst = ngu::run(model, src);
    std::ofstream out(argv[8], std::ios::binary);
    for (float v : dst.values) {
        const auto word = std::bit_cast<std::uint32_t>(v);
        for (int j = 0; j < 4; ++j)
            out.put(static_cast<char>((word >> (8 * j)) & 0xffu));
    }
    out.close();
    if (!out)
        throw std::runtime_error("write failed");
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
