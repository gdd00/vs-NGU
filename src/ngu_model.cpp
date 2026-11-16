#include "ngu_model.hpp"
#include "ngu_catalog.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <openssl/evp.h>
#include <stdexcept>
#include <utility>

namespace ngu {
namespace {
std::vector<unsigned char> load_bytes(const std::filesystem::path &root, const FileSpec &spec) {
    const std::filesystem::path path = root / spec.path;
    const std::string label = path.string();
    if (spec.bytes == 0 || spec.bytes > 16 * 16 * 16 * 4)
        throw std::runtime_error("invalid compiled parameter extent");
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open parameter file: " + label);
    std::vector<unsigned char> bytes(spec.bytes);
    if (!file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size())) ||
        file.peek() != std::char_traits<char>::eof() || file.bad())
        throw std::runtime_error("parameter size mismatch: " + label);

    // Hash the very snapshot we decode; never reopen after verification.
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int length = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length, EVP_sha256(), nullptr) != 1 ||
        length != 32)
        throw std::runtime_error("SHA-256 failed: " + label);
    constexpr char hex[] = "0123456789abcdef";
    std::string actual;
    actual.reserve(64);
    for (unsigned int i = 0; i < length; ++i) {
        actual.push_back(hex[digest[i] >> 4]);
        actual.push_back(hex[digest[i] & 15]);
    }
    if (actual != spec.sha256)
        throw std::runtime_error("parameter SHA-256 mismatch: " + label);

    return bytes;
}

std::vector<float> load_file(const std::filesystem::path &root, const FileSpec &spec) {
    const auto bytes = load_bytes(root, spec);
    if (bytes.size() % 4)
        throw std::runtime_error("invalid FP32 parameter extent");
    std::vector<float> values;
    values.reserve(bytes.size() / 4);
    for (std::size_t i = 0; i < bytes.size(); i += 4) {
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[i]) |
            (static_cast<std::uint32_t>(bytes[i+1]) << 8) |
            (static_cast<std::uint32_t>(bytes[i+2]) << 16) |
            (static_cast<std::uint32_t>(bytes[i+3]) << 24);
        const float value = std::bit_cast<float>(bits);
        if (!std::isfinite(value))
            throw std::runtime_error("non-finite parameter: " + std::string(spec.path));
        values.push_back(value);
    }
    return values;
}

std::vector<std::uint16_t> load_half_file(const std::filesystem::path &root, const FileSpec &spec) {
    const auto bytes = load_bytes(root, spec);
    if (bytes.size() % 2)
        throw std::runtime_error("invalid FP16 parameter extent");
    std::vector<std::uint16_t> values;
    values.reserve(bytes.size() / 2);
    for (std::size_t i = 0; i < bytes.size(); i += 2) {
        const auto bits = static_cast<std::uint16_t>(
            static_cast<unsigned>(bytes[i]) | (static_cast<unsigned>(bytes[i + 1]) << 8));
        if ((bits & 0x7c00u) == 0x7c00u)
            throw std::runtime_error("non-finite FP16 parameter: " + std::string(spec.path));
        values.push_back(bits);
    }
    return values;
}
} // namespace

std::span<const ModelSpec> model_specs() noexcept {
    return catalog::models;
}

const ModelSpec &select_model(std::string_view mode, std::string_view quality, std::int64_t option) {
    if (mode != "aa" && mode != "soft" && mode != "standard" && mode != "sharp")
        throw std::invalid_argument("mode must be aa, soft, standard or sharp");
    if (quality != "low" && quality != "medium" && quality != "high" && quality != "very_high")
        throw std::invalid_argument("quality must be low, medium, high or very_high");
    if (option < 0 || option > 14)
        throw std::invalid_argument("option must be in 0..14");
    // The selector really ignores 0..14 for these families. Do not invent
    // independent AA/Soft/Standard option weights or require their files.
    if (mode != "sharp")
        option = 0;
    for (const ModelSpec &spec : model_specs())
        if (spec.mode == mode && spec.quality == quality && spec.option == option)
            return spec;
    throw std::runtime_error("native-2x model is missing from compiled catalog");
}

Model load_model(const std::filesystem::path &root, const ModelSpec &spec, WeightPrecision precision) {
    if (precision != WeightPrecision::FP16 && precision != WeightPrecision::FP32)
        throw std::invalid_argument("invalid archive parameter precision");
    const auto archive = root / (precision == WeightPrecision::FP16 ? "weights-fp16" : "weights");
    if (!std::filesystem::is_directory(archive))
        throw std::runtime_error(std::string("missing ") +
            (precision == WeightPrecision::FP16 ? "FP16" : "FP32") +
            " weights archive: " + archive.string() +
            "; weights must name the common root containing weights/ and weights-fp16/ (no fallback)");
    const bool aa = std::string_view(spec.mode) == "aa";
    const std::size_t heads = aa ? 3 : 1;
    if (spec.stages.size() <= heads)
        throw std::runtime_error("invalid compiled native-2x graph");
    std::vector<Layer> body, output;
    int channels = 1;
    for (std::size_t i = 0; i < spec.stages.size(); ++i) {
        const StageSpec &s = spec.stages[i];
        const bool is_body = i < spec.stages.size() - heads;
        if (s.ci != channels || (is_body ? (s.co == 1 || s.kernel != 3) :
            (s.co != 1 || s.kernel != (aa ? 3 : 4))))
            throw std::runtime_error("disconnected or invalid native-2x graph");
        Layer layer = precision == WeightPrecision::FP16 ?
            Layer(s.ci, s.co, s.kernel, load_half_file(archive, s.half_weights), load_half_file(archive, s.half_bias)) :
            Layer(s.ci, s.co, s.kernel, load_file(archive, s.weights), load_file(archive, s.bias));
        if (is_body) {
            channels = s.co;
            body.push_back(std::move(layer));
        } else {
            output.push_back(std::move(layer));
        }
    }
    Layer head = aa ? make_aa_head(output) : std::move(output[0]);
    return Model{spec.id, aa, std::move(body), std::move(head), precision};
}

} // namespace ngu
