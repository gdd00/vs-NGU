#pragma once

#include "ngu_cpu.hpp"
#include <filesystem>

namespace ngu {

struct FileSpec {
    const char *path;
    std::size_t bytes;
    const char *sha256;
};
struct StageSpec {
    int ci;
    int co;
    int kernel;
    FileSpec weights;
    FileSpec bias;
    FileSpec half_weights;
    FileSpec half_bias;
};
struct ModelSpec {
    const char *id;
    const char *mode;
    const char *quality;
    int option;
    std::span<const StageSpec> stages;
};

std::span<const ModelSpec> model_specs() noexcept;
const ModelSpec &select_model(std::string_view mode, std::string_view quality, std::int64_t option);
// common_root/weights/ holds FP32; common_root/weights-fp16/ holds FP16.
// Only the requested archive is inspected. Never convert or fall back.
Model load_model(const std::filesystem::path &common_root, const ModelSpec &spec,
                 WeightPrecision precision = WeightPrecision::FP32);

} // namespace ngu
