#pragma once

#include <VapourSynth4.h>
#include "ngu_cpu.hpp"

#include <memory>

namespace ngu {

// Owns one core-device compute pool per filter instance. No Vulkan/GPU API is
// touched until this object is constructed; the CPU backend does not construct it.
class Vulkan {
  public:
    // Input bits select the entire inference precision: 16 = native FP16,
    // 32 = FP32. The plugin loads the matching archive for that same format;
    // GRAYH never loads FP32 files, and GRAYS never loads FP16 files.
    Vulkan(const Model &model, int width, int height, VSCore *core, const VSAPI *api, int bits = 32);
    ~Vulkan();
    Vulkan(const Vulkan &) = delete;
    Vulkan &operator=(const Vulkan &) = delete;

    VSFrame *new_frame(const VSFrame *property_source) const;
    void process(const VSFrame *source, VSFrame *output) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ngu
