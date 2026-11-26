#include <VapourSynth4.h>
#include "ngu_model.hpp"
#include "ngu_vulkan.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct NodeDeleter {
    const VSAPI *api;
    void operator()(VSNode *node) const noexcept { if (node) api->freeNode(node); }
};
struct FrameDeleter {
    const VSAPI *api;
    void operator()(const VSFrame *frame) const noexcept { if (frame) api->freeFrame(frame); }
};
using NodePtr = std::unique_ptr<VSNode, NodeDeleter>;
using FramePtr = std::unique_ptr<const VSFrame, FrameDeleter>;

struct Filter {
    NodePtr input;
    VSVideoInfo input_info;
    VSVideoInfo output_info;
    std::string model_id;
    const char *compute_precision;
    const char *weight_precision;
    const char *backend_name;
    std::optional<ngu::Model> cpu_model;
    std::unique_ptr<ngu::Vulkan> vulkan;
};

std::string get_string(const VSMap *in, const char *key, const char *fallback, const VSAPI *api) {
    if (api->mapNumElements(in, key) < 0) {
        if (!fallback)
            throw std::invalid_argument(std::string("missing ") + key);
        return fallback;
    }
    int error = 0;
    const char *value = api->mapGetData(in, key, 0, &error);
    if (error || !value)
        throw std::invalid_argument(std::string("invalid ") + key);
    const int size = api->mapGetDataSize(in, key, 0, &error);
    if (error || size <= 0 || std::memchr(value, 0, static_cast<std::size_t>(size)))
        throw std::invalid_argument(std::string(key) + " must be a non-empty string without NUL");
    return {value, static_cast<std::size_t>(size)};
}

std::filesystem::path weights_root(const VSMap *in, const VSPlugin *plugin, const VSAPI *api) {
    if (api->mapNumElements(in, "weights") >= 0) {
        const auto root = get_string(in, "weights", nullptr, api);
        return std::filesystem::path(std::u8string(root.begin(), root.end()));
    }
    const char *path = api->getPluginPath(plugin);
    if (!path || !*path)
        throw std::runtime_error("cannot determine the plugin path; pass weights explicitly");
    const std::filesystem::path library(std::u8string(path, path + std::strlen(path)));
    return library.parent_path() / "weights";
}

bool is_gray_float(const VSVideoFormat &f) noexcept {
    return f.colorFamily == cfGray && f.sampleType == stFloat &&
        (f.bitsPerSample == 16 || f.bitsPerSample == 32) &&
        f.bytesPerSample == f.bitsPerSample / 8 &&
        f.numPlanes == 1 && f.subSamplingW == 0 && f.subSamplingH == 0;
}

const VSFrame *VS_CC get_frame(int n, int reason, void *instance, void **,
                              VSFrameContext *context, VSCore *core, const VSAPI *api) {
    const auto &d = *static_cast<const Filter *>(instance);
    if (reason == arInitial) {
        api->requestFrameFilter(n, d.input.get(), context);
        return nullptr;
    }
    // Nothing is allocated in frameData, including on the upstream-error path.
    if (reason != arAllFramesReady)
        return nullptr;
    try {
        FramePtr source(api->getFrameFilter(n, d.input.get(), context), FrameDeleter{api});
        if (!source)
            throw std::runtime_error("input frame is unavailable");
        const int residency = d.vulkan ? nrGPU : nrCPU;
        if (api->getFrameResidency(source.get()) != residency)
            throw std::invalid_argument("frame residency does not match the selected backend");
        const VSVideoFormat *format = api->getVideoFrameFormat(source.get());
        if (!format || !is_gray_float(*format) ||
            format->bitsPerSample != d.input_info.format.bitsPerSample ||
            api->getFrameWidth(source.get(), 0) != d.input_info.width ||
            api->getFrameHeight(source.get(), 0) != d.input_info.height)
            throw std::invalid_argument("input frame must match the constant GRAYS/GRAYH format/dimensions");

        VSFrame *raw = d.vulkan ? d.vulkan->new_frame(source.get()) :
            api->newVideoFrame(&d.output_info.format, d.output_info.width,
                d.output_info.height, source.get(), core);
        FramePtr output(raw, FrameDeleter{api});
        if (!raw)
            throw std::bad_alloc();
        VSMap *props = api->getFramePropertiesRW(raw);
        if (api->mapSetData(props, "_NGUModel", d.model_id.c_str(),
                static_cast<int>(d.model_id.size()), dtUtf8, maReplace) ||
            api->mapSetData(props, "_NGUPrecision", d.compute_precision, 4, dtUtf8, maReplace) ||
            api->mapSetData(props, "_NGUWeightPrecision", d.weight_precision, 4, dtUtf8, maReplace) ||
            api->mapSetData(props, "_NGUBackend", d.backend_name,
                d.vulkan ? 6 : 3, dtUtf8, maReplace))
            throw std::runtime_error("cannot set output frame properties");
        if (d.vulkan) {
            d.vulkan->process(source.get(), raw);
        } else {
            const ngu::Tensor input = ngu::read_plane(api->getReadPtr(source.get(), 0),
                api->getStride(source.get(), 0), d.input_info.width, d.input_info.height,
                d.input_info.format.bitsPerSample);
            source.reset();
            api->releaseFrameEarly(d.input.get(), n, context);
            const ngu::Tensor result = ngu::run(*d.cpu_model, input);
            ngu::write_plane(result, api->getWritePtr(raw, 0), api->getStride(raw, 0),
                d.output_info.format.bitsPerSample);
        }
        return output.release();
    } catch (const std::exception &e) {
        char message[1024];
        std::snprintf(message, sizeof(message), "NGU2x: %s", e.what());
        api->setFilterError(message, context);
    } catch (...) {
        api->setFilterError("NGU2x: unknown native execution error", context);
    }
    return nullptr;
}

void VS_CC free_filter(void *instance, VSCore *, const VSAPI *) {
    delete static_cast<Filter *>(instance);
}

void VS_CC create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *api) {
    try {
        int error = 0;
        NodePtr input(api->mapGetNode(in, "clip", 0, &error), NodeDeleter{api});
        if (error || !input)
            throw std::invalid_argument("clip is required");
        // :all on BOTH ends prevents R82 inserting residency conversions.
        const std::string backend = get_string(in, "backend", "auto", api);
        if (backend != "auto" && backend != "cpu" && backend != "vulkan")
            throw std::invalid_argument("backend must be auto, cpu or vulkan (case sensitive)");
        const int residency = api->getNodeResidency(input.get());
        if (backend == "cpu" && residency != nrCPU)
            throw std::invalid_argument("CPU backend does not support GPU input; use std.GPUDownload explicitly");
        if (backend == "vulkan" && residency != nrGPU)
            throw std::invalid_argument("Vulkan backend requires GPU input; use std.GPUUpload explicitly");
        if (residency != nrCPU && residency != nrGPU)
            throw std::invalid_argument("unsupported input residency");
        const bool use_vulkan = residency == nrGPU;
        const VSVideoInfo *vi = api->getVideoInfo(input.get());
        if (!vi || !is_gray_float(vi->format) || vi->width <= 0 || vi->height <= 0)
            throw std::invalid_argument("constant format and dimensions, GRAYS/GRAYH (float32/float16, one plane) required");
        ngu::validate_2x_dimensions(vi->width, vi->height);

        const std::string mode = get_string(in, "mode", "sharp", api);
        const std::string quality = get_string(in, "quality", "medium", api);
        std::int64_t option = 0;
        if (api->mapNumElements(in, "option") >= 0) {
            option = api->mapGetInt(in, "option", 0, &error);
            if (error)
                throw std::invalid_argument("invalid option");
        }
        // The input format is the sole authority for archive selection, on both
        // backends. Keep the old option only as a compatibility assertion.
        const bool half = vi->format.bitsPerSample == 16;
        const char *expected_precision = half ? "fp16" : "fp32";
        if (api->mapNumElements(in, "weights_precision") >= 0) {
            const std::string precision = get_string(in, "weights_precision", nullptr, api);
            if (precision != "fp32" && precision != "fp16")
                throw std::invalid_argument("weights_precision must be fp32 or fp16 (case sensitive)");
            if (precision != expected_precision)
                throw std::invalid_argument(std::string("weights_precision conflicts with input format: ") +
                    (half ? "GRAYH requires fp16 (weights-fp16/)" : "GRAYS requires fp32 (weights/)") +
                    "; omit weights_precision to select the archive by input format");
        }
        const auto weight_precision = half ? ngu::WeightPrecision::FP16 : ngu::WeightPrecision::FP32;
        const ngu::ModelSpec &spec = ngu::select_model(mode, quality, option);
        const auto root = weights_root(in, static_cast<const VSPlugin *>(user_data), api);
        ngu::Model model = ngu::load_model(root, spec, weight_precision);
        VSVideoInfo output_info = *vi;
        output_info.width *= 2;
        output_info.height *= 2;
        std::string model_id = model.id;
        const char *compute_prec = (use_vulkan && half) ? "fp16" : "fp32";
        const char *weight_prec = (model.weight_precision == ngu::WeightPrecision::FP16) ? "fp16" : "fp32";
        const char *backend_str = use_vulkan ? "vulkan" : "cpu";
        std::unique_ptr<ngu::Vulkan> vulkan_engine;
        std::optional<ngu::Model> cpu_model;
        if (use_vulkan) {
            // Upload parameters to VRAM and immediately discard host-side weight vectors.
            vulkan_engine = std::make_unique<ngu::Vulkan>(
                model, vi->width, vi->height, core, api, vi->format.bitsPerSample);
        } else {
            cpu_model.emplace(std::move(model));
        }
        auto data = std::make_unique<Filter>(Filter{
            std::move(input), *vi, output_info, std::move(model_id),
            compute_prec, weight_prec, backend_str, std::move(cpu_model), std::move(vulkan_engine)});
        const VSFilterDependency dependency{data->input.get(), rpStrictSpatial};
        VSNode *node = api->createVideoFilterEx2("NGU2x", &data->output_info, get_frame, free_filter,
            fmParallel, use_vulkan ? ffGPUOutput : 0, &dependency, 1, data.get(), core);
        if (!node)
            throw std::runtime_error("failed to create native video filter");
        (void)data.release(); // From here the node owns Filter, even if map insertion fails.
        if (api->mapConsumeNode(out, "clip", node, maReplace))
            throw std::runtime_error("failed to return native video node");
    } catch (const std::exception &e) {
        char message[1024];
        std::snprintf(message, sizeof(message), "NGU2x: %s", e.what());
        api->mapSetError(out, message);
    } catch (...) {
        api->mapSetError(out, "NGU2x: unknown native construction error");
    }
}
} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *api) {
    if (!api->configPlugin("io.github.vs-ngu", "ngu", "Recovered NGU native 2x, format-selected Vulkan FP16 / FP32",
        VS_MAKE_VERSION(0, 1), VS_MAKE_VERSION(4, 3), 0, plugin))
        return;
    api->registerFunction("NGU2x",
        "clip:vnode:all;weights:data:opt;mode:data:opt;quality:data:opt;option:int:opt;backend:data:opt;"
        "weights_precision:data:opt;",
        "clip:vnode:all;", create, plugin, plugin);
}
