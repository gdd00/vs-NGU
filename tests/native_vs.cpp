#define VK_NO_PROTOTYPES
#include <VSVulkan4.h>
#include "ngu_model.hpp"
#include "ngu_half.hpp"
#include "ngu_half_shader.hpp"
#include "ngu_vulkan.hpp"
#include "half_reference.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const VSAPI *api = nullptr;
void check(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
struct Map {
    VSMap *value = api->createMap();
    ~Map() { api->freeMap(value); }
    Map() = default;
    Map(const Map &) = delete;
    Map &operator=(const Map &) = delete;
};
VSNode *node_from(VSMap *result) {
    const char *error = api->mapGetError(result);
    check(!error, error ? error : "");
    int e = 0;
    VSNode *node = api->mapGetNode(result, "clip", 0, &e);
    check(!e && node, "no returned node");
    return node;
}
void text_arg(VSMap *map, const char *name, const char *value) {
    check(api->mapSetData(map, name, value, -1, dtUtf8, maReplace) == 0, "cannot set string");
}
void int_arg(VSMap *map, const char *name, std::int64_t value) {
    check(api->mapSetInt(map, name, value, maReplace) == 0, "cannot set int");
}

void load_plugin(VSCore *core, const char *path) {
    Map args;
    text_arg(args.value, "path", path);
    std::unique_ptr<VSMap, decltype(api->freeMap)> loaded(
        api->invoke(api->getPluginByNamespace("std", core), "LoadPlugin", args.value), api->freeMap);
    const char *error = api->mapGetError(loaded.get());
    check(!error, error ? error : "");
}

float pattern_sample(int x, int y, int n) {
    return -.3f + 1.5f * static_cast<float>((x*37+y*11+n*17)%101) / 100.0f;
}
ngu::Tensor pattern(int width, int height, int n) {
    ngu::Tensor t(width, height, 1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            t.values[static_cast<std::size_t>(y)*width+x] =
                pattern_sample(x, y, n);
    return t;
}
struct Counts {
    std::atomic<int> freed{0};
    std::atomic<int> frames{0};
};
struct Source {
    VSNode *blank;
    VSVideoInfo vi;
    std::shared_ptr<Counts> counts;
    int nan_frame;
    int error_frame;
    int overflow_frame;
};
const VSFrame *VS_CC source_get(int n, int reason, void *instance, void **,
                               VSFrameContext *context, VSCore *core, const VSAPI *) {
    const auto &d = *static_cast<const Source *>(instance);
    if (reason == arInitial) {
        api->requestFrameFilter(n, d.blank, context);
        return nullptr;
    }
    if (reason != arAllFramesReady) return nullptr;
    if (n == d.error_frame) {
        api->setFilterError("intentional upstream error", context);
        return nullptr;
    }
    const VSFrame *input = api->getFrameFilter(n, d.blank, context);
    if (!input) return nullptr;
    VSFrame *output = api->newVideoFrame(&d.vi.format, d.vi.width, d.vi.height, input, core);
    api->freeFrame(input);
    if (!output) {
        api->setFilterError("test frame allocation", context);
        return nullptr;
    }
    try {
        auto samples = pattern(d.vi.width, d.vi.height, n);
        if (n == d.nan_frame) samples.values[0] = std::numeric_limits<float>::quiet_NaN();
        if (n == d.overflow_frame)
            std::fill(samples.values.begin(), samples.values.end(), std::numeric_limits<float>::max());
        ngu::write_plane(samples, api->getWritePtr(output, 0), api->getStride(output, 0), d.vi.format.bitsPerSample);
        VSMap *props = api->getFramePropertiesRW(output);
        int_arg(props, "_TestStamp", n);
        int_arg(props, "_Range", 1);
        int_arg(props, "_SARNum", 4);
        int_arg(props, "_SARDen", 3);
        ++d.counts->frames;
        return output;
    } catch (...) {
        api->freeFrame(output);
        api->setFilterError("test source failed", context);
        return nullptr;
    }
}
void VS_CC source_free(void *instance, VSCore *, const VSAPI *) {
    auto *d = static_cast<Source *>(instance);
    api->freeNode(d->blank);
    ++d->counts->freed;
    delete d;
}
VSNode *make_source(VSCore *core, const std::shared_ptr<Counts> &counts,
                    int nan_frame = -1, int error_frame = -1, int overflow_frame = -1, int bits = 32,
                    int width = 31, int height = 17) {
    Map args;
    int_arg(args.value, "width", width);
    int_arg(args.value, "height", height);
    int_arg(args.value, "format", bits == 16 ? pfGrayH : pfGrayS);
    int_arg(args.value, "length", 32);
    int_arg(args.value, "fpsnum", 24000);
    int_arg(args.value, "fpsden", 1001);
    VSPlugin *std = api->getPluginByNamespace("std", core);
    VSMap *out = api->invoke(std, "BlankClip", args.value);
    VSNode *blank = node_from(out);
    api->freeMap(out);
    auto data = std::make_unique<Source>(Source{blank, *api->getVideoInfo(blank), counts,
        nan_frame, error_frame, overflow_frame});
    const VSFilterDependency dependency{blank, rpStrictSpatial};
    VSNode *source = api->createVideoFilter2("NativeTestSource", &data->vi, source_get, source_free,
        fmParallel, &dependency, 1, data.get(), core);
    if (!source) {
        api->freeNode(blank);
        throw std::runtime_error("cannot create test source");
    }
    data.release();
    return source;
}

VSNode *transfer(VSCore *core, VSNode *source, const char *function) {
    Map args;
    check(api->mapSetNode(args.value, "clip", source, maReplace) == 0, "cannot set transfer source");
    VSMap *out = api->invoke(api->getPluginByNamespace("std", core), function, args.value);
    try {
        VSNode *node = node_from(out);
        api->freeMap(out);
        return node;
    } catch (...) {
        api->freeMap(out);
        throw;
    }
}

VSNode *filter(VSCore *core, VSNode *source, const char *weights, const char *mode = "sharp",
               const char *quality = "high", int option = 14, const char *backend = "auto",
               const char *weights_precision = nullptr) {
    Map args;
    api->mapSetNode(args.value, "clip", source, maReplace);
    if (weights)
        text_arg(args.value, "weights", weights);
    text_arg(args.value, "mode", mode);
    text_arg(args.value, "quality", quality);
    int_arg(args.value, "option", option);
    text_arg(args.value, "backend", backend);
    if (weights_precision)
        text_arg(args.value, "weights_precision", weights_precision);
    VSMap *out = api->invoke(api->getPluginByNamespace("ngu", core), "NGU2x", args.value);
    try {
        VSNode *node = node_from(out);
        api->freeMap(out);
        return node;
    } catch (...) {
        api->freeMap(out);
        throw;
    }
}

struct Completion {
    std::mutex mutex;
    std::condition_variable cv;
    int done = 0;
    int errors = 0;
    std::vector<const VSFrame *> retained;
};
struct Task {
    Completion *completion;
    const ngu::Tensor *expected;
    int n;
    const char *weights_precision = "fp32";
};
bool properties_match(const VSFrame *f, int n, const char *backend, const char *weights_precision = "fp32") {
    const VSMap *props = api->getFramePropertiesRO(f);
    int error = 0;
    if (api->mapGetInt(props, "_TestStamp", 0, &error) != n || error) return false;
    if (api->mapGetInt(props, "_Range", 0, &error) != 1 || error) return false;
    if (api->mapGetInt(props, "_SARNum", 0, &error) != 4 || error) return false;
    if (api->mapGetInt(props, "_SARDen", 0, &error) != 3 || error) return false;
    const char *precision = api->mapGetData(props, "_NGUPrecision", 0, &error);
    const bool half = std::strcmp(backend, "vulkan") == 0 &&
        api->getVideoFrameFormat(f)->bitsPerSample == 16;
    if (error || !precision || std::string(precision) != (half ? "fp16" : "fp32")) return false;
    const char *weights = api->mapGetData(props, "_NGUWeightPrecision", 0, &error);
    if (error || !weights || std::strcmp(weights, weights_precision) != 0) return false;
    const char *actual_backend = api->mapGetData(props, "_NGUBackend", 0, &error);
    return !error && actual_backend && std::string(actual_backend) == backend;
}
bool matches(const VSFrame *f, const ngu::Tensor &expected, int n, const char *backend = "cpu",
             const char *weights_precision = "fp32") {
    if (!f || api->getFrameWidth(f, 0) != expected.width ||
        api->getFrameHeight(f, 0) != expected.height || api->getFrameResidency(f) != nrCPU)
        return false;
    const int bits = api->getVideoFrameFormat(f)->bitsPerSample;
    const std::uint8_t *ptr = api->getReadPtr(f, 0);
    for (int y = 0; y < expected.height; ++y) {
        const auto *row = ptr + static_cast<std::ptrdiff_t>(y)*api->getStride(f, 0);
        const float *reference = expected.values.data()+static_cast<std::size_t>(y)*expected.width;
        if (bits == 16) {
            for (int x = 0; x < expected.width; ++x) {
                std::uint16_t sample;
                std::memcpy(&sample, row + static_cast<std::size_t>(x) * 2, 2);
                const auto ref = ngu::float_to_half(reference[x]);
                if (sample != ref) return false; // Current half-FMA schedule oracle, not an FP32 tolerance.
            }
        } else if (std::strcmp(backend, "cpu") == 0) {
            if (std::memcmp(row, reference, static_cast<std::size_t>(expected.width)*4))
                return false; // CPU serial/parallel must remain bit-identical.
        } else {
            for (int x = 0; x < expected.width; ++x) {
                float value = 0;
                std::memcpy(&value, row+static_cast<std::size_t>(x)*4, 4);
                if (!std::isfinite(value) || std::fabs(value-reference[x]) > 2e-5f+2e-5f*std::fabs(reference[x]))
                    return false;
            }
        }
    }
    return properties_match(f, n, backend, weights_precision);
}
void VS_CC complete(void *user, const VSFrame *frame, int n, VSNode *, const char *error) {
    const auto &task = *static_cast<const Task *>(user);
    Completion &c = *task.completion;
    const bool valid = !error && n == task.n && matches(frame, *task.expected, n, "cpu", task.weights_precision);
    {
        std::lock_guard lock(c.mutex);
        if (!valid) ++c.errors;
        // Transfer a callback's frame reference into retained storage. This tests
        // that newVideoFrame output outlives both source and filter nodes.
        c.retained.push_back(frame);
        ++c.done;
        c.cv.notify_one();
    }
}

void test_parallel_lifetime(VSCore *core, const char *root, const char *mode,
                            int bits = 32, const char *weights_precision = "fp32") {
    const auto counts = std::make_shared<Counts>();
    VSNode *source = make_source(core, counts, -1, -1, -1, bits);
    VSNode *output = filter(core, source, root, mode, "high", 14, "auto", weights_precision);
    api->freeNode(source); // Only the NGU dependency now keeps the source alive.
    check(api->getNodeFilterMode(output) == fmParallel, "not a parallel native filter");
    check(api->getNodeDependency(output, 0)->requestPattern == rpStrictSpatial, "wrong dependency pattern");
    const VSVideoInfo info = *api->getVideoInfo(output);
    check(info.width == 62 && info.height == 34 && info.fpsNum == 24000 &&
        info.fpsDen == 1001 && info.numFrames == 32 && info.format.bitsPerSample == bits,
        "video info was not preserved");

    const auto precision = std::strcmp(weights_precision, "fp16") == 0 ?
        ngu::WeightPrecision::FP16 : ngu::WeightPrecision::FP32;
    const auto model = ngu::load_model(root, ngu::select_model(mode, "high", 14), precision);
    std::vector<ngu::Tensor> expected;
    std::vector<Task> tasks;
    Completion c;
    c.retained.reserve(40);
    for (int n = 0; n < 32; ++n) {
        auto input = pattern(31, 17, n);
        if (bits == 16)
            for (float &v : input.values) v = ngu::round_half(v);
        expected.push_back(ngu::run(model, input));
    }
    for (int i = 0; i < 40; ++i) // shuffled and duplicated asynchronous requests
        tasks.push_back(Task{&c, &expected[(i*13)%32], (i*13)%32, weights_precision});
    for (auto &task : tasks)
        api->getFrameAsync(task.n, output, complete, &task);
    {
        std::unique_lock lock(c.mutex);
        c.cv.wait(lock, [&]{ return c.done == 40; });
    }
    check(c.errors == 0, "async pixels/properties differed from serial execution");
    api->freeNode(output);
    api->clearCoreCaches(core);
    check(counts->freed == 1, "source/filter node lifetime leaked");
    for (std::size_t i = 0; i < c.retained.size(); ++i) {
        const VSFrame *frame = c.retained[i];
        const VSMap *props = api->getFramePropertiesRO(frame);
        const int n = static_cast<int>(api->mapGetInt(props, "_TestStamp", 0, nullptr));
        check(matches(frame, expected[n], n, "cpu", weights_precision), "retained frame invalid after node destruction");
        api->freeFrame(frame);
    }
}

void VS_CC gpu_complete(void *user, const VSFrame *frame, int n, VSNode *, const char *error) {
    const auto &task = *static_cast<const Task *>(user);
    Completion &c = *task.completion;
    const bool valid = !error && frame && n == task.n &&
        api->getFrameResidency(frame) == nrGPU &&
        api->getFrameWidth(frame, 0) == task.expected->width &&
        api->getFrameHeight(frame, 0) == task.expected->height &&
        properties_match(frame, n, "vulkan", task.weights_precision);
    std::lock_guard lock(c.mutex);
    if (!valid) ++c.errors;
    c.retained.push_back(frame);
    ++c.done;
    c.cv.notify_one();
}

struct RetainedSource {
    std::vector<const VSFrame *> frames;
    VSVideoInfo vi;
};
const VSFrame *VS_CC retained_get(int n, int reason, void *instance, void **,
                                VSFrameContext *, VSCore *, const VSAPI *) {
    const auto &d = *static_cast<const RetainedSource *>(instance);
    return reason == arInitial ? api->addFrameRef(d.frames.at(static_cast<std::size_t>(n))) : nullptr;
}
void VS_CC retained_free(void *instance, VSCore *, const VSAPI *) {
    auto *d = static_cast<RetainedSource *>(instance);
    for (const VSFrame *f : d->frames) api->freeFrame(f);
    delete d;
}
void gpu_dependency(VSNode *node, VSNode *upstream) {
    check(api->getNodeResidency(node) == nrGPU, "Vulkan node lost GPU residency");
    check(api->getNodeFilterMode(node) == fmParallel, "Vulkan node is not fmParallel");
    check(api->getNumNodeDependencies(node) == 1, "unexpected Vulkan dependencies");
    const auto *dep = api->getNodeDependency(node, 0);
    check(dep && dep->source == upstream && dep->requestPattern == rpStrictSpatial,
        "Vulkan dependency is not direct / strict spatial (implicit conversion?)");
}
void test_gpu_parallel_lifetime(VSCore *core, const char *root, const char *mode, bool chain,
                                int bits = 32, const char *weights_precision = "fp32") {
    const auto counts = std::make_shared<Counts>();
    VSNode *source = make_source(core, counts, -1, -1, -1, bits);
    VSNode *upload = transfer(core, source, "GPUUpload");
    VSNode *output = filter(core, upload, root, mode, "high", 14, "auto", weights_precision);
    gpu_dependency(output, upload);
    if (chain) {
        VSNode *second = filter(core, output, root, "aa", "low", 0, "vulkan", weights_precision);
        gpu_dependency(second, output);
        api->freeNode(output);
        output = second;
    }
    api->freeNode(upload);
    api->freeNode(source);
    const VSVideoInfo info = *api->getVideoInfo(output);
    const int scale = chain ? 4 : 2;
    check(info.width == 31*scale && info.height == 17*scale && info.format.sampleType == stFloat &&
        info.format.bitsPerSample == bits && info.fpsNum == 24000 && info.fpsDen == 1001 &&
        info.numFrames == 32, "GPU format/dimensions/fps changed");

    const auto precision = std::strcmp(weights_precision, "fp16") == 0 ?
        ngu::WeightPrecision::FP16 : ngu::WeightPrecision::FP32;
    const auto model = ngu::load_model(root, ngu::select_model(mode, "high", 14), precision);
    const auto second_model = ngu::load_model(root, ngu::select_model("aa", "low", 0), precision);
    std::vector<ngu::Tensor> expected;
    for (int n = 0; n < 32; ++n) {
        auto input = pattern(31, 17, n);
        if (bits == 16)
            for (float &v : input.values) v = ngu::round_half(v);
        auto first = bits == 16 ? half_reference::run(model, input) : ngu::run(model, input);
        if (bits == 16)
            for (float &v : first.values) v = ngu::round_half(v);
        expected.push_back(chain ? (bits == 16 ? half_reference::run(second_model, first) :
            ngu::run(second_model, first)) : std::move(first));
    }
    Completion c;
    c.retained.reserve(40);
    std::vector<Task> tasks;
    for (int i = 0; i < 40; ++i)
        tasks.push_back(Task{&c, &expected[(i*13)%32], (i*13)%32, weights_precision});
    for (auto &task : tasks) api->getFrameAsync(task.n, output, gpu_complete, &task);
    {
        std::unique_lock lock(c.mutex);
        c.cv.wait(lock, [&]{ return c.done == 40; });
    }
    check(c.errors == 0, "GPU async frame residency/properties failed");

    // Read via the core's transfer pool while the NGU producer is still alive:
    // this exercises its published timeline rather than a host wait in NGU.
    VSNode *download = transfer(core, output, "GPUDownload");
    for (int n : {5, 7}) {
        char error[1024]{};
        const VSFrame *f = api->getFrame(n, download, error, sizeof(error));
        check(matches(f, expected[n], n, "vulkan", weights_precision), std::string("GPU download pixels: ")+error);
        api->freeFrame(f);
    }
    api->freeNode(download);
    api->freeNode(output);
    api->clearCoreCaches(core);
    check(counts->freed == 1, "GPU source/filter lifecycle leaked");

    // Download the retained GPU frames only AFTER both NGU nodes and their exec
    // pools have gone away. This checks the data, not just live frame handles.
    auto retained = std::make_unique<RetainedSource>(RetainedSource{std::move(c.retained), info});
    retained->vi.numFrames = 40;
    std::unique_ptr<VSNode, decltype(api->freeNode)> resident(
        api->createVideoFilterEx2("RetainedGPUFrames", &retained->vi,
            retained_get, retained_free, fmParallel, ffGPUOutput, nullptr, 0, retained.get(), core),
        api->freeNode);
    check(resident != nullptr, "cannot create retained GPU frame source");
    retained.release();
    download = transfer(core, resident.get(), "GPUDownload");
    resident.reset();
    for (int i = 0; i < 40; ++i) {
        char error[1024]{};
        const VSFrame *f = api->getFrame(i, download, error, sizeof(error));
        check(f != nullptr, std::string("retained GPU download: ")+error);
        int e = 0;
        const int n = static_cast<int>(api->mapGetInt(api->getFramePropertiesRO(f), "_TestStamp", 0, &e));
        check(!e && n >= 0 && n < 32 && matches(f, expected[n], n, "vulkan", weights_precision),
            "retained GPU pixels invalid after node destruction");
        api->freeFrame(f);
    }
    api->freeNode(download);
    api->clearCoreCaches(core);
}

void test_precision_guards() {
    const ngu::HalfCapabilities good{true,true,true,true,true};
    ngu::require_half_capabilities(good);
    const std::array<ngu::HalfCapabilities,5> missing{{
        {false,true,true,true,true}, {true,false,true,true,true},
        {true,true,false,true,true}, {true,true,true,false,true}, {true,true,true,true,false}}};
    for (const auto &c:missing) {
        bool rejected=false;
        try { ngu::require_half_capabilities(c); }
        catch (const std::exception &e) {
            rejected=std::string(e.what()).find("no FP32 fallback")!=std::string::npos;
        }
        check(rejected,"missing half capability did not give an explicit error");
    }
    std::vector<std::uint32_t> code{0x07230203u,0x10600u,0,2,0,(3u<<16)|22u,1,16};
    ngu::require_spirv_precision(code,16);
    for (int bits:{16,32}) {
        auto mixed=code;
        mixed.insert(mixed.end(),{(3u<<16)|22u,2,static_cast<unsigned>(bits==16?32:16)});
        bool rejected=false;
        try { ngu::require_spirv_precision(mixed,bits); }
        catch (const std::exception &) { rejected=true; }
        check(rejected,"mixed SPIR-V floating types were accepted");
    }
}

void test_gpu_half_tiles(VSCore *core, const char *root) {
    for (const auto [w,h]:{std::pair{1,1},{1,5},{5,1},{5,3},{6,4},{17,15},{31,11},{32,12},{33,13},{65,25}}) {
        const auto counts=std::make_shared<Counts>();
        VSNode *source=make_source(core,counts,-1,-1,-1,16,w,h);
        VSNode *upload=transfer(core,source,"GPUUpload");
        for (const char *mode:{"sharp","aa"})
            for (const char *quality:{"low","medium","high","very_high"}) {
                const auto model=ngu::load_model(root,ngu::select_model(mode,quality,0),ngu::WeightPrecision::FP16);
                const auto expected=half_reference::run(model,pattern(w,h,0));
                for (const char *precision:{static_cast<const char *>(nullptr),"fp16"}) {
                    VSNode *output=filter(core,upload,root,mode,quality,0,"vulkan",precision);
                    VSNode *download=transfer(core,output,"GPUDownload");
                    char error[1024]{};
                    const VSFrame *frame=api->getFrame(0,download,error,sizeof(error));
                    const bool ok=matches(frame,expected,0,"vulkan","fp16");
                    api->freeFrame(frame);
                    api->freeNode(download);
                    api->freeNode(output);
                    check(ok,std::string("half scalar tile mismatch: ")+mode+"/"+quality+" "+
                        std::to_string(w)+"x"+std::to_string(h)+" "+(precision?precision:"format default")+" "+error);
                }
            }
        api->freeNode(upload);
        api->freeNode(source);
        api->clearCoreCaches(core);
        check(counts->freed==1,"half tile test source leaked");
    }
}

void test_gpu_half_large(VSCore *core, const char *root) {
    int checked = 0;
    for (const auto [w,h]:{std::pair{1920,1080},{1919,1079}}) {
        const auto counts = std::make_shared<Counts>();
        VSNode *source = make_source(core,counts,-1,-1,-1,16,w,h);
        VSNode *upload = transfer(core,source,"GPUUpload");
        for (const char *mode:{"sharp","aa"})
            for (const char *quality:{"low","medium","high","very_high"}) {
                const auto model = ngu::load_model(root,ngu::select_model(mode,quality,0),ngu::WeightPrecision::FP16);
                VSNode *output = filter(core,upload,root,mode,quality,0,"vulkan");
                VSNode *download = transfer(core,output,"GPUDownload");
                char error[1024]{};
                const VSFrame *frame = api->getFrame(0,download,error,sizeof(error));
                check(frame != nullptr,error);
                check(properties_match(frame,0,"vulkan","fp16"),"large image properties");
                const auto *ptr = api->getReadPtr(frame,0);
                const auto stride = api->getStride(frame,0);
                // Inspect every output for nonfinite values, not just patches.
                for (int y=0; y<h*2; ++y)
                    for (int x=0; x<w*2; ++x) {
                        std::uint16_t code;
                        std::memcpy(&code,ptr+y*stride+x*2,2);
                        check((code & 0x7c00u) != 0x7c00u,"nonfinite large image output");
                        if (model.aa && !(x&1) && !(y&1))
                            check(code == ngu::float_to_half(pattern_sample(x/2,y/2,0)),
                                  "AA native phase differs in large image");
                    }
                const int radius = static_cast<int>(model.body.size())+1;
                for (const auto [cx,cy]:{std::pair{0,0},{w-1,0},{0,h-1},{w-1,h-1},{1,2},
                        {w-2,h-2},{13,13},{15,15},{31,11},{32,12},{33,13},{w/2,h/2},{w-33,h-13}}) {
                    const int x0=std::max(0,cx-radius), y0=std::max(0,cy-radius);
                    const int x1=std::min(w-1,cx+radius), y1=std::min(h-1,cy+radius);
                    ngu::Tensor patch(x1-x0+1,y1-y0+1,1);
                    for (int y=0; y<patch.height; ++y)
                        for (int x=0; x<patch.width; ++x)
                            patch.values[y*patch.width+x]=ngu::round_half(pattern_sample(x0+x,y0+y,0));
                    // Crop only outside the complete receptive field; when at
                    // image borders the crop has the real per-layer clamp edge.
                    const auto expected = half_reference::run(model,patch);
                    for (int sy=0; sy<2; ++sy)
                        for (int sx=0; sx<2; ++sx) {
                            std::uint16_t code;
                            std::memcpy(&code,ptr+(cy*2+sy)*stride+(cx*2+sx)*2,2);
                            const auto ref = ngu::float_to_half(expected.at((cx-x0)*2+sx,(cy-y0)*2+sy,0));
                            check(code == ref,std::string("large FMA patch mismatch: ")+mode+"/"+quality);
                            ++checked;
                        }
                }
                api->freeFrame(frame);
                api->freeNode(download);
                api->freeNode(output);
            }
        api->freeNode(upload);
        api->freeNode(source);
        api->clearCoreCaches(core);
        check(counts->freed == 1,"large source leaked");
    }
    std::cout << "large nonconstant FP16: 16 full HD images, " << checked
              << " scalar FMA patch values exact; finite scan and all AA native samples passed\n";
}

void test_gpu_fallback(VSCore *core) {
    // Synthetic unknown topology exercises same-precision kernel fallback.
    // No archive files are read, and half input uses native half model storage.
    const auto layer=[](int ci,int co,int k,ngu::WeightPrecision precision) {
        std::vector<float> w(ci*co*k*k),b(co);
        for (std::size_t i=0;i<w.size();++i) w[i]=static_cast<float>(static_cast<int>(i%7)-3)*0.04f;
        for (int i=0;i<co;++i) b[i]=0.1f*static_cast<float>(i+1);
        return ngu::Layer(ci,co,k,std::move(w),std::move(b),precision);
    };
    for (bool aa:{false,true}) {
        for (int bits:{16,32}) {
            const auto precision=bits==16?ngu::WeightPrecision::FP16:ngu::WeightPrecision::FP32;
            const ngu::Model model{"synthetic_fallback",aa,{layer(1,4,3,precision)},
                layer(4,aa?4:1,aa?3:4,precision),precision};
            const auto counts=std::make_shared<Counts>();
            VSNode *source=make_source(core,counts,-1,-1,-1,bits,5,3);
            VSNode *upload=transfer(core,source,"GPUUpload");
            char error[1024]{};
            const VSFrame *input=api->getFrame(0,upload,error,sizeof(error));
            check(input!=nullptr,error);
            VSVideoInfo info=*api->getVideoInfo(upload);
            info.width*=2; info.height*=2; info.numFrames=1;
            VSFrame *output=nullptr;
            {
                ngu::Vulkan engine(model,5,3,core,api,bits);
                output=engine.new_frame(input);
                check(output!=nullptr,"fallback output allocation");
                text_arg(api->getFramePropertiesRW(output),"_NGUPrecision",bits==16?"fp16":"fp32");
                text_arg(api->getFramePropertiesRW(output),"_NGUWeightPrecision",bits==16?"fp16":"fp32");
                text_arg(api->getFramePropertiesRW(output),"_NGUBackend","vulkan");
                engine.process(input,output);
            } // Pipeline/workspace destruction while output is retained.
            api->freeFrame(input); api->freeNode(upload); api->freeNode(source);
            auto retained=std::make_unique<RetainedSource>(RetainedSource{{output},info});
            VSNode *resident=api->createVideoFilterEx2("FallbackGPUFrame",&retained->vi,
                retained_get,retained_free,fmParallel,ffGPUOutput,nullptr,0,retained.get(),core);
            check(resident!=nullptr,"fallback retained source");
            retained.release();
            VSNode *download=transfer(core,resident,"GPUDownload");
            api->freeNode(resident);
            const VSFrame *frame=api->getFrame(0,download,error,sizeof(error));
            const auto expected=bits==16 ? half_reference::run(model,pattern(5,3,0),false) :
                ngu::run(model,pattern(5,3,0));
            const bool ok=matches(frame,expected,0,"vulkan",bits==16?"fp16":"fp32");
            api->freeFrame(frame); api->freeNode(download); api->clearCoreCaches(core);
            check(ok,std::string("fallback scalar mismatch: ")+error);
        }
    }
}

void test_gpu_creation_errors(VSCore *core, const char *root) {
    const auto counts = std::make_shared<Counts>();
    VSNode *source = make_source(core, counts);
    VSNode *upload = transfer(core, source, "GPUUpload");
    const auto rejected = [&](VSNode *input, const char *weights, int option, const char *backend) {
        bool failed = false;
        try {
            VSNode *node = filter(core, input, weights, "sharp", "low", option, backend);
            api->freeNode(node);
        } catch (const std::exception &) { failed = true; }
        check(failed, "bad backend / GPU construction unexpectedly accepted");
    };
    rejected(source, root, 0, "vulkan");
    rejected(upload, root, 0, "cpu");
    for (const char *backend : {"CPU", "Vulkan", "AUTO", "", "gpu"})
        rejected(source, root, 0, backend);
    rejected(upload, root, 15, "vulkan");
    rejected(upload, "/nonexistent/ngu-weights", 0, "vulkan");
    Map args;
    int_arg(args.value, "width", 4);
    int_arg(args.value, "height", 4);
    int_arg(args.value, "format", pfGray8);
    int_arg(args.value, "length", 1);
    VSMap *result = api->invoke(api->getPluginByNamespace("std", core), "BlankClip", args.value);
    VSNode *integer = node_from(result);
    api->freeMap(result);
    VSNode *bad_gpu = transfer(core, integer, "GPUUpload");
    rejected(bad_gpu, root, 0, "vulkan");
    api->freeNode(bad_gpu);
    api->freeNode(integer);
    api->freeNode(upload);
    api->freeNode(source);
    api->clearCoreCaches(core);
    check(counts->freed == 1, "GPU construction errors leaked source");
}

void test_format_precision_errors(VSCore *core) {
    for (int bits : {16, 32}) {
        const auto counts = std::make_shared<Counts>();
        VSNode *source = make_source(core, counts, -1, -1, -1, bits);
        VSNode *upload = transfer(core, source, "GPUUpload");
        for (VSNode *input : {source, upload}) {
            bool rejected = false;
            try {
                // The assertion must fail before trying this nonexistent root.
                VSNode *node = filter(core, input, "/nonexistent/ngu-weights",
                    "sharp", "low", 0, "auto", bits == 16 ? "fp32" : "fp16");
                api->freeNode(node);
            } catch (const std::exception &e) {
                rejected = std::string(e.what()).find(bits == 16 ?
                    "GRAYH requires fp16" : "GRAYS requires fp32") != std::string::npos;
            }
            check(rejected, "input/weights_precision mismatch was not rejected before archive access");
        }
        api->freeNode(upload);
        api->freeNode(source);
        api->clearCoreCaches(core);
        check(counts->freed == 1, "precision rejection leaked source");
    }
}

namespace fs = std::filesystem;

struct TempDir {
    fs::path path;
    TempDir() {
        const auto base = fs::canonical(fs::temp_directory_path());
        std::random_device random;
        do {
            path = base / ("vsngu-test-" + std::to_string(random()));
        } while (!fs::create_directory(path));
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;
};

void test_adjacent_weights(const char *plugin_path, const char *root) {
    const TempDir temp;
    const auto directory = temp.path / "vsngu";
    fs::create_directory(directory);
    const auto library = directory / fs::path(plugin_path).filename();
    fs::copy_file(plugin_path, library);

    // Unload the copy before TempDir removes it, including on Windows.
    std::unique_ptr<VSCore, decltype(api->freeCore)> core_guard(
        api->createCore(ccfDisableAutoLoading | ccfEnableFrameRefDebug), api->freeCore);
    VSCore *core = core_guard.get();
    check(core != nullptr, "createCore for adjacent weights failed");
    api->setThreadCount(8, core);
    const auto library_utf8 = library.u8string();
    load_plugin(core, reinterpret_cast<const char *>(library_utf8.c_str()));

    using Node = std::unique_ptr<VSNode, decltype(api->freeNode)>;
    using Frame = std::unique_ptr<const VSFrame, decltype(api->freeFrame)>;
    const auto adjacent = directory / "weights";
    const auto &spec = ngu::select_model("sharp", "low", 0);
    for (int bits : {32, 16}) {
        const bool half = bits == 16;
        const char *precision = half ? "fp16" : "fp32";
        const auto archive = adjacent / (half ? "weights-fp16" : "weights");
        const auto model = ngu::load_model(root, spec, half ? ngu::WeightPrecision::FP16 : ngu::WeightPrecision::FP32);
        auto samples = pattern(3, 2, 0);
        if (half)
            for (float &sample : samples.values) sample = ngu::round_half(sample);
        const auto cpu_expected = ngu::run(model, samples);
        const auto gpu_expected = half ? half_reference::run(model, samples) : cpu_expected;
        const auto counts = std::make_shared<Counts>();
        Node source(make_source(core, counts, -1, -1, -1, bits, 3, 2), api->freeNode);
        Node upload(transfer(core, source.get(), "GPUUpload"), api->freeNode);

        const auto expect_error = [&](const char *assertion, std::string expected) {
            std::replace(expected.begin(), expected.end(), '\\', '/');
            for (VSNode *input : {source.get(), upload.get()}) {
                std::string message;
                try {
                    const Node out(filter(core, input, nullptr, "sharp", "low", 0, "auto", assertion), api->freeNode);
                } catch (const std::exception &e) {
                    message = e.what();
                }
                std::replace(message.begin(), message.end(), '\\', '/');
                check(message.find(expected) != std::string::npos, "expected " + expected + "; got: " + message);
            }
        };
        const auto infer = [&](const char *weights) {
            for (VSNode *input : {source.get(), upload.get()}) {
                const bool gpu = input == upload.get();
                const Node out(filter(core, input, weights, "sharp", "low", 0), api->freeNode);
                check(api->getNodeResidency(out.get()) == (gpu ? nrGPU : nrCPU), "wrong NGU output residency");
                const Node readable(gpu ? transfer(core, out.get(), "GPUDownload") : api->addNodeRef(out.get()),
                                    api->freeNode);
                char error[1024]{};
                const Frame frame(api->getFrame(0, readable.get(), error, sizeof(error)), api->freeFrame);
                check(frame && api->getVideoFrameFormat(frame.get())->bitsPerSample == bits &&
                      matches(frame.get(), gpu ? gpu_expected : cpu_expected, 0, gpu ? "vulkan" : "cpu", precision),
                      std::string("adjacent weights inference failed: ") + precision + " " + error);
            }
        };

        const std::string missing = std::string(half ? "missing FP16" : "missing FP32") +
            " weights archive: " + archive.string();
        expect_error(half ? "fp32" : "fp16", half ? "GRAYH requires fp16" : "GRAYS requires fp32");
        expect_error(nullptr, missing); // Neither archive nor the default root exists.
        infer(root);                   // Explicit root still works.
        for (const char *child : {"weights", "weights-fp16"}) {
            fs::create_directories(adjacent / child);
            fs::copy(fs::path(root) / child / spec.id, adjacent / child / spec.id, fs::copy_options::recursive);
        }
        infer(nullptr);
        fs::remove_all(archive);
        expect_error(nullptr, missing); // The opposite archive exists but must not be used.
        fs::remove_all(adjacent);
        upload.reset();
        source.reset();
        api->clearCoreCaches(core);
        check(counts->freed == 1, "adjacent weights test leaked source");
    }

    VSVulkanCoreInfo gpu_info{};
    char gpu_error[1024]{};
    const int result = api->getVulkanAPI()->getVulkanCoreInfo(core, &gpu_info, gpu_error, sizeof(gpu_error));
    check(result == 0, std::string("adjacent weights GPU accounting query: ") + gpu_error);
    check(gpu_info.allocated == 0, "adjacent weights GPU allocations were not released after cache sweep");
    core_guard.reset();
}

void test_frame_error(VSCore *core, const char *root, int kind, bool gpu = false) {
    const auto counts = std::make_shared<Counts>();
    VSNode *source = make_source(core, counts, kind == 0 ? 0 : -1,
        kind == 1 ? 0 : -1, kind == 2 ? 0 : -1);
    VSNode *input = gpu ? transfer(core, source, "GPUUpload") : api->addNodeRef(source);
    VSNode *output = filter(core, input, root, "soft", "low", 0);
    api->freeNode(input);
    api->freeNode(source);
    char error[1024]{};
    const VSFrame *frame = api->getFrame(0, output, error, sizeof(error));
    check(!frame && std::strlen(error) > 0, "bad input/upstream failure was not rejected");
    frame = api->getFrame(1, output, error, sizeof(error));
    check(frame != nullptr, "failed frame poisoned later requests");
    api->freeFrame(frame);
    api->freeNode(output);
    api->clearCoreCaches(core);
    check(counts->freed == 1, "frame error leaked filter/source");
}
void test_creation_error(VSCore *core, const char *root, bool bad_weights) {
    const auto counts = std::make_shared<Counts>();
    VSNode *source = make_source(core, counts);
    bool rejected = false;
    try {
        VSNode *unexpected = filter(core, source, bad_weights ? "/nonexistent/ngu-weights" : root,
            "sharp", "low", bad_weights ? 0 : 15);
        api->freeNode(unexpected);
    } catch (const std::exception &) { rejected = true; }
    api->freeNode(source);
    api->clearCoreCaches(core);
    check(rejected && counts->freed == 1, "creation failure leaked source or was accepted");
}
} // namespace

int main(int argc, char **argv) {
    try {
        const bool core_only = argc == 2 && std::string(argv[1]) == "--vulkan-core-only";
        const bool large = argc == 4 && std::string(argv[3]) == "--large";
        check(argc == 3 || large || core_only, "usage: ngu_vs_tests plugin weights-root [--large] | --vulkan-core-only");
        test_precision_guards();
        api = getVapourSynthAPI(VS_MAKE_VERSION(4,3));
        check(api != nullptr, "API4.3 is required for the native residency contract");
        std::unique_ptr<VSCore, decltype(api->freeCore)> core_guard(
            api->createCore(ccfDisableAutoLoading | ccfEnableFrameRefDebug), api->freeCore);
        VSCore *core = core_guard.get();
        check(core != nullptr, "createCore failed");
        api->setThreadCount(8, core);
        if (core_only) {
            // Minimal diagnostic for runtime/driver LSan reports: NO plugin is loaded.
            VSVulkanCoreInfo info{};
            char error[1024]{};
            const int result = api->getVulkanAPI()->getVulkanCoreInfo(core, &info, error, sizeof(error));
            core_guard.reset();
            std::cout << "core Vulkan init/free without NGU: status=" << result << " " << error << std::endl;
            return result;
        }
        load_plugin(core, argv[1]);
        if (large) {
            test_gpu_half_large(core,argv[2]);
        } else {
            for (int repeat = 0; repeat < 3; ++repeat) {
                test_parallel_lifetime(core, argv[2], "sharp");
                test_parallel_lifetime(core, argv[2], "aa");
                for (int kind = 0; kind < 3; ++kind) test_frame_error(core, argv[2], kind);
                test_creation_error(core, argv[2], false);
                test_creation_error(core, argv[2], true);
            }
            test_gpu_half_tiles(core,argv[2]);
            test_gpu_fallback(core);
            test_gpu_parallel_lifetime(core, argv[2], "sharp", false);
            test_gpu_parallel_lifetime(core, argv[2], "aa", false);
            test_gpu_parallel_lifetime(core, argv[2], "sharp", true);
            test_gpu_creation_errors(core, argv[2]);
            test_format_precision_errors(core);
            test_frame_error(core, argv[2], 1, true);
            for (const char *mode : {"sharp", "aa"}) {
                test_parallel_lifetime(core, argv[2], mode, 16, "fp16");
                test_gpu_parallel_lifetime(core, argv[2], mode, false, 16, "fp16");
                test_gpu_parallel_lifetime(core, argv[2], mode, true, 16, "fp16");
            }
        }
        api->clearCoreCaches(core);
        VSVulkanCoreInfo gpu_info{};
        char gpu_error[1024]{};
        check(api->getVulkanAPI()->getVulkanCoreInfo(core, &gpu_info, gpu_error, sizeof(gpu_error)) == 0,
            std::string("GPU accounting query: ") + gpu_error);
        check(gpu_info.allocated == 0, "GPU frames/scratch/weights were not released after cache sweep");
        core_guard.reset();
        if (!large) test_adjacent_weights(argv[1], argv[2]);
        if (!large) std::cout << "native VS: 320 CPU + 280 GPU async requests (8 threads), format-selected FP32 / FP16 archives; GPU chain, explicit "
                     "download, retained frames, properties, errors, GPU accounting zero and repeated lifecycle; "
                     "80 exact half-FMA tile graphs x default/compatible option, half/float fallback, precision guards and adjacent weights passed"
                  << std::endl;
        return 0;
    } catch (const std::exception &e) {
        // On failure exit nonzero; do not pretend an incomplete lifetime test
        // proves cleanup. ASan/LSan and the core's frame-ref diagnostics remain enabled.
        std::cerr << "native VS test: " << e.what() << "\n";
        return 1;
    }
}
