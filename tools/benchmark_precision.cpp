#define VK_NO_PROTOTYPES
#include <VSVulkan4.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

const VSAPI *api;
struct Map {
    VSMap *p = api->createMap();
    ~Map() { api->freeMap(p); }
};
struct NodeDelete { void operator()(VSNode *p) const { if (p) api->freeNode(p); } };
using Node = std::unique_ptr<VSNode, NodeDelete>;
void check(bool ok, const std::string &s) { if (!ok) throw std::runtime_error(s); }
void integer(Map &m, const char *key, std::int64_t value) {
    check(!api->mapSetInt(m.p,key,value,maReplace),key);
}
void string(Map &m, const char *key, const std::string &value) {
    check(!api->mapSetData(m.p,key,value.c_str(),-1,dtUtf8,maReplace),key);
}
Node invoke(VSCore *core, const char *ns, const char *function, Map &in) {
    VSMap *out=api->invoke(api->getPluginByNamespace(ns,core),function,in.p);
    const char *error=api->mapGetError(out);
    const std::string message=error?error:"";
    VSNode *node=message.empty()?api->mapGetNode(out,"clip",0,nullptr):nullptr;
    api->freeMap(out);
    check(message.empty()&&node,message.empty()?"no output node":message);
    return Node(node);
}
struct Sync {
    const VSVULKANAPI *gpu = nullptr;
    VSGPUExecPool *pool = nullptr;
    ~Sync() { if (pool) gpu->freeGPUExecPool(pool); }
    void fence(const std::vector<const VSFrame *> &frames) {
        if (!pool) return;
        char error[1024]{};
        auto *exec=gpu->gpuExecAcquire(pool,error,sizeof(error));
        check(exec!=nullptr,error);
        for (auto *f:frames) gpu->gpuExecReadsFrame(exec,f);
        std::uint64_t value=0;
        check(!gpu->gpuExecSubmit(exec,&value,error,sizeof(error)),error);
        check(gpu->gpuExecWaitValue(pool,value,error,sizeof(error))==gdDrained,error);
    }
    void drain() {
        if (!pool) return;
        char error[1024]{};
        check(gpu->gpuExecPoolWaitIdle(pool,error,sizeof(error))==gdDrained,error);
    }
};
struct Completion {
    std::mutex mutex;
    std::condition_variable ready;
    int count=0;
    std::vector<const VSFrame *> frames;
    std::string error;
};
void VS_CC complete(void *user, const VSFrame *f, int, VSNode *, const char *error) {
    auto &c=*static_cast<Completion *>(user);
    std::lock_guard lock(c.mutex);
    if (error) c.error=error;
    if (f) c.frames.push_back(f);
    ++c.count;
    c.ready.notify_one();
}
struct Case {
    Node output;
    int bits;
    std::string precision;
    std::string compute;
    int next=0;
    std::vector<double> milliseconds;
};
double window(Case &c, int frames, Sync &sync) {
    Completion completion;
    completion.frames.reserve(frames);
    const auto start=std::chrono::steady_clock::now();
    for (int n=0;n<frames;++n)
        api->getFrameAsync(c.next++,c.output.get(),complete,&completion);
    {
        std::unique_lock lock(completion.mutex);
        completion.ready.wait(lock,[&]{return completion.count==frames;});
    }
    check(completion.error.empty(),completion.error);
    check(static_cast<int>(completion.frames.size())==frames,"missing frames");
    sync.fence(completion.frames); // Real GPU completion, not merely graph readiness.
    const auto stop=std::chrono::steady_clock::now();
    for (auto *f:completion.frames) {
        check(api->getVideoFrameFormat(f)->bitsPerSample==c.bits,"wrong format");
        check(api->getFrameResidency(f)==(sync.pool?nrGPU:nrCPU),"wrong residency");
        const char *precision=api->mapGetData(api->getFramePropertiesRO(f),"_NGUPrecision",0,nullptr);
        check(precision!=nullptr,"missing compute precision property");
        if (c.compute.empty()) c.compute=precision;
        check(c.compute==precision,"compute precision changed");
        const char *weights=api->mapGetData(api->getFramePropertiesRO(f),"_NGUWeightPrecision",0,nullptr);
        check(weights&&c.precision==weights,"weight precision does not match input format");
        api->freeFrame(f);
    }
    sync.drain(); // Release benchmark-owned references outside the measured window.
    return std::chrono::duration<double,std::milli>(stop-start).count()/frames;
}
int main(int argc,char **argv) {
    try {
        check(argc==10||argc==11,"plugin root width height quality backend frames trials warmup [mode]");
        const std::filesystem::path root=argv[2];
        const int width=std::stoi(argv[3]),height=std::stoi(argv[4]);
        const std::string quality=argv[5],backend=argv[6],mode=argc==11?argv[10]:"sharp";
        const int frames=std::stoi(argv[7]),trials=std::stoi(argv[8]),warm=std::stoi(argv[9]);
        check(frames>0&&trials>0&&warm>0,"positive counts required");
        const bool gpu=backend=="vulkan";
        api=getVapourSynthAPI(VS_MAKE_VERSION(4,3));
        check(api!=nullptr,"API 4.3 required");
        std::unique_ptr<VSCore,decltype(api->freeCore)> core(api->createCore(ccfDisableAutoLoading),api->freeCore);
        check(core!=nullptr,"core creation");
        api->setThreadCount(8,core.get());
        Map load; string(load,"path",argv[1]);
        VSMap *loaded=api->invoke(api->getPluginByNamespace("std",core.get()),"LoadPlugin",load.p);
        const char *e=api->mapGetError(loaded);
        const std::string load_error=e?e:"";
        api->freeMap(loaded);
        check(load_error.empty(),load_error);
        Sync sync;
        if (gpu) {
            sync.gpu=api->getVulkanAPI();
            char error[1024]{};
            sync.pool=sync.gpu->createGPUExecPool(core.get(),vqCompute,error,sizeof(error));
            check(sync.pool!=nullptr,error);
        }
        std::vector<Case> cases;
        for (int bits:{32,16}) {
            const char *precision=bits==16?"fp16":"fp32";
            Map args;
            integer(args,"width",width); integer(args,"height",height);
            integer(args,"format",bits==16?pfGrayH:pfGrayS); integer(args,"length",1);
            api->mapSetFloat(args.p,"color",0.375,maReplace);
            Node input=invoke(core.get(),"std","BlankClip",args);
            if (gpu) {
                Map upload; api->mapSetNode(upload.p,"clip",input.get(),maReplace);
                input=invoke(core.get(),"std","GPUUpload",upload);
                api->setCacheMode(input.get(),cmForceEnable);
                api->setCacheOptions(input.get(),1,1,1);
                char error[1024]{};
                const VSFrame *first=api->getFrame(0,input.get(),error,sizeof(error));
                check(first!=nullptr,error);
                sync.fence({first});
                api->freeFrame(first);
                sync.drain();
            }
            Map loop; api->mapSetNode(loop.p,"clip",input.get(),maReplace);
            integer(loop,"times",warm+trials*frames);
            input=invoke(core.get(),"std","Loop",loop);
            check(api->getNodeResidency(input.get())==(gpu?nrGPU:nrCPU),"Loop inserted transfer");
            Map ngu; api->mapSetNode(ngu.p,"clip",input.get(),maReplace);
            string(ngu,"weights",root.string());
            string(ngu,"mode",mode); string(ngu,"quality",quality); integer(ngu,"option",0);
            string(ngu,"backend",backend);
            Node output=invoke(core.get(),"ngu","NGU2x",ngu);
            api->setCacheMode(output.get(),cmForceDisable);
            cases.push_back(Case{std::move(output),bits,precision,"",0,{}});
        }
        // All shader compilation, uploads and workspace warm-up precede timing.
        for (auto &c:cases) window(c,warm,sync);
        // Only the two format-matched archives are legal now. Alternate
        // their order between trials; archived four-way runs are historical.
        for (int trial=0;trial<trials;++trial) for (int j=0;j<2;++j) {
            auto &c=cases[trial%2?1-j:j];
            c.milliseconds.push_back(window(c,frames,sync));
        }
        std::cout<<std::setprecision(10);
        for (auto &c:cases) {
            auto sorted=c.milliseconds; std::sort(sorted.begin(),sorted.end());
            double median=sorted[sorted.size()/2];
            std::cout<<"{\"backend\":\""<<backend<<"\",\"quality\":\""<<quality
                     <<"\",\"width\":"<<width<<",\"height\":"<<height
                     <<",\"mode\":\""<<mode<<"\",\"weights\":\""<<c.precision<<"\",\"clip_bits\":"<<c.bits
                     <<",\"compute_precision\":\""<<c.compute<<"\""
                     <<",\"frames_per_trial\":"<<frames<<",\"trials\":"<<trials
                     <<",\"warmup_frames\":"<<warm<<",\"threads\":8,\"median_ms\":"<<median
                     <<",\"fps\":"<<1000/median<<",\"trial_ms\":[";
            for (std::size_t i=0;i<c.milliseconds.size();++i)
                std::cout<<(i?",":"")<<c.milliseconds[i];
            std::cout<<"]}\n";
        }
        cases.clear(); sync.drain(); api->clearCoreCaches(core.get());
        return 0;
    } catch (const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
}
