// the video memory of the node's device: free memory, and on Windows how much of this process went to shared memory
//
// When the weights or the context do not fit in the video memory, the Windows driver does not fail the allocation:
// it puts the rest into shared memory (RAM over PCIe), and the part runs several times slower (bench/RESULTS.md,
// bench-batch 01.11: 9 GB of layers with 7 GB free). DXGI tells the usage of shared memory by this process,
// the number Task Manager shows as "shared GPU memory"
#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <string>

#if defined(_WIN32) && defined(_MSC_VER)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <dxgi1_4.h>
// the COM headers define small as char (rpcndr.h): it breaks variables of that name
#    undef small
#    define AGENT_HAVE_DXGI
#endif

namespace agent {

struct vram_state {
    int64_t free   = -1; // bytes, -1: unknown (not a GPU)
    int64_t total  = -1;
    int64_t shared = -1; // shared memory used by this process, -1: unknown (not Windows)

    // shared memory of a process without overflow: pinned buffers of the backend, a few MB
    static constexpr int64_t shared_normal = 256ll << 20;

    // the weights or the context did not fit: bytes in shared memory, 0: they fit or unknown
    // (elsewhere the driver fails the allocation instead: Linux has no such fallback)
    int64_t over() const {
        return shared > shared_normal ? shared : 0;
    }
};

#ifdef AGENT_HAVE_DXGI
// the DXGI adapter of the device: the one with the same name, else the one with the most video memory
inline IDXGIAdapter3 * dxgi_adapter(const std::string & name) {
    using create_t = HRESULT (WINAPI *)(REFIID, void **);
    HMODULE lib = LoadLibraryA("dxgi.dll");
    create_t create = lib ? (create_t) GetProcAddress(lib, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory1 * factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void **) &factory))) {
        return nullptr;
    }
    IDXGIAdapter1 * best = nullptr;
    SIZE_T best_mem = 0;
    bool best_named = false;
    IDXGIAdapter1 * a = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 d = {};
        a->GetDesc1(&d);
        char desc[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, desc, sizeof(desc) - 1, nullptr, nullptr);
        const bool named = !name.empty() && name == desc;
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && (named || (!best_named && d.DedicatedVideoMemory > best_mem))) {
            if (best) {
                best->Release();
            }
            best = a;
            best_mem = d.DedicatedVideoMemory;
            best_named = named;
        } else {
            a->Release();
        }
    }
    factory->Release();
    IDXGIAdapter3 * out = nullptr;
    if (best) {
        best->QueryInterface(__uuidof(IDXGIAdapter3), (void **) &out);
        best->Release();
    }
    return out;
}
#endif

// dev: the GPU of the node, null: no GPU (all unknown)
inline vram_state vram_probe(ggml_backend_dev_t dev) {
    vram_state v;
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return v;
    }
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    v.free  = (int64_t) free;
    v.total = (int64_t) total;
#ifdef AGENT_HAVE_DXGI
    static IDXGIAdapter3 * adapter = dxgi_adapter(ggml_backend_dev_description(dev));
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (adapter && SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
        v.shared = (int64_t) info.CurrentUsage;
    }
#endif
    return v;
}

} // namespace agent
