// holds the clocks of an NVIDIA GPU while the node has sessions
//
// on a network a node waits ~100+ ms for every token; meanwhile the GPU drops to P3 (memory at half clock, core at
// ~10%) and the next token runs 1.8x slower on one PC and 4-12x slower on home PCs (bench/RESULTS.md). Locking the
// clocks removes that completely. NVML comes with the driver and is loaded at run time; setting clocks needs
// administrator rights (root on Linux), without them the node works as before
#pragma once

#include <cstdio>
#include <string>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace agent {

class gpu_clocks {
  public:
    // pci_bus_id: "0000:01:00.0" from ggml_backend_dev_props::device_id, "" or null: the only GPU
    bool open(const char * pci_bus_id) {
#ifdef _WIN32
        lib = (void *) LoadLibraryA("nvml.dll");
        if (!lib) {
            char path[MAX_PATH];
            if (ExpandEnvironmentStringsA("%ProgramFiles%\\NVIDIA Corporation\\NVSMI\\nvml.dll", path, MAX_PATH)) {
                lib = (void *) LoadLibraryA(path);
            }
        }
#else
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
#endif
        if (!lib) {
            err = "no NVML (not an NVIDIA driver)";
            return false;
        }
        if (!sym(init, "nvmlInit_v2") || !sym(by_pci, "nvmlDeviceGetHandleByPciBusId_v2") || !sym(by_index, "nvmlDeviceGetHandleByIndex_v2") ||
            !sym(count, "nvmlDeviceGetCount_v2") || !sym(max_clock, "nvmlDeviceGetMaxClockInfo") ||
            !sym(lock_gpu, "nvmlDeviceSetGpuLockedClocks") || !sym(reset_gpu, "nvmlDeviceResetGpuLockedClocks") ||
            !sym(error_string, "nvmlErrorString")) {
            err = "NVML too old";
            return false;
        }
        sym(lock_mem, "nvmlDeviceSetMemoryLockedClocks");
        sym(reset_mem, "nvmlDeviceResetMemoryLockedClocks");
        if (int r = init(); r != 0) {
            err = std::string("nvmlInit: ") + error_string(r);
            return false;
        }
        int r = -1;
        if (pci_bus_id && *pci_bus_id) {
            r = by_pci(pci_bus_id, &dev);
        }
        if (r != 0) {
            unsigned n = 0;
            if (count(&n) != 0 || n != 1 || by_index(0, &dev) != 0) {
                err = "cannot find the GPU " + std::string(pci_bus_id ? pci_bus_id : "") + " in NVML";
                return false;
            }
        }
        unsigned g = 0, m = 0;
        if (max_clock(dev, 0 /* NVML_CLOCK_GRAPHICS */, &g) != 0 || g == 0) {
            err = "no max clock";
            return false;
        }
        max_clock(dev, 2 /* NVML_CLOCK_MEM */, &m);
        gpu_max = g;
        mem_max = m;
        return true;
    }

    // true: the clocks are held now
    bool lock() {
        if (locked) {
            return true;
        }
        // the lower bound keeps the GPU out of the power-saving states; the upper one leaves boosting to the driver
        const int r = lock_gpu(dev, gpu_max * 4 / 5, gpu_max);
        if (r != 0) {
            err = error_string(r);
            if (r == 4 /* NVML_ERROR_NO_PERMISSION */) {
                err += " (run the node as administrator)";
            }
            return false;
        }
        if (lock_mem && mem_max > 0) {
            mem_locked = lock_mem(dev, mem_max, mem_max) == 0; // not every GPU allows it; the core lock is the main part
        }
        locked = true;
        return true;
    }

    void unlock() {
        if (!locked) {
            return;
        }
        reset_gpu(dev);
        if (mem_locked && reset_mem) {
            reset_mem(dev);
        }
        locked = mem_locked = false;
    }

    ~gpu_clocks() { unlock(); }

    unsigned    gpu_max = 0, mem_max = 0;
    bool        locked = false, mem_locked = false;
    std::string err;

  private:
    template <typename F> bool sym(F & f, const char * name) {
#ifdef _WIN32
        f = (F) (void *) GetProcAddress((HMODULE) lib, name);
#else
        f = (F) dlsym(lib, name);
#endif
        return f != nullptr;
    }

    typedef struct nvml_device * nvml_device_t;
    void *        lib = nullptr;
    nvml_device_t dev = nullptr;
    int (*init)() = nullptr;
    int (*by_pci)(const char *, nvml_device_t *) = nullptr;
    int (*by_index)(unsigned, nvml_device_t *) = nullptr;
    int (*count)(unsigned *) = nullptr;
    int (*max_clock)(nvml_device_t, int, unsigned *) = nullptr;
    int (*lock_gpu)(nvml_device_t, unsigned, unsigned) = nullptr;
    int (*reset_gpu)(nvml_device_t) = nullptr;
    int (*lock_mem)(nvml_device_t, unsigned, unsigned) = nullptr;
    int (*reset_mem)(nvml_device_t) = nullptr;
    const char * (*error_string)(int) = nullptr;
};

} // namespace agent
