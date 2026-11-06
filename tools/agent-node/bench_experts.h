#pragma once

// bench-experts: what the CPU of this PC does with the experts of an MoE model kept in RAM (--cpu-moe), by weight type
// and number of threads, against the RAM bandwidth. One measured step is the expert block of one layer of
// Qwen3.5-35B-A3B for one token (gate and up 2048 -> 512, down 512 -> 2048, 8 of 256 experts, SiLU between):
// what a node with --cpu-moe computes on the CPU per layer per token. The experts change between steps (32 sets
// in turn), so the weights come from RAM and not from the CPU cache. Weights are random: the speed does not
// depend on their values. Results and conclusions: docs/НИОКР ускорение узла.md (repo JunkyYankee1/-).
//
// Only the generic backend API: in the Windows build the CPU backend is a DLL (GGML_BACKEND_DL).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <cpuid.h>
#    endif
#endif

namespace bench_experts {

struct cpu_flags {
    bool avx2 = false, avx512f = false, avx512_vnni = false, avx512_vbmi = false, avx512_bf16 = false, avx_vnni = false;
};

static cpu_flags detect_cpu() {
    cpu_flags f;
#if defined(__x86_64__) || defined(_M_X64)
    int r[4] = { 0, 0, 0, 0 };
#    if defined(_MSC_VER)
    __cpuidex(r, 7, 0);
#    else
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
#    endif
    f.avx2        = (r[1] >> 5) & 1;
    f.avx512f     = (r[1] >> 16) & 1;
    f.avx512_vbmi = (r[2] >> 1) & 1;
    f.avx512_vnni = (r[2] >> 11) & 1;
#    if defined(_MSC_VER)
    __cpuidex(r, 7, 1);
#    else
    __cpuid_count(7, 1, r[0], r[1], r[2], r[3]);
#    endif
    f.avx_vnni    = (r[0] >> 4) & 1;
    f.avx512_bf16 = (r[0] >> 5) & 1;
#endif
    return f;
}

// the features the loaded CPU backend was built with (e.g. AVX2 = 1, AVX512 = 1)
static std::string backend_features(ggml_backend_reg_t reg) {
    struct feature { const char * name; const char * value; };
    typedef feature * (*get_features_t)(ggml_backend_reg_t);
    auto get = (get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    std::string out;
    for (feature * f = get ? get(reg) : nullptr; f && f->name; f++) {
        if (strcmp(f->value, "1") == 0) {
            out += (out.empty() ? "" : " ") + std::string(f->name);
        }
    }
    return out;
}

// RAM read bandwidth with n threads: each sums its part of a buffer much larger than the CPU caches
static double ram_gbps(const std::vector<uint64_t> & buf, int n) {
    std::atomic<uint64_t> sink { 0 };
    const size_t per = buf.size() / n;
    const auto t0 = std::chrono::steady_clock::now();
    double el = 0;
    int    reps = 0;
    do {
        std::vector<std::thread> th;
        for (int t = 0; t < n; t++) {
            th.emplace_back([&, t] {
                const uint64_t * p = buf.data() + t * per;
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (size_t i = 0; i + 4 <= per; i += 4) {
                    s0 += p[i]; s1 += p[i + 1]; s2 += p[i + 2]; s3 += p[i + 3];
                }
                sink += s0 + s1 + s2 + s3;
            });
        }
        for (auto & x : th) {
            x.join();
        }
        reps++;
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } while (el < 0.5);
    return (double) per * n * 8 * reps / el / 1e9;
}

struct variant {
    ggml_type type;
    bool      repack; // weights in the CPU_REPACK buffer type (Q4_0, Q4_K, IQ4_NL on AVX2), which --cpu-moe skips
    bool      stock = false; // IQ2_XXS, IQ3_XXS: the stock ggml kernel instead of the AVX-512 one (CPU feature IQ512)
};

struct result {
    std::string name;
    int         threads, tokens;
    double      us_per_layer, weight_gbps, bpw;
    std::vector<float> out; // the output of the block (first set of experts), in the first result of each token count
};

struct runner {
    ggml_backend_dev_t         dev;
    ggml_backend_t             be;
    ggml_backend_buffer_type_t repack = nullptr;
    void (*set_threads)(ggml_backend_t, int) = nullptr;
    void (*set_iq512)(int)                   = nullptr; // only in builds with the AVX-512 kernels

    runner() {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            throw std::runtime_error("bench-experts: no CPU backend");
        }
        be = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        set_threads = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        set_iq512 = (void (*)(int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_iq512");
        typedef ggml_backend_buffer_type_t * (*extra_t)(ggml_backend_dev_t);
        auto extra = (extra_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        for (ggml_backend_buffer_type_t * b = extra ? extra(dev) : nullptr; b && *b; b++) {
            if (strstr(ggml_backend_buft_name(*b), "REPACK")) {
                repack = *b;
            }
        }
    }
    ~runner() { ggml_backend_free(be); }

    // the expert block of one layer: microseconds per step for each (threads, tokens), the weights quantized once
    std::vector<result> run(const variant & v, const std::vector<int> & threads, const std::vector<int> & tokens, double secs) {
        const int64_t n_embd = 2048, n_ff = 512, n_exp = 256, n_used = 8, n_set = 32;
        if (set_iq512) {
            set_iq512(v.stock ? 0 : 1);
        }
        std::mt19937  rng(1);
        ggml_quantize_init(v.type);
        ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context *   wctx = ggml_init(wp);
        ggml_tensor *    W[3] = { ggml_new_tensor_3d(wctx, v.type, n_embd, n_ff, n_exp), ggml_new_tensor_3d(wctx, v.type, n_embd, n_ff, n_exp),
                                  ggml_new_tensor_3d(wctx, v.type, n_ff, n_embd, n_exp) };
        ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors_from_buft(wctx, v.repack ? repack : ggml_backend_dev_buffer_type(dev));
        if (!wb) {
            ggml_free(wctx);
            throw std::runtime_error("bench-experts: no memory for the weights of " + std::string(ggml_type_name(v.type)));
        }
        std::normal_distribution<float> nd(0.f, 0.02f);
        for (ggml_tensor * t : W) {
            // 64 rows quantized once and repeated: quantizing all rows of the low-bit types takes minutes
            const int64_t ne0 = t->ne[0], nrows = ggml_nrows(t), blk = 64;
            std::vector<float> src(ne0 * blk), imat(ne0, 1.0f);
            for (auto & x : src) {
                x = nd(rng);
            }
            const size_t      rs = ggml_row_size(v.type, ne0);
            std::vector<char> q(rs * blk), all(rs * nrows);
            ggml_quantize_chunk(v.type, src.data(), q.data(), 0, blk, ne0, imat.data());
            for (int64_t r = 0; r < nrows; r += blk) {
                memcpy(all.data() + r * rs, q.data(), rs * std::min(blk, nrows - r));
            }
            ggml_backend_tensor_set(t, all.data(), 0, all.size()); // the repack buffer repacks here
        }
        const double bytes_tok = (double) (ggml_row_size(v.type, n_embd) * n_ff * 2 + ggml_row_size(v.type, n_ff) * n_embd) * n_used;
        std::vector<result> out;
        for (int nt : tokens) {
            ggml_init_params gp = { ggml_tensor_overhead() * 16 * n_set + ggml_graph_overhead() * n_set, nullptr, true };
            ggml_context *   gctx = ggml_init(gp);
            ggml_tensor *    x    = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd, 1, nt);
            std::vector<ggml_tensor *> idss;
            std::vector<ggml_cgraph *> gfs;
            for (int s = 0; s < n_set; s++) {
                ggml_tensor * ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_used, nt);
                ggml_tensor * g   = ggml_mul_mat_id(gctx, W[0], x, ids);
                ggml_tensor * u   = ggml_mul_mat_id(gctx, W[1], x, ids);
                ggml_tensor * d   = ggml_mul_mat_id(gctx, W[2], ggml_mul(gctx, ggml_silu(gctx, g), u), ids);
                ggml_cgraph * gf  = ggml_new_graph(gctx);
                ggml_build_forward_expand(gf, d);
                idss.push_back(ids);
                gfs.push_back(gf);
            }
            ggml_backend_buffer_t ab = ggml_backend_alloc_ctx_tensors_from_buft(gctx, ggml_backend_dev_buffer_type(dev));
            std::vector<float>    xv(n_embd * nt);
            for (auto & e : xv) {
                e = nd(rng) * 50;
            }
            ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
            std::uniform_int_distribution<int> ud(0, (int) n_exp - 1);
            for (ggml_tensor * ids : idss) {
                std::vector<int32_t> iv(n_used * nt);
                for (int t = 0; t < nt; t++) {
                    for (int j = 0; j < n_used; j++) {
                        int e;
                        do {
                            e = ud(rng);
                        } while (std::find(iv.begin() + t * n_used, iv.begin() + t * n_used + j, e) != iv.begin() + t * n_used + j);
                        iv[t * n_used + j] = e;
                    }
                }
                ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
            }
            const size_t first = out.size();
            for (int th : threads) {
                if (set_threads) {
                    set_threads(be, th);
                }
                for (int w = 0; w < 4; w++) {
                    ggml_backend_graph_compute(be, gfs[w % n_set]);
                }
                int        reps = 0;
                double     el   = 0;
                const auto t0   = std::chrono::steady_clock::now();
                do {
                    ggml_backend_graph_compute(be, gfs[reps % n_set]);
                    reps++;
                    el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                } while (el < secs);
                const double us = el / reps * 1e6;
                result r;
                r.name         = std::string(ggml_type_name(v.type)) + (v.repack ? "+repack" : "") + (v.stock ? "-ggml" : "");
                r.threads      = th;
                r.tokens       = nt;
                r.us_per_layer = us;
                r.weight_gbps  = bytes_tok * nt / us / 1e3; // distinct experts per step: 8 per token (rarely shared)
                r.bpw          = 8.0 * ggml_row_size(v.type, n_embd) / n_embd;
                out.push_back(r);
            }
            // the result itself, to compare kernels (the same seed: the same weights, activations and experts)
            ggml_backend_graph_compute(be, gfs[0]);
            ggml_tensor * res = ggml_graph_node(gfs[0], -1);
            out[first].out.resize(ggml_nelements(res));
            ggml_backend_tensor_get(res, out[first].out.data(), 0, ggml_nbytes(res));
            ggml_backend_buffer_free(ab);
            ggml_free(gctx);
        }
        ggml_backend_buffer_free(wb);
        ggml_free(wctx);
        if (set_iq512) {
            set_iq512(1);
        }
        return out;
    }
};

} // namespace bench_experts
