// agent-node: one part of a model in a chain of nodes, plus the client that drives the chain
//
//   agent-node node      -m model.gguf --layers 0:12 --port 50152
//   agent-node client    -m model.gguf --chain 127.0.0.1:50152,127.0.0.1:50153 -p "Hello" -n 64
//   agent-node reference -m model.gguf -p "Hello" -n 64 --dump ref.txt
//   agent-node compare   chain.txt ref.txt
//
// the client sends tokens to the first node, every node sends its hidden states (int8) straight to the next one,
// the last node sends the top-k logits back to the client: one transfer per node per token

#include "coordinator.h"
#include "gpu_clocks.h"
#include "p2p.h"
#include "protocol.h"
#include "secure.h"
#include "slice.h"

#ifdef _WIN32
#    include <windows.h>
#    include <shellapi.h>
#    undef X509_NAME
#    undef X509_EXTENSIONS
#    undef OCSP_REQUEST
#    undef OCSP_RESPONSE
#endif
#include "vram.h"

#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-cpp.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <deque>
#include <condition_variable>
#include <set>
#include <random>
#include <sstream>
#include <thread>

using namespace agent;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static int g_verbose = 0;

#define AN_LOG(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while (0)

static void log_callback(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

//
// command line
//

struct cli_args {
    std::map<std::string, std::string> opts;
    std::vector<std::string>           positional;

    static bool is_flag(const std::string & name) {
        return name == "--ignore-eos" || name == "--no-chat" || name == "--same-tokens" || name == "--tls" || name == "-v" || name == "--verbose" ||
               name == "--vocab-only" || name == "--cpu-moe" || name == "--check";
    }

    cli_args(int argc, char ** argv, int first) {
        for (int i = first; i < argc; i++) {
            const std::string a = argv[i];
            if (a.size() > 1 && a[0] == '-' && !std::isdigit((unsigned char) a[1])) {
                if (is_flag(a)) {
                    opts[a] = "1";
                } else if (i + 1 < argc) {
                    opts[a] = argv[++i];
                } else {
                    throw std::runtime_error("missing value for " + a);
                }
            } else {
                positional.push_back(a);
            }
        }
        if (has("-v") || has("--verbose")) {
            g_verbose = 1;
        }
        // an unknown option must not be ignored silently (an old build would run without a feature asked for)
        static const char * known[] = {
            "--chain", "--ctx-size", "--device", "--draft", "--draft-max", "--draft-ngl", "--draft-noise", "--dump", "--file",
            "--force", "--hidden", "--host", "--ignore-eos", "--json", "--layers", "--model", "--n-gpu-layers", "--n-predict",
            "--no-chat", "--port", "--prompt", "--same-tokens", "--threads", "--tokens", "--tol", "--top", "--top1", "--verbose",
            "--keys", "--pins", "--spare", "--spare-keys", "--spare-tokens", "--spare-pins", "--tls", "--timeout", "--session",
            "--token-ttl", "--key", "--identity", "--ttl", "--coordinator", "--public-addr", "--city", "--spares",
            "--keep-warm", "--keep-warm-mb", "--gpu-clocks",
            "--account", "--parent-pid", "--chat-template", "--test-corrupt", "--verifier-token", "--bind-host", "--p2p", "--rendezvous", "--secret", "--bytes", "--every", "--rounds",
            "--listen-host", "--listen-port", "--relay", "--cpu-moe", "--ram-gb", "--from", "--out", "--vocab-only", "--model-url", "--models-dir", "--disk-gb", "--vram-gb", "--vram-reserve-mb", "--mirror", "--seqs", "--check",
            "--max-sessions", "--seq-ctx", "--seq-rollback", "--vram-margin-mb", "--batch-max", "--keep-chain-s", "--gpu-experts", "--moe-cache-mib",
            "-c", "-dev", "-f", "-m", "-n", "-ngl", "-ngld", "-p", "-t", "-v",
        };
        for (const auto & kv : opts) {
            const std::string & name = kv.first;
            if (std::find_if(std::begin(known), std::end(known), [&](const char * k) { return name == k; }) == std::end(known)) {
                throw std::runtime_error("unknown option " + name);
            }
        }
    }

    bool has(const std::string & name) const { return opts.count(name) > 0; }

    std::string get(const std::string & name, const std::string & alt, const std::string & def) const {
        auto it = opts.find(name);
        if (it == opts.end() && !alt.empty()) {
            it = opts.find(alt);
        }
        return it == opts.end() ? def : it->second;
    }
    int get_int(const std::string & name, const std::string & alt, int def) const {
        return std::stoi(get(name, alt, std::to_string(def)));
    }
};

// file paths are UTF-8; on Windows fopen and the narrow fstream constructors take the ANSI code page,
// so paths with non-ASCII characters (e.g. a Cyrillic user folder) go through the wide-char API
static std::filesystem::path u8_path(const std::string & path) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(path.begin(), path.end()));
#else
    return std::filesystem::u8path(path);
#endif
}

static FILE * fopen_utf8(const std::string & path, const char * mode) {
#ifdef _WIN32
    const std::wstring wmode(mode, mode + strlen(mode));
    return _wfopen(u8_path(path).c_str(), wmode.c_str());
#else
    return fopen(path.c_str(), mode);
#endif
}

static std::string read_file(const std::string & path) {
    std::ifstream f(u8_path(path), std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

//
// model loading
//

struct model_opts {
    std::string path;
    int32_t     layer_start = 0;
    int32_t     layer_end   = 0;
    int32_t     n_gpu_layers = 999;
    std::string device;
    bool        vocab_only = false;
    bool        cpu_moe    = false; // the experts of an MoE model in RAM, the rest on the GPU
    // with cpu_moe: the experts of the last layers of the part stay on the GPU (the free video memory holds them;
    // a layer with its experts on the CPU runs 4 times slower, docs/Параллельные сессии.md)
    int32_t     gpu_expert_layers = 0;
    // with cpu_moe: a cache on the GPU for the experts in RAM (llama.cpp, per context): of the 256 experts of a layer
    // 8 work per token and the same ones are needed often, the cache keeps them. On an RTX 5060 Ti with 3 GB of cache
    // 4 sessions get 2.3 times the tokens per second of all experts on the CPU, 10 layers of experts on the GPU 1.4 times
    uint64_t    moe_cache_bytes = 0;
};

// the number of layers from the metadata of a GGUF file, without loading it
static int32_t gguf_n_layer(const std::string & path) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        throw std::runtime_error("cannot read the GGUF metadata of " + path);
    }
    int32_t n = 0;
    const int64_t a = gguf_find_key(g, "general.architecture");
    if (a >= 0 && gguf_get_kv_type(g, a) == GGUF_TYPE_STRING) {
        const int64_t id = gguf_find_key(g, (std::string(gguf_get_val_str(g, a)) + ".block_count").c_str());
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32) {
            n = (int32_t) gguf_get_val_u32(g, id);
        }
    }
    gguf_free(g);
    return n;
}

static llama_model_ptr load_model(const model_opts & mo) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = mo.n_gpu_layers;
    mp.layer_start  = mo.layer_start;
    mp.layer_end    = mo.layer_end;
    mp.vocab_only   = mo.vocab_only;

    // the experts take most of an MoE layer and only a few of them work per token: in RAM a node holds more layers.
    // The experts of the last gpu_expert_layers layers stay on the GPU; the patterns live until the model is loaded
    static const llama_model_tensor_buft_override cpu_moe[] = { llm_ffn_exps_cpu_override(), { nullptr, nullptr } };
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;
    if (mo.cpu_moe && !mo.vocab_only) {
        mp.tensor_buft_overrides = cpu_moe;
        if (mo.gpu_expert_layers > 0) {
            const int32_t first = std::max(mo.layer_start, 0);
            const int32_t last  = mo.layer_end > 0 ? mo.layer_end : gguf_n_layer(mo.path);
            for (int32_t l = first; l < last - mo.gpu_expert_layers; l++) {
                patterns.push_back(llm_ffn_block_regex(l, LLM_FFN_EXPS_REGEX));
            }
            for (const auto & p : patterns) {
                overrides.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
            }
            overrides.push_back({ nullptr, nullptr });
            mp.tensor_buft_overrides = overrides.data();
        }
    }

    std::vector<ggml_backend_dev_t> devs;
    if (!mo.device.empty() && !mo.vocab_only) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(mo.device.c_str());
        if (dev == nullptr) {
            std::string names;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                names += std::string(" ") + ggml_backend_dev_name(ggml_backend_dev_get(i));
            }
            throw std::runtime_error("unknown device " + mo.device + ", available:" + names);
        }
        devs = { dev, nullptr };
        mp.devices = devs.data();
    }

    llama_model_ptr model(llama_model_load_from_file(mo.path.c_str(), mp));
    if (!model) {
        throw std::runtime_error("failed to load model " + mo.path);
    }
    return model;
}

static model_opts model_opts_from(const cli_args & a) {
    model_opts mo;
    mo.path = a.get("-m", "--model", "");
    if (mo.path.empty()) {
        throw std::runtime_error("-m model.gguf is required");
    }
    mo.n_gpu_layers = a.get_int("-ngl", "--n-gpu-layers", 999);
    mo.device       = a.get("--device", "-dev", "");
    mo.cpu_moe      = a.has("--cpu-moe");
    mo.gpu_expert_layers = a.get_int("--gpu-experts", "", 0);
    mo.moe_cache_bytes   = mo.cpu_moe ? (uint64_t) a.get_int("--moe-cache-mib", "", 0) << 20 : 0;
    const std::string layers = a.get("--layers", "", "");
    if (!layers.empty()) {
        const size_t p = layers.find(':');
        if (p == std::string::npos) {
            throw std::runtime_error("--layers must be start:end, e.g. 0:12 or 24:");
        }
        mo.layer_start = std::stoi(layers.substr(0, p));
        mo.layer_end   = p + 1 < layers.size() ? std::stoi(layers.substr(p + 1)) : 0;
    }
    return mo;
}

static std::string model_desc(const llama_model * model) {
    char buf[256];
    llama_model_desc(model, buf, sizeof(buf));
    return buf;
}

//
// one part of the model: [layer_start, layer_end) in this process
//

// the k largest logits, sorted descending (ties: the smaller id first), in one pass without allocations:
// a vocabulary of 248k tokens made a sort per token cost milliseconds, for every checked draft token
static void top_k(const float * logits, int32_t n_vocab, uint32_t k, top_entry * out) {
    k = std::min<uint32_t>(k, (uint32_t) n_vocab);
    uint32_t n = 0;
    for (int32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (n == k && !(v > out[k - 1].logit)) {
            continue;
        }
        uint32_t j = n < k ? n++ : k - 1;
        while (j > 0 && out[j - 1].logit < v) {
            out[j] = out[j - 1];
            j--;
        }
        out[j] = { i, v };
    }
}

// one forward frame of one sequence; the forwards of several sequences run in one pass (model_part::run_batch)
struct part_job {
    forward_hdr     h;
    const uint8_t * data  = nullptr;
    llama_seq_id    seq   = 0;
    uint32_t        n_top = 8;

    // results: not the last part: hidden gets n_tokens rows of n_embd, the last part: top gets n_out rows of n_top
    std::vector<float>     hidden;
    std::vector<top_entry> top;
    uint32_t               n_out = 0;
    std::string            err;          // this forward failed; the other forwards of the pass are not affected
    double                 pass_ms = 0;  // time of the pass and of reading its outputs
    double                 output_ms = 0;
    uint32_t               n_pass  = 0;  // forwards in the pass

    std::vector<float> hidden_in; // the hidden states of the input as f32: the batch only points at them
    int32_t            i0 = 0;    // batch index of the first token
};

struct model_part {
    const llama_model * model = nullptr;
    llama_context_ptr   ctx;
    int32_t  first = 0, last = 0, n_layer = 0, n_embd = 0, n_vocab = 0;
    int32_t  n_batch = 512;
    uint32_t n_seq = 1, n_rollback = 0;
    std::vector<int32_t> n_past; // tokens of each sequence in the memory: its next forward continues at this position

    // saved states of a sequence (FORWARD_CHECKPOINT): the recurrent layers at a position; the KV cache before it stays
    struct checkpoint {
        int32_t              pos = 0;
        std::vector<uint8_t> data;
    };
    static constexpr size_t max_checkpoints = 4; // per sequence, the oldest goes first (the client counts the same way)
    std::vector<std::vector<checkpoint>> ckpts;
    double   last_output_ms = 0; // time of reading the outputs in the last pass

    // n_rollback: how many trailing tokens of a forward can be rolled back (speculative decoding)
    // n_seq > 1: one context for several sequences, n_ctx tokens shared by them all (a unified cache: llama.cpp puts
    // sequences with separate caches into one pass only when their ids follow one another, and sessions end in any order)
    model_part(const llama_model * model, const model_opts & mo, uint32_t n_ctx, int32_t n_threads, uint32_t n_rollback = 0,
            uint32_t n_seq = 1, uint64_t moe_cache = 0) : model(model), n_seq(n_seq), n_rollback(n_rollback), n_past(n_seq, 0), ckpts(n_seq) {
        if (n_seq == 0 || n_seq > 64) {
            throw std::runtime_error("1..64 sequences in a context");
        }
        n_layer = llama_model_n_layer(model);
        n_embd  = llama_model_n_embd(model);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        first   = std::clamp(mo.layer_start, 0, n_layer);
        last    = mo.layer_end <= 0 ? n_layer : std::min(mo.layer_end, n_layer);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = n_batch;
        cp.n_ubatch        = n_batch;
        cp.n_seq_max       = n_seq;
        cp.kv_unified      = n_seq > 1;
        // recurrent layers (Gated DeltaNet) keep snapshots of their state to undo rejected draft tokens
        cp.n_rs_seq        = n_rollback;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.no_perf         = true;
        cp.moe_cache_size  = (size_t) moe_cache;
        if (!is_last()) {
            cp.embeddings   = true;
            cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        }
        ctx.reset(llama_init_from_model(const_cast<llama_model *>(model), cp));
        if (!ctx) {
            throw std::runtime_error("failed to create the context");
        }
    }

    bool is_first() const { return first == 0; }
    bool is_last () const { return last == n_layer; }

    // the sequence ended: its tokens leave the memory
    void clear(llama_seq_id seq) {
        llama_memory_seq_rm(llama_get_memory(ctx.get()), seq, -1, -1);
        n_past[seq] = 0;
        ckpts[seq].clear();
    }

    // runs the tokens or the hidden states of one forward frame of sequence 0
    void run(const forward_hdr & h, const uint8_t * data,
            std::vector<float> & hidden_out, std::vector<top_entry> & top, uint32_t n_top, uint32_t & n_out) {
        part_job j;
        j.h     = h;
        j.data  = data;
        j.n_top = n_top;
        part_job * jobs[] = { &j };
        run_batch(jobs, 1);
        if (!j.err.empty()) {
            throw std::runtime_error(j.err);
        }
        hidden_out.swap(j.hidden);
        top.swap(j.top);
        n_out = j.n_out;
    }

    // runs the forwards of different sequences in one pass, at most n_batch tokens in all
    void run_batch(part_job * const * jobs, size_t n_jobs) {
        common_batch batch(ctx.get());
        std::vector<part_job *> run;
        uint64_t seqs = 0; // up to 64 sequences
        for (size_t k = 0; k < n_jobs; k++) {
            part_job & j = *jobs[k];
            j.err.clear();
            try {
                add(j, batch, seqs);
                run.push_back(&j);
            } catch (const std::exception & e) {
                j.err = e.what();
            }
        }
        if (run.empty()) {
            return;
        }

        const double t0 = now_ms();
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            for (part_job * j : run) {
                j->err = "llama_process failed at position " + std::to_string(j->h.pos0);
            }
            return;
        }
        const double t_out = now_ms();
        for (part_job * j : run) {
            n_past[j->seq] = j->h.pos0 + (int32_t) j->h.n_tokens;
            j->n_pass = (uint32_t) run.size();
            if (j->h.flags & FORWARD_CHECKPOINT) {
                try {
                    save_checkpoint(j->seq);
                } catch (const std::exception & e) {
                    j->err = e.what();
                }
            }
            try {
                read(*j);
            } catch (const std::exception & e) {
                j->err = e.what();
            }
        }
        last_output_ms = now_ms() - t_out;
        for (part_job * j : run) {
            j->pass_ms   = now_ms() - t0;
            j->output_ms = last_output_ms;
        }
    }

private:
    void save_checkpoint(llama_seq_id seq) {
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        auto & cs = ckpts[seq];
        cs.erase(std::remove_if(cs.begin(), cs.end(), [&](const checkpoint & c) { return c.pos == n_past[seq]; }), cs.end());
        if (cs.size() >= max_checkpoints) {
            cs.erase(cs.begin());
        }
        checkpoint c;
        c.pos = n_past[seq];
        c.data.resize(llama_state_seq_get_size_ext(ctx.get(), seq, fl));
        if (llama_state_seq_get_data_ext(ctx.get(), c.data.data(), c.data.size(), seq, fl) != c.data.size()) {
            throw std::runtime_error("cannot save the state of the session");
        }
        cs.push_back(std::move(c));
    }

    // the next forward of seq starts at pos, before the end: the rejected draft tokens (the rollback snapshots of the
    // recurrent layers cover the last n_rollback tokens) or the start of a new request at a checkpoint
    void roll_back(llama_seq_id seq, int32_t pos) {
        llama_memory_t mem = llama_get_memory(ctx.get());
        int32_t & past = n_past[seq];
        auto & cs = ckpts[seq];
        // the states after pos belong to the text that is dropped now
        cs.erase(std::remove_if(cs.begin(), cs.end(), [&](const checkpoint & c) { return c.pos > pos; }), cs.end());
        if (!llama_memory_seq_rm(mem, seq, pos, -1)) {
            auto it = std::find_if(cs.begin(), cs.end(), [&](const checkpoint & c) { return c.pos == pos; });
            const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            if (it == cs.end() || llama_state_seq_set_data_ext(ctx.get(), it->data.data(), it->data.size(), seq, fl) != it->data.size() ||
                    !llama_memory_seq_rm(mem, seq, pos, -1)) {
                throw std::runtime_error("cannot roll back from " + std::to_string(past) + " to " + std::to_string(pos) +
                    " (at most n_rollback tokens of the last forward, or to a checkpoint)");
            }
        }
        past = pos;
    }

    void add(part_job & j, common_batch & batch, uint64_t & seqs) {
        const forward_hdr & h = j.h;
        if (j.seq < 0 || (uint32_t) j.seq >= n_seq) {
            throw std::runtime_error("sequence " + std::to_string(j.seq) + " out of range");
        }
        if (seqs & (1ull << j.seq)) {
            throw std::runtime_error("two forwards of one sequence in a pass");
        }
        if (h.n_tokens == 0 || (int32_t) h.n_tokens > n_batch) {
            throw std::runtime_error("forward must carry 1.." + std::to_string(n_batch) + " tokens");
        }
        if (batch.size() + (int32_t) h.n_tokens > n_batch) {
            throw std::runtime_error("the pass is full");
        }
        const bool out_all = h.flags & FORWARD_OUT_ALL;
        if (h.kind == payload::tokens) {
            if (!is_first()) {
                throw std::runtime_error("this node runs layers from " + std::to_string(first) + " and takes hidden states, not tokens");
            }
        } else if (h.kind == payload::f16 || h.kind == payload::f32 || h.kind == payload::q8) {
            if (is_first()) {
                throw std::runtime_error("the first node takes tokens, not hidden states");
            }
        } else {
            throw std::runtime_error("unknown payload kind");
        }

        // a forward that starts before the end of the previous one drops the rejected draft tokens
        int32_t & past = n_past[j.seq];
        if (h.pos0 > past) {
            throw std::runtime_error("forward at position " + std::to_string(h.pos0) + " leaves a gap after " + std::to_string(past));
        }
        if (h.pos0 < past) {
            roll_back(j.seq, h.pos0);
        }

        seqs |= 1ull << j.seq;
        j.i0 = batch.size();
        if (h.kind == payload::tokens) {
            const int32_t * ids = (const int32_t *) j.data;
            for (uint32_t i = 0; i < h.n_tokens; i++) {
                const bool out = !is_last() || out_all || i + 1 == h.n_tokens;
                batch.add(ids[i], h.pos0 + (int32_t) i, j.seq, out);
            }
            return;
        }
        j.hidden_in.resize((size_t) h.n_tokens * n_embd);
        if (h.kind == payload::f16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) j.data, j.hidden_in.data(), (int64_t) j.hidden_in.size());
        } else if (h.kind == payload::q8) {
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(j.data, j.hidden_in.data(), (int64_t) j.hidden_in.size());
        } else {
            memcpy(j.hidden_in.data(), j.data, j.hidden_in.size() * sizeof(float));
        }
        for (uint32_t i = 0; i < h.n_tokens; i++) {
            const llama_pos p = h.pos0 + (llama_pos) i;
            // text positions, for M-RoPE they expand to [p, p, p, 0]
            const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
            const bool out = !is_last() || out_all || i + 1 == h.n_tokens;
            batch.add_embd({ j.hidden_in.data() + (size_t) i * n_embd, 1, (size_t) n_embd }, pos, j.seq, out);
        }
    }

    void read(part_job & j) {
        const forward_hdr & h = j.h;
        if (!is_last()) {
            j.hidden.resize((size_t) h.n_tokens * n_embd);
            for (uint32_t i = 0; i < h.n_tokens; i++) {
                const float * e = llama_get_embeddings_ith(ctx.get(), j.i0 + (int32_t) i);
                if (e == nullptr) {
                    throw std::runtime_error("no hidden state for token " + std::to_string(i));
                }
                memcpy(j.hidden.data() + (size_t) i * n_embd, e, (size_t) n_embd * sizeof(float));
            }
            j.n_out = 0;
            return;
        }

        const bool out_all = h.flags & FORWARD_OUT_ALL;
        j.n_out = out_all ? h.n_tokens : 1;
        j.top.resize((size_t) j.n_out * j.n_top);
        for (uint32_t o = 0; o < j.n_out; o++) {
            const int32_t i = j.i0 + (out_all ? (int32_t) o : (int32_t) h.n_tokens - 1);
            const float * logits = llama_get_logits_ith(ctx.get(), i);
            if (logits == nullptr) {
                throw std::runtime_error("no logits for token " + std::to_string(i - j.i0));
            }
            top_k(logits, n_vocab, j.n_top, j.top.data() + (size_t) o * j.n_top);
        }
    }
};

//
// node: serves one part of the model to clients and to the previous node
//

// sessions that ended in this process: the node tells the coordinator at once (coord_link::ended)
static std::atomic<uint64_t> g_sessions_ended{0};

struct session {
    ~session() {
        g_sessions_ended++;
        if (n_forward > 0) {
            AN_LOG("session %" PRIu64 ": %" PRIu64 " forwards, %" PRIu64 " of them in passes with other sessions\n", id, n_forward, n_shared);
        }
        if (release) {
            release();
        }
        // the connection to the next node belongs to the session: closing it ends the thread that reads it and,
        // on the next node, the side of the session that came from here
        if (next) {
            next->close();
        }
    }
    uint64_t                    id = 0;
    llama_seq_id                seq   = -1; // in the context of the node
    uint32_t                    n_ctx = 0;
    std::function<void()>       release;    // gives the sequence back to the node
    uint64_t                    n_forward = 0, n_shared = 0;
    uint32_t                    n_top = 8;
    payload                     hidden = payload::f16;
    tcp_conn_ptr                owner; // the client that opened the session, gets the results of the last node
    tcp_conn_ptr                next;  // the next node, null for the last node
    std::mutex                  mtx;
    stats_msg                   st;
};

// keeps the GPU of a node awake between tokens. On a network the node waits ~100+ ms for every token; meanwhile the GPU
// lowers its clocks and the next token runs 1.8x slower on one PC and 4-12x slower on home PCs (bench/RESULTS.md),
// and the power setting of the NVIDIA control panel does not help CUDA. While a session is open and the node waits,
// a small memory-bound job (adding two buffers) runs every interval on its own backend instance
struct gpu_warmer {
    ggml_backend_t        be  = nullptr;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    std::atomic<bool>     stop{ false };
    std::atomic<int64_t>  n_jobs{ 0 };
    std::thread           th;

    // false: no GPU for this device (CPU nodes need no warming)
    bool init(const std::string & device, size_t mb) {
        ggml_backend_dev_t dev = device.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)
                                                : ggml_backend_dev_by_name(device.c_str());
        if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            return false;
        }
        be = ggml_backend_dev_init(dev, nullptr);
        if (be == nullptr) {
            return false;
        }
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        const int64_t n = (int64_t) (std::max<size_t>(mb, 1) << 20) / 3 / (int64_t) sizeof(float); // a, b and a + b
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_add(ctx, a, b));
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (buf == nullptr) {
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        return true;
    }

    double run() {
        const double t0 = now_ms();
        ggml_backend_graph_compute(be, gf);
        n_jobs++;
        return now_ms() - t0;
    }

    // idle(): a session is open and the node has been waiting for at least the interval
    void start(int interval_ms, std::function<bool()> idle) {
        th = std::thread([this, interval_ms, idle]() {
            while (!stop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
                if (idle()) {
                    run();
                }
            }
        });
    }

    ~gpu_warmer() {
        stop = true;
        if (th.joinable()) {
            th.join();
        }
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (be)  ggml_backend_free(be);
    }
};

struct node_server {
    llama_model_ptr model;
    model_opts      mo;
    int32_t         n_threads = 4;
    uint32_t        max_sessions = 4;

    // one context for all sessions (a sequence each): the worker runs the forwards that wait at the moment
    // in one pass, and the weights of the part are read once for them all (docs/Параллельные сессии.md)
    std::unique_ptr<model_part> part;
    uint32_t                    batch_max = 4;  // forwards in one pass; 1: one at a time

    struct fwd_job : part_job {
        bool done    = false;
        bool waited  = false; // a long forward let the short ones go first once
    };
    // a forward this long (reading a prompt) waits for one pass when short ones (generation steps) wait too: else
    // every session at the node would wait for it, and on a slow node their clients time out
    static constexpr uint32_t long_forward = 64;
    std::mutex                q_mtx;
    std::condition_variable   q_cv, q_done;
    std::deque<fwd_job *>     queue;
    std::vector<llama_seq_id> free_seqs, ended_seqs; // ended: the worker clears them first
    uint64_t                  ctx_used = 0;          // tokens of the context promised to open sessions
    // since the start: forwards, and those that ran in a pass with other sessions
    std::atomic<uint64_t>     n_forwards{ 0 }, n_shared{ 0 };

    void start_worker() {
        for (uint32_t i = part->n_seq; i-- > 0;) {
            free_seqs.push_back((llama_seq_id) i);
        }
        std::thread([this]() { worker(); }).detach();
    }

    void worker() {
        std::vector<fwd_job *>    pass;
        std::vector<llama_seq_id> ended;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(q_mtx);
                q_cv.wait(lock, [&]() { return !queue.empty() || !ended_seqs.empty(); });
                ended.swap(ended_seqs);
                // in the order of arrival; a session has at most one forward in the queue (its connection waits for it)
                pass.clear();
                uint64_t seqs = 0;
                int32_t  n_tokens = 0;
                const bool short_waits = std::any_of(queue.begin(), queue.end(), [](const fwd_job * j) { return j->h.n_tokens < long_forward; });
                for (auto it = queue.begin(); it != queue.end() && pass.size() < batch_max;) {
                    fwd_job * j = *it;
                    if (short_waits && j->h.n_tokens >= long_forward && !j->waited) {
                        j->waited = true;
                        ++it;
                        continue;
                    }
                    if ((seqs & (1ull << j->seq)) || (!pass.empty() && n_tokens + (int32_t) j->h.n_tokens > part->n_batch)) {
                        ++it;
                        continue;
                    }
                    seqs |= 1ull << j->seq;
                    n_tokens += (int32_t) j->h.n_tokens;
                    pass.push_back(j);
                    it = queue.erase(it);
                }
            }
            if (!ended.empty()) {
                for (llama_seq_id seq : ended) {
                    part->clear(seq);
                }
                std::lock_guard<std::mutex> lock(q_mtx);
                free_seqs.insert(free_seqs.end(), ended.begin(), ended.end());
                ended.clear();
            }
            if (pass.empty()) {
                continue;
            }
            {
                busy_guard guard(*this);
                std::vector<part_job *> jobs(pass.begin(), pass.end());
                part->run_batch(jobs.data(), jobs.size());
            }
            n_forwards += pass.size();
            if (pass.size() > 1) {
                n_shared += pass.size();
            }
            {
                std::lock_guard<std::mutex> lock(q_mtx);
                for (fwd_job * j : pass) {
                    j->done = true;
                }
            }
            q_done.notify_all();
        }
    }

    // runs the forward in the next pass of the worker
    void run_queued(fwd_job & j) {
        std::unique_lock<std::mutex> lock(q_mtx);
        queue.push_back(&j);
        q_cv.notify_one();
        q_done.wait(lock, [&]() { return j.done; });
    }

    // a sequence of the context for a new session, with n_ctx tokens of it
    llama_seq_id take_seq(uint32_t n_ctx) {
        std::lock_guard<std::mutex> lock(q_mtx);
        const uint64_t total = llama_n_ctx(part->ctx.get());
        if (free_seqs.empty()) {
            throw std::runtime_error("too many sessions (" + std::to_string(part->n_seq) + ")");
        }
        if (ctx_used + n_ctx > total) {
            throw std::runtime_error("no room for a context of " + std::to_string(n_ctx) + " tokens: " +
                std::to_string(total - ctx_used) + " of " + std::to_string(total) + " are free");
        }
        const llama_seq_id seq = free_seqs.back();
        free_seqs.pop_back();
        ctx_used += n_ctx;
        return seq;
    }

    void give_back_seq(llama_seq_id seq, uint32_t n_ctx) {
        {
            std::lock_guard<std::mutex> lock(q_mtx);
            ended_seqs.push_back(seq);
            ctx_used -= n_ctx;
        }
        q_cv.notify_one();
    }

    struct busy_guard {
        node_server & srv;
        busy_guard(node_server & srv) : srv(srv) { srv.busy++; }
        ~busy_guard() { srv.last_end_us = (int64_t) (now_ms() * 1000); srv.busy--; }
    };

    // for the GPU warmer: forwards running now and the end of the last one
    std::atomic<int>     busy{ 0 };
    std::atomic<int64_t> last_end_us{ 0 };

    bool waiting_with_session(int interval_ms) {
        if (busy > 0 || (int64_t) (now_ms() * 1000) - last_end_us < (int64_t) interval_ms * 1000) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx);
        return !sessions.empty();
    }

    float corrupt = 0; // --test-corrupt: a dishonest node for the tests of the verifier (noise in the outputs)

    std::vector<uint8_t> key;                 // secret key: open and attach need an access token; empty: open node (tests)
    void *               server_ctx = nullptr; // SSL_CTX of incoming connections, null: plain TCP
    void *               client_ctx = nullptr; // SSL_CTX of the connection to the next node

    // "" when the token gives access to the session
    std::string check_token(uint64_t session, const std::string & token) const {
        if (key.empty()) {
            return "";
        }
#ifdef AGENT_HAVE_TLS
        return token_check(key, session, token);
#else
        (void) session; (void) token;
        return "access tokens need a build with OpenSSL or BoringSSL";
#endif
    }

    std::mutex                                     mtx;
    std::map<uint64_t, std::shared_ptr<session>>   sessions;

    // tokens per session that went through this node, for the coordinator (points of the owner):
    // kept for a while after the session closes, so that the last heartbeat counts them too
    struct work_item {
        int64_t tokens = 0;
        double  last_ms = 0;
    };
    std::mutex                    work_mtx;
    std::map<uint64_t, work_item> work;

    void count_work(uint64_t session, int64_t n_tokens) {
        std::lock_guard<std::mutex> lock(work_mtx);
        work_item & w = work[session];
        w.tokens += n_tokens;
        w.last_ms = now_ms();
    }

    // [{"session": "<16 hex>", "tokens": N}] of the last 15 minutes
    json work_json() {
        std::lock_guard<std::mutex> lock(work_mtx);
        json out = json::array();
        const double t = now_ms();
        for (auto it = work.begin(); it != work.end();) {
            if (t - it->second.last_ms > 15 * 60 * 1000.0) {
                it = work.erase(it);
                continue;
            }
            char hex[17];
            snprintf(hex, sizeof(hex), "%016" PRIx64, it->first);
            out.push_back({ { "session", hex }, { "tokens", it->second.tokens } });
            ++it;
        }
        return out;
    }

    std::shared_ptr<session> find(uint64_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = sessions.find(id);
        if (it == sessions.end()) {
            throw std::runtime_error("unknown session " + std::to_string(id));
        }
        return it->second;
    }

    void on_open(const tcp_conn_ptr & conn, reader & r, std::set<uint64_t> & authed) {
        const open_msg m = open_msg::read(r);
        if (m.version != PROTOCOL_VERSION) {
            throw std::runtime_error("protocol version " + std::to_string(m.version) + ", this node speaks " + std::to_string(PROTOCOL_VERSION));
        }
        if (const std::string err = check_token(m.session, m.token); !err.empty()) {
            throw std::runtime_error("access denied: " + err);
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = sessions.find(m.session);
            // the client reopens a session after a failure: the connection of the old one may not be noticed as closed yet
            if (it != sessions.end() && it->second->owner->is_closed()) {
                sessions.erase(it);
                it = sessions.end();
            }
            if (it != sessions.end()) {
                throw std::runtime_error("session already open");
            }
            if (sessions.size() >= max_sessions) {
                throw std::runtime_error("too many sessions");
            }
        }
        auto s = std::make_shared<session>();
        s->id    = m.session;
        s->n_top = std::clamp<uint32_t>(m.n_top, 1, 256);
        if (m.hidden != payload::f16 && m.hidden != payload::f32 && m.hidden != payload::q8) {
            throw std::runtime_error("hidden states must be f16, f32 or q8");
        }
        if (m.hidden == payload::q8 && llama_model_n_embd(model.get()) % Q8_BLOCK != 0) {
            throw std::runtime_error("q8 hidden states need n_embd divisible by 32");
        }
        s->hidden = m.hidden;
        s->owner = conn;
        if (part->is_last() != m.next.empty()) {
            throw std::runtime_error(part->is_last() ? "this node runs the last layers, it has no next node"
                                                     : "this node does not run the last layers, it needs a next node");
        }
        if (m.n_rollback > part->n_rollback) {
            throw std::runtime_error("this node rolls back at most " + std::to_string(part->n_rollback) + " draft tokens, the session asks for " +
                std::to_string(m.n_rollback) + " (agent-node node --seq-rollback)");
        }
        s->n_ctx   = std::max<uint32_t>(m.n_ctx, 1);
        s->seq     = take_seq(s->n_ctx);
        s->release = [this, seq = s->seq, n = s->n_ctx]() { give_back_seq(seq, n); };
        if (!m.next.empty()) {
            s->next = secure_connect(m.next, client_ctx, m.next_pin);
            attach_msg am;
            am.session = m.session;
            am.token   = m.next_token;
            if (!send_frame(*s->next, msg::attach, am.write())) {
                throw std::runtime_error("cannot attach to the next node " + m.next);
            }
            // errors of the next node go back to the client
            std::thread([next = s->next, owner = conn, addr = m.next]() {
                msg type;
                std::vector<uint8_t> body;
                try {
                    while (recv_frame(*next, type, body)) {
                        if (type == msg::error) {
                            reader rr(body.data(), body.size());
                            send_error(*owner, addr + ": " + rr.get_str());
                        }
                    }
                } catch (const std::exception &) {
                }
            }).detach();
        }

        open_ok_msg ok;
        ok.layer_start = part->first;
        ok.layer_end   = part->last;
        ok.n_layer     = part->n_layer;
        ok.n_embd      = part->n_embd;
        ok.n_vocab     = part->n_vocab;
        ok.desc        = model_desc(model.get());
        {
            std::lock_guard<std::mutex> lock(mtx);
            sessions[m.session] = s;
        }
        authed.insert(m.session);
        AN_LOG("session %" PRIu64 ": layers [%d, %d), n_ctx = %u, sequence %d, next = %s\n",
            m.session, ok.layer_start, ok.layer_end, m.n_ctx, (int) s->seq, m.next.empty() ? "(client)" : m.next.c_str());
        send_frame(*conn, msg::open_ok, ok.write());
    }

    void on_forward(const std::vector<uint8_t> & body) {
        reader r(body.data(), body.size());
        forward_hdr h = forward_hdr::read(r);
        auto s = find(h.session);

        const model_part & p = *part;
        const size_t need = payload_size(h.kind, h.n_tokens, p.n_embd);
        const uint8_t * data = r.get_bytes(need);
        if ((uint64_t) h.pos0 + h.n_tokens > s->n_ctx) {
            throw std::runtime_error("forward ends at position " + std::to_string((uint64_t) h.pos0 + h.n_tokens) +
                ", past the context of the session (" + std::to_string(s->n_ctx) + ")");
        }

        fwd_job j;
        j.h     = h;
        j.data  = data;
        j.seq   = s->seq;
        j.n_top = s->n_top;
        run_queued(j);
        if (!j.err.empty()) {
            throw std::runtime_error(j.err);
        }
        std::vector<float> &     hidden = j.hidden;
        std::vector<top_entry> & top    = j.top;
        const uint32_t           n_out  = j.n_out;

        std::lock_guard<std::mutex> lock(s->mtx);
        s->st.bytes_in += 5 + body.size();
        s->st.n_tokens += h.n_tokens;
        s->st.n_forward++;
        s->n_forward++;
        s->n_shared += j.n_pass > 1;
        count_work(h.session, h.n_tokens);
        s->st.compute_ms += j.pass_ms;
        s->st.output_ms  += j.output_ms;
        if (corrupt > 0) {
            // a node that saves work: its outputs are partly made up
            static thread_local std::mt19937 rng(7);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (auto & v : hidden) {
                v = (1.0f - corrupt) * v + corrupt * nd(rng) * std::max(1e-3f, std::fabs(v) + 0.5f);
            }
            for (auto & e : top) {
                e.logit += corrupt * 10.0f * nd(rng);
            }
            std::sort(top.begin(), top.end(), [](const top_entry & x, const top_entry & y) { return x.logit > y.logit; });
        }

        if (!p.is_last()) {
            forward_hdr out = h;
            out.kind = s->hidden;
            writer w;
            out.write(w);
            if (s->hidden == payload::f32) {
                w.put_bytes(hidden.data(), hidden.size() * sizeof(float));
            } else if (s->hidden == payload::q8) {
                // rows are whole blocks (n_embd % 32 == 0), so the rows can be quantized as one run
                ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(hidden.data(),
                    w.grow(payload_size(payload::q8, out.n_tokens, p.n_embd)), (int64_t) hidden.size());
            } else {
                ggml_fp32_to_fp16_row(hidden.data(), (ggml_fp16_t *) w.grow(hidden.size() * 2), (int64_t) hidden.size());
            }
            s->st.bytes_out += frame_size(w);
            if (!send_frame(*s->next, msg::forward, w)) {
                throw std::runtime_error("the next node closed the connection");
            }
        } else {
            result_msg res;
            res.session = h.session;
            res.step    = h.step;
            res.n_out   = n_out;
            res.n_top   = s->n_top;
            res.top     = top;
            writer w = res.write();
            s->st.bytes_out += frame_size(w);
            if (!send_frame(*s->owner, msg::result, w)) {
                throw std::runtime_error("the client closed the connection");
            }
        }
    }

    void drop_owned_by(const tcp_conn_ptr & conn) {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto it = sessions.begin(); it != sessions.end();) {
            if (it->second->owner == conn) {
                AN_LOG("session %" PRIu64 ": client gone, closing\n", it->first);
                if (it->second->next) {
                    it->second->next->close();
                }
                it = sessions.erase(it);
            } else {
                ++it;
            }
        }
    }

    void serve(tcp_conn_ptr conn) {
#ifdef AGENT_HAVE_TLS
        if (server_ctx) {
            const std::string err = conn->tls_handshake((SSL_CTX *) server_ctx, true);
            if (!err.empty()) {
                AN_LOG("error: %s\n", err.c_str());
                return;
            }
        }
#endif
        // sessions this connection may use: opened by it, or attached to with a valid token
        std::set<uint64_t> authed;
        auto require = [&](uint64_t session) {
            if (!key.empty() && !authed.count(session)) {
                throw std::runtime_error("access denied: session " + std::to_string(session) + " was not opened or attached on this connection");
            }
        };

        msg type;
        std::vector<uint8_t> body;
        try {
            while (recv_frame(*conn, type, body)) {
                try {
                    reader r(body.data(), body.size());
                    switch (type) {
                        case msg::open:    on_open(conn, r, authed); break;
                        case msg::attach: {
                            // the previous node: forward frames follow
                            const attach_msg am = attach_msg::read(r);
                            if (const std::string err = check_token(am.session, am.token); !err.empty()) {
                                throw std::runtime_error("access denied: " + err);
                            }
                            authed.insert(am.session);
                        } break;
                        case msg::forward: {
                            reader rh(body.data(), body.size());
                            require(rh.get<uint64_t>());
                            on_forward(body);
                        } break;
                        case msg::hello: {
                            hello_ok_msg h;
                            h.n_layer     = llama_model_n_layer(model.get());
                            h.layer_start = std::clamp(mo.layer_start, 0, h.n_layer);
                            h.layer_end   = mo.layer_end <= 0 ? h.n_layer : std::min(mo.layer_end, h.n_layer);
                            h.tls         = server_ctx != nullptr;
                            h.auth        = !key.empty();
                            h.desc        = model_desc(model.get());
                            send_frame(*conn, msg::hello_ok, h.write());
                        } break;
                        case msg::stats: {
                            const uint64_t id = r.get<uint64_t>();
                            require(id);
                            auto s = find(id);
                            stats_msg st;
                            {
                                std::lock_guard<std::mutex> lock(s->mtx);
                                st = s->st;
                            }
                            send_frame(*conn, msg::stats_ok, st.write());
                        } break;
                        case msg::close: {
                            const uint64_t id = r.get<uint64_t>();
                            require(id);
                            std::lock_guard<std::mutex> lock(mtx);
                            sessions.erase(id);
                        } break;
                        default:
                            throw std::runtime_error("unexpected frame type " + std::to_string((int) type));
                    }
                } catch (const std::exception & e) {
                    AN_LOG("error: " "%s\n", e.what());
                    send_error(*conn, e.what());
                }
            }
        } catch (const std::exception & e) {
            AN_LOG("error: " "connection: %s\n", e.what());
        }
        conn->close();
        drop_owned_by(conn);
    }
};

// a quick measurement of the speed of this node for the coordinator: single-token forwards through the layers of the part,
// as in generation; returns milliseconds per layer per token
static double measure_ms_per_layer(const llama_model * model, const model_opts & mo, int32_t n_threads) {
    model_part p(model, mo, 64, n_threads);
    std::vector<float> in((size_t) p.n_embd, 0.01f), hidden;
    std::vector<top_entry> top;
    uint32_t n_out = 0;
    const int n_warm = 3, n_run = 16;
    double t0 = 0;
    for (int i = 0; i < n_warm + n_run; i++) {
        if (i == n_warm) {
            t0 = now_ms();
        }
        forward_hdr h;
        h.pos0     = i;
        h.n_tokens = 1;
        h.kind     = p.is_first() ? payload::tokens : payload::f32;
        const int32_t tok = 1;
        p.run(h, p.is_first() ? (const uint8_t *) &tok : (const uint8_t *) in.data(), hidden, top, 8, n_out);
    }
    return (now_ms() - t0) / n_run / std::max(1, p.last - p.first);
}

// the context of a node for up to `want` sessions of n_ctx tokens each: on a GPU fewer, if the memory of the video card
// would not hold them with `margin` bytes to spare (beyond it the driver moves weights to shared memory, bench/RESULTS.md).
// The memory per session (KV cache, DeltaNet states with their rollback snapshots) is measured with contexts for 1 and 2
static std::unique_ptr<model_part> make_node_part(const llama_model * model, const model_opts & mo, int32_t n_threads,
        uint32_t want, uint32_t n_ctx, uint32_t n_rollback, uint64_t margin) {
    ggml_backend_dev_t dev = mo.device.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)
                                               : ggml_backend_dev_by_name(mo.device.c_str());
    uint32_t n = want;
    if (want > 1 && dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        auto free_mem = [dev]() {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            return (int64_t) free;
        };
        const int64_t f0 = free_mem();
        int64_t f1, f2;
        {
            model_part p(model, mo, n_ctx, n_threads, n_rollback, 1, mo.moe_cache_bytes);
            f1 = free_mem();
        }
        {
            model_part p(model, mo, 2 * n_ctx, n_threads, n_rollback, 2, mo.moe_cache_bytes);
            f2 = free_mem();
        }
        const int64_t per  = std::max<int64_t>(f1 - f2, 1);
        const int64_t base = f0 - f1 - per; // compute buffers
        n = (uint32_t) std::clamp<int64_t>((f0 - (int64_t) margin - base) / per, 1, want);
        AN_LOG("node: %.0f MiB free, a session takes %.0f MiB, the context %.0f MiB more: %u of %u sessions fit with %.0f MiB to spare\n",
            f0 / 1048576.0, per / 1048576.0, std::max<int64_t>(base, 0) / 1048576.0, n, want, margin / 1048576.0);
    }
    return std::make_unique<model_part>(model, mo, n * n_ctx, n_threads, n_rollback, n, mo.moe_cache_bytes);
}

//
// a node without a model file: the coordinator assigns the layers, the node downloads only them
//

static std::string fmt_gb(uint64_t n) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f GB", n / 1e9);
    return buf;
}

// prints the progress of a download about every 5 s (the scripts show the lines with "download")
static std::function<void(const slice_progress &)> download_printer() {
    struct st { double t0 = 0, t_last = -1; uint64_t d0 = 0; bool first = true; };
    auto state = std::make_shared<st>();
    state->t0 = now_ms();
    return [state](const slice_progress & p) {
        const double t = now_ms();
        if (state->first) {
            // a resumed download: the speed counts only what comes now (about: up to 8 MB of this call are new)
            state->first = false;
            state->d0    = p.done > (8u << 20) ? p.done - (8u << 20) : 0;
        }
        if (p.done < p.total && state->t_last >= 0 && t - state->t_last < 5000) {
            return;
        }
        state->t_last = t;
        const double mbps = t > state->t0 ? (p.done - state->d0) / 1e6 / ((t - state->t0) / 1000) : 0;
        printf("agent-node: download %s of %s (%.0f%%), %.1f MB/s\n", fmt_gb(p.done).c_str(), fmt_gb(p.total).c_str(),
            p.total ? 100.0 * p.done / p.total : 100.0, mbps);
        fflush(stdout);
    };
}

// the full model by URL; a failing huggingface.co URL is retried through the mirror
static void open_model_source(model_source & ms, const std::string & url, const std::string & mirror) {
    std::string first_err;
    try {
        ms.open(url);
        return;
    } catch (const std::exception & e) {
        const std::string hf = "https://huggingface.co/";
        if (mirror.empty() || url.rfind(hf, 0) != 0) {
            throw;
        }
        first_err = e.what();
        AN_LOG("warning: %s; trying %s\n", e.what(), mirror.c_str());
    }
    std::string m = mirror;
    while (!m.empty() && m.back() == '/') {
        m.pop_back();
    }
    ms = model_source();
    try {
        ms.open(m + url.substr(std::string("https://huggingface.co").size()));
    } catch (const std::exception & e) {
        // both errors: the one of Hugging Face is usually the telling one
        throw std::runtime_error(first_err + "; the mirror: " + e.what());
    }
}

struct slice_file {
    std::string path;
    int32_t     first = 0, last = 0;
    uint64_t    bytes = 0;
};

// the slices of this model already in the folder
static std::vector<slice_file> find_slices(const std::string & dir, const std::string & key) {
    std::vector<slice_file> out;
    std::error_code ec;
    for (const auto & e : std::filesystem::directory_iterator(slice_path(dir), ec)) {
        if (!e.is_regular_file() || e.path().extension() != ".gguf") {
            continue;
        }
#if defined(__cpp_char8_t)
        const auto u8 = e.path().u8string();
        const std::string p(u8.begin(), u8.end());
#else
        const std::string p = e.path().u8string();
#endif
        gguf_ptr g = read_local_header(p);
        if (!g || gguf_str(g.get(), "agent.model_key") != key) {
            continue;
        }
        slice_file f;
        f.path  = p;
        f.first = (int32_t) gguf_int(g.get(), "agent.layer_start");
        f.last  = (int32_t) gguf_int(g.get(), "agent.layer_end");
        f.bytes = (uint64_t) e.file_size();
        if (f.first >= 0 && f.last > f.first) {
            out.push_back(f);
        }
    }
    return out;
}

static std::string slice_name(const std::string & url, int32_t first, int32_t last) {
    std::string base = url.substr(url.find_last_of("/\\") + 1);
    const size_t q = base.find('?');
    if (q != std::string::npos) {
        base = base.substr(0, q);
    }
    if (base.size() > 5 && base.compare(base.size() - 5, 5, ".gguf") == 0) {
        base = base.substr(0, base.size() - 5);
    }
    if (base.size() > 15 && base.compare(base.size() - 15, 4, "-000") == 0 && base.find("-of-", base.size() - 10) != std::string::npos) {
        base = base.substr(0, base.size() - 15); // "-00001-of-00004"
    }
    return base + ".L" + std::to_string(first) + "-" + std::to_string(last) + ".gguf";
}

// asks the coordinator which layers to hold (by the free video memory and the disk space this node gives),
// downloads that slice unless it is already here, and points mo at it
// asks the coordinator for layers and downloads them if they are not here; returns the bytes of the slice in video memory.
// vram_limit: no more video memory than this for the model (the last slice did not fit), 0: what is free
static uint64_t prepare_assigned_slice(const cli_args & a, model_opts & mo, const std::string & fingerprint, uint64_t vram_limit = 0) {
    const std::string url    = a.get("--model-url", "", "");
    const std::string coord  = a.get("--coordinator", "", "");
    const std::string dir    = a.get("--models-dir", "", "models");
    if (coord.empty()) {
        throw std::runtime_error("--model-url needs --coordinator: the coordinator assigns the layers");
    }
    std::filesystem::create_directories(slice_path(dir));

    model_source ms;
    open_model_source(ms, url, a.get("--mirror", "", ""));
    AN_LOG("node: model %s, %d layers, %s in all\n", ms.key.c_str(), ms.n_layer, fmt_gb(ms.slice_bytes(0, ms.n_layer)).c_str());

    // video memory: the free memory of the device minus a reserve for the context and the compute buffers
    uint64_t vram = 0;
    if (a.has("--vram-gb")) {
        vram = (uint64_t) (std::stod(a.get("--vram-gb", "", "0")) * 1e9);
    } else {
        ggml_backend_dev_t dev = mo.device.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)
                                                   : ggml_backend_dev_by_name(mo.device.c_str());
        if (dev == nullptr) {
            throw std::runtime_error("no GPU found; give the memory for the model with --vram-gb");
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        const uint64_t reserve = (uint64_t) a.get_int("--vram-reserve-mb", "", 1024) << 20;
        vram = free > reserve ? free - reserve : 0;
        AN_LOG("node: %s: %s free of %s, %s for the model\n", ggml_backend_dev_description(dev), fmt_gb(free).c_str(),
            fmt_gb(total).c_str(), fmt_gb(vram).c_str());
    }
    if (vram_limit > 0 && vram_limit < vram) {
        vram = vram_limit;
        AN_LOG("node: at most %s for the model: the last layers did not fit in the video memory\n", fmt_gb(vram).c_str());
    }

    // RAM for the experts of an MoE model (--cpu-moe): the free memory minus 4 GB for the system, or --ram-gb
    uint64_t ram = 0;
    if (mo.cpu_moe) {
        if (a.has("--ram-gb")) {
            ram = (uint64_t) (std::stod(a.get("--ram-gb", "", "0")) * 1e9);
        } else if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(cpu, &free, &total);
            const uint64_t reserve = 4ull << 30;
            ram = free > reserve ? free - reserve : 0;
            AN_LOG("node: RAM %s free of %s, %s for the experts\n", fmt_gb(free).c_str(), fmt_gb(total).c_str(), fmt_gb(ram).c_str());
        }
    }

    // disk: what the participant gives, but no more than there is (slices of this model already here count as free)
    std::vector<slice_file> have = find_slices(dir, ms.key);
    uint64_t have_bytes = 0;
    for (const auto & h : have) {
        have_bytes += h.bytes;
    }
    std::error_code ec;
    const auto sp = std::filesystem::space(slice_path(dir), ec);
    const uint64_t avail  = ec ? 0 : (uint64_t) sp.available;
    const uint64_t margin = 1ull << 30;
    uint64_t disk = (uint64_t) (std::stod(a.get("--disk-gb", "", "25")) * 1e9);
    disk = std::min(disk, avail + have_bytes > margin ? avail + have_bytes - margin : 0);
    AN_LOG("node: disk for the model %s (%s free in %s)\n", fmt_gb(disk).c_str(), fmt_gb(avail).c_str(), dir.c_str());

    json have_j = json::array();
    for (const auto & h : have) {
        have_j.push_back({ { "layer_start", h.first }, { "layer_end", h.last } });
    }
    json req = {
        { "model", ms.key },
        { "n_layer", ms.n_layer },
        { "layer_bytes", ms.layer_bytes },
        { "embd_bytes", ms.embd_bytes },
        { "output_bytes", ms.output_bytes },
        { "other_bytes", ms.other_bytes },
        { "vram_bytes", vram },
        { "cpu_moe", mo.cpu_moe },
        { "expert_bytes", ms.expert_bytes },
        { "ram_bytes", ram },
        { "disk_bytes", disk },
        { "have", have_j },
        { "tls_fingerprint", fingerprint },
        { "city", a.get("--city", "", "") },
    };
    int32_t first = -1, last = -1;
    for (int attempt = 0; first < 0; attempt++) {
        try {
            const http_reply r = coord_post(coord, "/v1/assign", req);
            if (r.status != 200) {
                throw std::runtime_error("the coordinator assigns no layers: " + r.error());
            }
            first = r.body.value("layer_start", -1);
            last  = r.body.value("layer_end", -1);
            if (first < 0 || last <= first || last > ms.n_layer) {
                throw std::runtime_error("the coordinator assigned a bad range");
            }
        } catch (const std::exception & e) {
            if (attempt == 0 || attempt % 10 == 0) {
                AN_LOG("warning: %s (retrying)\n", e.what());
            }
            std::this_thread::sleep_for(std::chrono::seconds(std::min(30, 2 + attempt * 2)));
        }
    }
    printf("agent-node: the coordinator assigned layers %d:%d (%s)\n", first, last, fmt_gb(ms.slice_bytes(first, last)).c_str());
    fflush(stdout);

    std::string path;
    for (const auto & h : have) {
        if (h.first == first && h.last == last) {
            path = h.path;
        }
    }
    // the other slices of this model are not needed any more: they free the disk for the new one
    for (const auto & h : have) {
        if (h.path != path) {
            AN_LOG("node: removing the old slice %s (layers %d:%d)\n", h.path.c_str(), h.first, h.last);
            std::filesystem::remove(slice_path(h.path), ec);
        }
    }
    if (path.empty()) {
        path = (std::filesystem::path(slice_path(dir)) / slice_path(slice_name(url, first, last))).u8string().c_str();
        write_slice(ms, first, last, path, download_printer());
    }
    mo.path        = path;
    mo.layer_start = first;
    mo.layer_end   = last;

    uint64_t bytes = ms.slice_bytes(first, last);
    if (mo.cpu_moe && (int32_t) ms.expert_bytes.size() == ms.n_layer) {
        for (int32_t l = first; l < last; l++) {
            bytes -= ms.expert_bytes[l];
        }
        // the video memory left after the rest of the slice takes the experts of the last layers, with the same
        // headroom the coordinator keeps (10%, at least 512 MB: compute buffers grow, other programs share the GPU)
        // by default the room goes to the expert cache (--moe-cache-mib to set it), --gpu-experts N: the experts of N layers
        mo.gpu_expert_layers = 0;
        mo.moe_cache_bytes   = 0;
        const uint64_t headroom = std::max<uint64_t>(vram / 10, 512ull << 20);
        const uint64_t room = vram > bytes + headroom ? vram - bytes - headroom : 0;
        uint64_t gpu_experts = 0;
        if (a.has("--gpu-experts")) {
            mo.gpu_expert_layers = std::clamp(a.get_int("--gpu-experts", "", 0), 0, last - first);
            for (int32_t l = last - mo.gpu_expert_layers; l < last; l++) {
                gpu_experts += ms.expert_bytes[l];
            }
            AN_LOG("node: the experts of %d of %d layers on the GPU (%s), the rest in RAM\n", mo.gpu_expert_layers, last - first,
                fmt_gb(gpu_experts).c_str());
        } else {
            uint64_t all = 0;
            for (int32_t l = first; l < last; l++) {
                all += ms.expert_bytes[l];
            }
            // more than all experts of the slice is of no use; less than one layer of them neither
            mo.moe_cache_bytes = a.has("--moe-cache-mib") ? (uint64_t) a.get_int("--moe-cache-mib", "", 0) << 20 : std::min(room, all);
            if (mo.moe_cache_bytes < (64ull << 20)) {
                mo.moe_cache_bytes = 0;
            }
            gpu_experts = mo.moe_cache_bytes;
            AN_LOG("node: the experts in RAM (%s), a cache of %s of them on the GPU\n", fmt_gb(all).c_str(), fmt_gb(gpu_experts).c_str());
        }
        bytes += gpu_experts;
    }
    return bytes;
}

static std::string model_key(const std::string & path);

// one node of the chain as the client knows it
struct node_ref {
    std::string addr;
    std::string key;   // secret key of the node, to issue tokens itself (tests; normally the coordinator issues them)
    std::string token; // access token issued by the coordinator for the session
    std::string pin;   // expected certificate fingerprint ("" = not checked)
};

// the identity of this PC for the coordinator: the welcome points go only to the first account of a PC.
// sha256 of the machine id (Windows MachineGuid, Linux /etc/machine-id); "" when unknown
static std::string device_hash() {
    std::string id;
#ifdef _WIN32
    char buf[128];
    DWORD sz = sizeof(buf);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", "MachineGuid", RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
                     nullptr, buf, &sz) == ERROR_SUCCESS) {
        id = buf;
    }
#else
    std::ifstream f("/etc/machine-id");
    std::getline(f, id);
#endif
    if (id.empty()) {
        return "";
    }
#ifdef AGENT_HAVE_TLS
    const std::string m = "agent-device|" + id;
    uint8_t md[32];
    SHA256((const uint8_t *) m.data(), m.size(), md);
    static const char * hex = "0123456789abcdef";
    std::string out;
    for (uint8_t b : md) {
        out += hex[b >> 4];
        out += hex[b & 15];
    }
    return out;
#else
    return "";
#endif
}

//
// the account of this PC at the coordinator: the owner of a node gets points, a client spends them
//

struct coord_account {
    std::string url;
    std::string file;  // the token lives here (created on the first run)
    std::string token;

    // reads the token, or makes an account and saves its token; renew: the coordinator forgot the old one
    // (restarted with the in-memory store). false: the coordinator has no accounts
    bool load_or_create(bool renew = false) {
        if (!renew && token.empty() && !file.empty()) {
            std::ifstream f(u8_path(file));
            std::getline(f, token);
            while (!token.empty() && (token.back() == '\r' || token.back() == '\n' || token.back() == ' ')) {
                token.pop_back();
            }
        }
        if (!token.empty() && !renew) {
            return true;
        }
        json body = json::object();
        if (const std::string d = device_hash(); !d.empty()) {
            body["device"] = d;
        }
        const http_reply r = coord_post(url, "/v1/accounts", body);
        if (r.status == 404) {
            token.clear();
            return false;
        }
        if (r.status != 200 || r.body.value("token", "").empty()) {
            throw std::runtime_error("cannot create an account at the coordinator: " + r.error());
        }
        token = r.body.value("token", "");
        if (!file.empty()) {
            std::ofstream f(u8_path(file), std::ios::trunc);
            f << token << "\n";
        }
        printf("agent-node: new account %s at the coordinator, %lld points, the token is in %s\n",
            r.body.value("account_id", "").c_str(), (long long) r.body.value("balance", 0), file.empty() ? "(memory)" : file.c_str());
        fflush(stdout);
        return true;
    }

    httplib::Headers headers() const {
        if (token.empty()) {
            return {};
        }
        return { { "Authorization", "Bearer " + token } };
    }

    // POST with the account; once more with a new account if the coordinator does not know this one
    http_reply post(const std::string & path, const json & body) {
        http_reply r = coord_post(url, path, body, headers());
        if (r.status == 401 && r.error().find("account") != std::string::npos && load_or_create(true)) {
            r = coord_post(url, path, body, headers());
        }
        return r;
    }
};

static coord_account account_from(const cli_args & a) {
    coord_account acc;
    acc.url  = a.get("--coordinator", "", "");
    acc.file = a.get("--account", "", "agent-account.txt");
    if (a.get("--account", "", "") != "off") {
        acc.load_or_create();
    }
    return acc;
}

// a chain from the coordinator for one session: nodes and spares with their pins and tokens
struct coord_chain {
    std::vector<node_ref> nodes, spares;
    uint64_t              session = 0;
    std::string           session_hex;
    int                   no_spare = 0;
    double                ms = 0;
};

static coord_chain request_chain(coord_account & acc, const std::string & model_key, int spares, int ttl_s, const std::string & city) {
    json req = { { "model", model_key }, { "spares", spares }, { "ttl_s", ttl_s } };
    if (!city.empty()) {
        req["city"] = city;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const http_reply r = acc.post("/v1/chains", req);
    if (r.status == 402) {
        throw std::runtime_error("not enough points for a request: " + r.error());
    }
    if (r.status != 200) {
        throw std::runtime_error("the coordinator gave no chain: " + r.error());
    }
    auto refs_of = [](const json & arr) {
        std::vector<node_ref> out;
        for (const auto & n : arr) {
            out.push_back({ n.at("addr").get<std::string>(), "", n.at("token").get<std::string>(), n.at("pin").get<std::string>() });
        }
        return out;
    };
    coord_chain c;
    c.nodes       = refs_of(r.body.at("nodes"));
    c.spares      = refs_of(r.body.at("spares"));
    c.session_hex = r.body.at("session").get<std::string>();
    c.session     = std::stoull(c.session_hex, nullptr, 16);
    c.no_spare    = r.body.value("stages_without_spare", 0);
    c.ms          = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return c;
}

// after the answer: the client tells the coordinator how many tokens there were and pays for them
// returns the balance after the payment, -1 when the coordinator has no accounts
static long long report_session(coord_account & acc, const std::string & session_hex, int32_t n_prompt, int32_t n_gen, long long * charged = nullptr) {
    if (acc.token.empty()) {
        return -1;
    }
    try {
        const http_reply r = acc.post("/v1/sessions/" + session_hex + "/report", { { "prompt_tokens", n_prompt }, { "gen_tokens", n_gen } });
        if (r.status != 200) {
            AN_LOG("warning: the report of session %s was not accepted: %s\n", session_hex.c_str(), r.error().c_str());
            return -1;
        }
        if (charged) {
            *charged = r.body.value("charged", 0LL);
        }
        return r.body.value("balance", 0LL);
    } catch (const std::exception & e) {
        AN_LOG("warning: report of session %s: %s\n", session_hex.c_str(), e.what());
        return -1;
    }
}

// a node behind NAT: it keeps a connection to the relay of the network, and the relay hands it the connections
// of the others (relay://relay/<fingerprint>); TLS between the nodes goes through the relay unchanged
static void relay_listen(node_server & srv, const std::string & coord, const std::string & fingerprint, json ticket) {
    int backoff = 1;
    while (true) {
        try {
            if (ticket.value("expires", 0LL) - (long long) std::time(nullptr) < 3600) {
                const http_reply r = coord_post(coord, "/v1/relay/ticket", { { "tls_fingerprint", fingerprint } });
                if (r.status != 200) {
                    throw std::runtime_error("relay ticket: " + r.error());
                }
                ticket = r.body;
            }
            const std::string relay = ticket.value("relay", "");
            if (p2p_dial) {
                // the same ticket registers the node at the rendezvous server (UDP, the same port)
                p2p_endpoint::get().listen(relay, ticket.value("id", ""), ticket.value("expires", 0LL), ticket.value("ticket", ""),
                                           [&srv](tcp_conn_ptr c) { srv.serve(c); });
            }
            tcp_conn_ptr ctl = tcp_connect_direct(relay, 10000);
            const std::string req = "LISTEN " + ticket.value("id", "") + " " + std::to_string(ticket.value("expires", 0LL)) + " " +
                                    ticket.value("ticket", "") + "\n";
            if (!ctl->send_all(req.data(), req.size())) {
                throw std::runtime_error("the relay closed the connection");
            }
            const std::string ok = relay_read_line(*ctl);
            if (ok != "OK") {
                throw std::runtime_error("relay: " + ok);
            }
            printf("agent-node: listening on the relay %s\n", relay.c_str());
            fflush(stdout);
            backoff = 1;
            ctl->recv_timeout_ms = 70000; // the relay pings every 20 s
            while (true) {
                const std::string line = relay_read_line(*ctl);
                if (line.rfind("INCOMING ", 0) != 0) {
                    continue; // PING
                }
                const std::string cid = line.substr(9);
                std::thread([&srv, relay, cid]() {
                    try {
                        tcp_conn_ptr c = tcp_connect_direct(relay, 10000);
                        const std::string acc = "ACCEPT " + cid + "\n";
                        if (!c->send_all(acc.data(), acc.size()) || relay_read_line(*c) != "OK") {
                            return;
                        }
                        srv.serve(c); // TLS handshake and frames as for a direct connection
                    } catch (const std::exception & e) {
                        AN_LOG("warning: relay connection: %s\n", e.what());
                    }
                }).detach();
            }
        } catch (const std::exception & e) {
            AN_LOG("warning: relay: %s (retrying in %d s)\n", e.what(), backoff);
        }
        std::this_thread::sleep_for(std::chrono::seconds(backoff));
        backoff = std::min(backoff * 2, 30);
    }
}

static int cmd_node(const cli_args & a) {
    node_server srv;
    const bool assigned = a.has("--model-url");
    if (assigned && (a.has("-m") || a.has("--model") || a.has("--layers"))) {
        throw std::runtime_error("--model-url takes the place of -m and --layers: the coordinator assigns the layers");
    }
    if (assigned) {
        srv.mo.n_gpu_layers = a.get_int("-ngl", "--n-gpu-layers", 999);
        srv.mo.device       = a.get("--device", "-dev", "");
        srv.mo.cpu_moe      = a.has("--cpu-moe");
    } else {
        srv.mo = model_opts_from(a);
    }
    srv.n_threads = a.get_int("-t", "--threads", common_cpu_get_num_math());
    srv.corrupt   = std::stof(a.get("--test-corrupt", "", "0"));

    const std::string host = a.get("--host", "", "127.0.0.1");
    const int         port = a.get_int("--port", "", 50152);

    // access: the secret key of this node (the coordinator issues tokens with it)
    std::string key_hex = a.get("--key", "", "");
    if (key_hex.empty()) {
        if (const char * env = getenv("AGENT_NODE_KEY")) {
            key_hex = env;
        }
    }
    if (!key_hex.empty()) {
        require_tls("--key");
        srv.key = from_hex(key_hex);
        if (srv.key.size() < 16) {
            throw std::runtime_error("--key must have at least 16 bytes (32 hex digits)");
        }
    } else {
        AN_LOG("warning: no --key, the node accepts sessions without access tokens\n");
    }
    std::string fingerprint;
#ifdef AGENT_HAVE_TLS
    tls_identity identity;
    if (a.has("--tls")) {
        identity.load_or_create(a.get("--identity", "", ""));
        fingerprint      = identity.fingerprint();
        srv.server_ctx   = tls_server_ctx(identity);
        srv.client_ctx   = tls_client_ctx();
    }
#else
    if (a.has("--tls")) {
        require_tls("--tls");
    }
#endif
    if (!a.has("--tls")) {
        AN_LOG("warning: no --tls, the traffic of the node is not encrypted\n");
    }

    if (assigned) {
        if (fingerprint.empty() || srv.key.empty()) {
            throw std::runtime_error("--model-url needs --tls and --key: the coordinator only hands out protected nodes");
        }
    }
    uint64_t slice_vram = assigned ? prepare_assigned_slice(a, srv.mo, fingerprint) : 0;

    ggml_backend_dev_t gpu = srv.mo.device.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)
                                                   : ggml_backend_dev_by_name(srv.mo.device.c_str());
    const uint64_t margin = (uint64_t) a.get_int("--vram-margin-mb", "", 256) << 20;
    double ms_per_layer = 0;
    for (int attempt = 0;; attempt++) {
        srv.model = load_model(srv.mo);
        {
            const int32_t n_layer = llama_model_n_layer(srv.model.get());
            const int32_t first   = std::clamp(srv.mo.layer_start, 0, n_layer);
            const int32_t last    = srv.mo.layer_end <= 0 ? n_layer : std::min(srv.mo.layer_end, n_layer);
            AN_LOG("node: %s, layers [%d, %d) of %d, listening on %s:%d\n", model_desc(srv.model.get()).c_str(),
                first, last, n_layer, host.c_str(), port);
        }

        // for the coordinator, which prefers fast nodes; before the ready line, so that a measurement does not meet clients
        if (a.has("--coordinator")) {
            ms_per_layer = measure_ms_per_layer(srv.model.get(), srv.mo, srv.n_threads);
            AN_LOG("node: %.3f ms per layer per token\n", ms_per_layer);
        }

        // one context for the sessions: --max-sessions of --seq-ctx tokens, rolling back up to --seq-rollback draft tokens
        srv.part = make_node_part(srv.model.get(), srv.mo, srv.n_threads, (uint32_t) std::clamp(a.get_int("--max-sessions", "", 4), 1, 64),
            (uint32_t) a.get_int("--seq-ctx", "", 16384), (uint32_t) std::clamp(a.get_int("--seq-rollback", "", 6), 0, 32), margin);

        // the weights or the context went past the video memory into shared memory: the part would run several times slower.
        // Layers from the coordinator: ask for fewer (the video memory of the slice minus what did not fit and the margin)
        const vram_state v = vram_probe(gpu);
        if (v.over() == 0) {
            break;
        }
        AN_LOG("warning: the video memory is full: %s of the node went to shared memory, the node runs several times slower; "
            "close programs that use the video card\n", fmt_gb((uint64_t) v.over()).c_str());
        const uint64_t cut = (uint64_t) v.over() + margin;
        if (!assigned || a.has("--vram-gb") || attempt >= 2 || cut >= slice_vram) {
            break;
        }
        AN_LOG("node: asking the coordinator for fewer layers: %s of video memory for them instead of %s\n",
            fmt_gb(slice_vram - cut).c_str(), fmt_gb(slice_vram).c_str());
        srv.part.reset();
        srv.model.reset();
        slice_vram = prepare_assigned_slice(a, srv.mo, fingerprint, slice_vram - cut);
    }
    srv.max_sessions = srv.part->n_seq;
    srv.batch_max    = (uint32_t) std::clamp(a.get_int("--batch-max", "", (int) srv.max_sessions), 1, 64);
    srv.start_worker();
    AN_LOG("node: %u sessions of up to %u tokens, up to %u forwards in one pass\n", srv.max_sessions,
        llama_n_ctx(srv.part->ctx.get()) / srv.max_sessions, srv.batch_max);

    // hold the GPU clocks while there are sessions: between tokens the GPU would drop to a power-saving state
    static gpu_clocks clocks; // static: the console handler resets the clocks when the window is closed
    std::thread clocks_thread;
    if (a.get("--gpu-clocks", "", "auto") != "off") {
        ggml_backend_dev_t dev = srv.mo.device.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)
                                                       : ggml_backend_dev_by_name(srv.mo.device.c_str());
        ggml_backend_dev_props props = {};
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ggml_backend_dev_get_props(dev, &props);
        }
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && std::string(ggml_backend_dev_name(dev)).rfind("CUDA", 0) == 0 &&
            clocks.open(props.device_id)) {
            // a lock left by a node that was killed
            clocks.locked = true;
            clocks.unlock();
#ifdef _WIN32
            SetConsoleCtrlHandler([](DWORD) -> BOOL { clocks.unlock(); return FALSE; }, TRUE);
#else
            for (int sig : { SIGINT, SIGTERM }) {
                std::signal(sig, [](int) { clocks.unlock(); _exit(1); });
            }
#endif
            if (clocks.lock()) {
                clocks.unlock();
                AN_LOG("node: GPU clocks are held at %u-%u MHz while there are sessions\n", clocks.gpu_max * 4 / 5, clocks.gpu_max);
                clocks_thread = std::thread([&srv]() {
                    double idle_since = -1;
                    while (true) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        bool busy;
                        {
                            std::lock_guard<std::mutex> lock(srv.mtx);
                            busy = !srv.sessions.empty();
                        }
                        if (busy) {
                            idle_since = -1;
                            clocks.lock();
                        } else if (clocks.locked) {
                            if (idle_since < 0) {
                                idle_since = now_ms();
                            } else if (now_ms() - idle_since > 10000) {
                                clocks.unlock();
                            }
                        }
                    }
                });
                clocks_thread.detach();
            } else {
                AN_LOG("warning: cannot hold the GPU clocks: %s; on a network the GPU will be slower between tokens\n", clocks.err.c_str());
            }
        } else if (!clocks.err.empty()) {
            AN_LOG("warning: GPU clocks: %s\n", clocks.err.c_str());
        }
    }

    // keep the GPU awake while a session waits for the next token (--keep-warm MS; off by default: holding the clocks works
    // better, the warmer only helps partly and slows real work a little, bench/RESULTS.md)
    gpu_warmer warmer;
    const int warm_ms = a.get_int("--keep-warm", "", 0);
    if (warm_ms > 0 && warmer.init(srv.mo.device, (size_t) a.get_int("--keep-warm-mb", "", 24))) {
        warmer.run();
        const double job_ms = warmer.run();
        AN_LOG("node: keeping the GPU awake between tokens: a job of %.2f ms every %d ms while a session waits\n", job_ms, warm_ms);
        warmer.start(warm_ms, [&srv, warm_ms]() { return srv.waiting_with_session(warm_ms); });
    }

    // the state of the node for the app (this line every 30 s) and for the coordinator (in the heartbeat)
    auto status = [&srv, gpu]() {
        const vram_state v = vram_probe(gpu);
        int sessions;
        {
            std::lock_guard<std::mutex> lock(srv.mtx);
            sessions = (int) srv.sessions.size();
        }
        json j = {
            { "sessions", sessions },
            { "max_sessions", srv.max_sessions },
            { "batch_max", srv.batch_max },
            { "forwards", srv.n_forwards.load() },
            { "shared", srv.n_shared.load() },
            { "gpu_expert_layers", srv.mo.cpu_moe ? srv.mo.gpu_expert_layers : -1 },
            { "moe_cache_mb", (long long) (srv.mo.moe_cache_bytes >> 20) },
            { "vram_full", v.over() > 0 },
        };
        if (v.free >= 0) {
            j["vram_free_mb"]  = v.free >> 20;
            j["vram_total_mb"] = v.total >> 20;
        }
        if (v.shared >= 0) {
            j["vram_shared_mb"] = v.shared >> 20;
        }
        return j;
    };
    std::thread([status]() {
        bool full = false;
        while (true) {
            const json j = status();
            printf("agent-node: status %s\n", j.dump().c_str());
            fflush(stdout);
            // the memory ran out while working (another program took it): once, until it is back
            if (j.value("vram_full", false) != full) {
                full = !full;
                if (full) {
                    AN_LOG("warning: the video memory is full: %lld MB of the node went to shared memory, the node runs several times "
                        "slower; close programs that use the video card\n", (long long) j.value("vram_shared_mb", 0));
                } else {
                    AN_LOG("node: the layers are back in video memory\n");
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
    }).detach();

    tcp_listener ln(host, port);
    // the scripts wait for this line
    printf("agent-node ready on %s:%d%s%s\n", host.c_str(), port, fingerprint.empty() ? "" : " tls ", fingerprint.c_str());
    fflush(stdout);

    // the coordinator hands this node out in chains; it needs the address, the pin and the key to issue tokens
    coord_link link;
    std::thread link_thread;
    if (a.has("--coordinator")) {
        if (srv.key.empty() || fingerprint.empty()) {
            throw std::runtime_error("--coordinator needs --tls and --key: the coordinator only hands out protected nodes");
        }
        const int32_t n_layer = llama_model_n_layer(srv.model.get());
        // an empty host: the coordinator takes the address the registration came from
        std::string pub = a.get("--public-addr", "", (host == "0.0.0.0" || host == "::" ? "" : host) + ":" + std::to_string(port));
        // behind NAT: the relay of the network (--relay auto: when no --public-addr is given; only: always)
        const std::string relay_mode = a.get("--relay", "", "off");
        if (relay_mode != "off") {
            const http_reply r = coord_post(a.get("--coordinator", "", ""), "/v1/relay/ticket", { { "tls_fingerprint", fingerprint } });
            if (r.status == 200) {
                if (relay_mode == "only" || !a.has("--public-addr")) {
                    pub = r.body.value("addr", pub);
                }
                std::thread(relay_listen, std::ref(srv), a.get("--coordinator", "", ""), fingerprint, r.body).detach();
                // the relay link must be up before the coordinator checks the node through it
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            } else {
                AN_LOG("warning: no relay in this network (%s): the node must be reachable at %s\n", r.error().c_str(), pub.c_str());
            }
        }
        link.url  = a.get("--coordinator", "", "");
        link.key  = srv.key;
        link.reg  = json{
            { "ms_per_layer", ms_per_layer },
            { "addr", pub },
            { "tls_fingerprint", fingerprint },
            { "node_key", key_hex },
            { "model", model_key(srv.mo.path) },
            { "desc", model_desc(srv.model.get()) },
            { "n_layer", n_layer },
            { "layer_start", std::clamp(srv.mo.layer_start, 0, n_layer) },
            { "layer_end", srv.mo.layer_end <= 0 ? n_layer : std::min(srv.mo.layer_end, n_layer) },
            { "max_sessions", srv.max_sessions },
            { "city", a.get("--city", "", "") },
        };
        if (a.has("--disk-gb")) {
            link.reg["disk_gb"] = std::stod(a.get("--disk-gb", "", "0"));
        }
        if (const std::string d = device_hash(); !d.empty()) {
            link.reg["device"] = d; // a PC caught returning wrong results cannot come back with a new account
        }
        // the owner of this node gets the points for its work
        static coord_account acc;
        acc = account_from(a);
        if (!acc.token.empty()) {
            link.reg["account_token"] = acc.token;
        }
        link.renew_account = [&link]() {
            if (!acc.load_or_create(true)) {
                return false;
            }
            link.reg["account_token"] = acc.token;
            return true;
        };
        link.work   = [&srv]() { return srv.work_json(); };
        link.status = status;
        link.ended    = []() { return g_sessions_ended.load(); };
        link.sessions = [&srv]() {
            std::lock_guard<std::mutex> lock(srv.mtx);
            return (int) srv.sessions.size();
        };
        link_thread = std::thread([&link]() { link.run(); });
        link_thread.detach();
    }

    while (true) {
        tcp_conn_ptr c = ln.accept();
        if (!c) {
            continue;
        }
        std::thread([&srv, c]() { srv.serve(c); }).detach();
    }
}

//
// generation, the same loop for the chain and for the local reference
//

struct evaluator {
    virtual ~evaluator() = default;
    // runs n tokens from position pos0, returns the top-k of the last token, or of every token with out_all;
    // checkpoint: keep the state after these tokens, a later eval may start there again (only the chain needs it)
    virtual std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all = false,
            bool checkpoint = false) = 0;
};

struct chain_opts {
    std::vector<node_ref> nodes;
    std::vector<node_ref> spares;   // nodes that can replace a failed node of the same layer range
    bool     tls            = false;
    int      timeout_ms     = 3000;  // a node that does not answer in time counts as failed (plus 20 ms per token of a batch)
    uint32_t n_ctx          = 4096;
    uint32_t n_top          = 8;
    payload  hidden         = payload::q8; // int8: the quality of fp16 at half the traffic (T5)
    uint32_t n_rollback     = 0;
    uint64_t session        = 0;    // 0: chosen by the client
    uint64_t token_ttl      = 3600;
    int      max_recoveries = 5;
};

// a node failed: closed connection, error or timeout; the chain can be rebuilt
struct chain_failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// frames from all nodes, read by one thread per connection
struct chain_events {
    struct event {
        uint64_t             gen;
        size_t               node;
        msg                  type;
        std::vector<uint8_t> body;
        bool                 closed;
    };
    std::mutex              m;
    std::condition_variable cv;
    std::deque<event>       q;

    void push(event && e) {
        {
            std::lock_guard<std::mutex> lock(m);
            q.push_back(std::move(e));
        }
        cv.notify_all();
    }
};

struct chain_eval : evaluator {
    chain_opts                    o;
    std::vector<node_ref>         cur;   // the nodes of the chain now (failed ones replaced by spares)
    std::vector<std::string>      addrs;
    std::vector<tcp_conn_ptr>     conns;
    std::vector<open_ok_msg>      parts;
    uint64_t                      sid  = 0;
    uint32_t                      step = 0;
    uint32_t                      n_top = 8;
    std::vector<llama_token>      hist;  // the tokens in the memory of the chain, by position
    std::vector<int32_t>          ckpts; // positions the nodes keep a state at (FORWARD_CHECKPOINT), the oldest first
    std::shared_ptr<chain_events> ev = std::make_shared<chain_events>();
    uint64_t                      gen = 0; // connections of older generations are ignored
    void *                        cctx = nullptr;

    int    n_recoveries = 0;
    double recover_ms   = 0;     // total pause of the generation for recoveries
    double recover_max_ms = 0;

    explicit chain_eval(const chain_opts & opts) : o(opts), cur(opts.nodes), n_top(opts.n_top) {
#ifdef AGENT_HAVE_TLS
        if (o.tls) {
            cctx = tls_client_ctx();
        }
#else
        if (o.tls) {
            require_tls("--tls");
        }
#endif
        sid = o.session ? o.session : random_session();
        open_chain();
    }

    ~chain_eval() override {
        close_chain(true);
#ifdef AGENT_HAVE_TLS
        if (cctx) {
            SSL_CTX_free((SSL_CTX *) cctx);
        }
#endif
    }

    static uint64_t random_session() {
        std::random_device rd;
        return ((uint64_t) rd() << 32) | rd();
    }

    std::string token_for(const node_ref & n) const {
        if (!n.token.empty()) {
            return n.token;
        }
        if (!n.key.empty()) {
#ifdef AGENT_HAVE_TLS
            return token_issue(n.key, sid, o.token_ttl);
#else
            require_tls("--keys");
#endif
        }
        return "";
    }

    // tokens issued by the coordinator are bound to the session: then a recovery keeps the session id
    bool can_issue_tokens() const {
        for (const auto & v : { o.nodes, o.spares }) {
            for (const auto & n : v) {
                if (!n.token.empty()) {
                    return false;
                }
            }
        }
        return true;
    }

    void start_reader(size_t i, const tcp_conn_ptr & c) {
        std::thread([events = ev, c, i, g = gen]() {
            msg type;
            std::vector<uint8_t> body;
            try {
                while (recv_frame(*c, type, body)) {
                    events->push({ g, i, type, body, false });
                }
            } catch (const std::exception &) {
            }
            events->push({ g, i, msg::error, {}, true });
        }).detach();
    }

    void open_chain() {
        {
            std::lock_guard<std::mutex> lock(ev->m);
            ev->q.clear();
        }
        gen++;
        conns.clear();
        addrs.clear();
        for (const auto & n : cur) {
            addrs.push_back(n.addr);
            tcp_conn_ptr c;
            try {
                c = secure_connect(n.addr, cctx, n.pin);
            } catch (const std::exception & e) {
                throw chain_failure(e.what());
            }
            conns.push_back(c);
            start_reader(conns.size() - 1, c);
        }
        parts.assign(cur.size(), {});
        // open from the last node, so that every node can attach to an open session of the next one
        for (size_t i = cur.size(); i-- > 0;) {
            open_msg m;
            m.session    = sid;
            m.n_ctx      = o.n_ctx;
            m.n_top      = o.n_top;
            m.hidden     = o.hidden;
            m.n_rollback = o.n_rollback;
            m.token      = token_for(cur[i]);
            if (i + 1 < cur.size()) {
                m.next       = cur[i + 1].addr;
                m.next_token = token_for(cur[i + 1]);
                m.next_pin   = cur[i + 1].pin;
            }
            if (!send_frame(*conns[i], msg::open, m.write())) {
                throw chain_failure("node " + addrs[i] + " closed the connection");
            }
            std::vector<uint8_t> body = expect(i, msg::open_ok);
            reader r(body.data(), body.size());
            parts[i] = open_ok_msg::read(r);
        }

        for (size_t i = 0; i < parts.size(); i++) {
            const auto & p = parts[i];
            AN_LOG("node %zu (%s): layers [%d, %d) of %d%s\n", i, addrs[i].c_str(), p.layer_start, p.layer_end, p.n_layer,
                conns[i]->is_tls() ? ", TLS" : "");
            const int32_t want_start = i == 0 ? 0 : parts[i - 1].layer_end;
            if (p.layer_start != want_start || p.desc != parts[0].desc || p.n_layer != parts[0].n_layer) {
                throw std::runtime_error("node " + addrs[i] + " does not continue the chain: layers [" +
                    std::to_string(p.layer_start) + ", " + std::to_string(p.layer_end) + ") of " + p.desc);
            }
        }
        if (parts.back().layer_end != parts.back().n_layer) {
            throw std::runtime_error("the chain stops at layer " + std::to_string(parts.back().layer_end));
        }
    }

    void close_chain(bool polite) {
        for (auto & c : conns) {
            if (polite) {
                writer w;
                w.put(sid);
                send_frame(*c, msg::close, w);
            }
            c->close();
        }
        conns.clear();
        gen++;
    }

    // waits for frame `want` from node i; an error or a closed connection of any node is a failure of the chain
    std::vector<uint8_t> expect(size_t i, msg want, int extra_ms = 0) {
        const int  wait_ms  = o.timeout_ms + extra_ms;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);
        std::unique_lock<std::mutex> lock(ev->m);
        while (true) {
            while (!ev->q.empty()) {
                chain_events::event e = std::move(ev->q.front());
                ev->q.pop_front();
                if (e.gen != gen) {
                    continue;
                }
                const std::string who = "node " + addrs[e.node];
                if (e.closed) {
                    throw chain_failure(who + " closed the connection");
                }
                if (e.type == msg::error) {
                    reader r(e.body.data(), e.body.size());
                    const std::string text = r.get_str();
                    // a refused access or a wrong chain cannot be fixed by a spare node
                    if (text.rfind("access denied", 0) == 0 || text.find("does not continue") != std::string::npos) {
                        throw std::runtime_error(who + ": " + text);
                    }
                    throw chain_failure(who + ": " + text);
                }
                if (e.node == i && e.type == want) {
                    return std::move(e.body);
                }
            }
            if (ev->cv.wait_until(lock, deadline) == std::cv_status::timeout && ev->q.empty()) {
                throw chain_failure("node " + addrs[i] + " did not answer in " + std::to_string(wait_ms) + " ms");
            }
        }
    }

    std::vector<top_entry> eval_once(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all, bool checkpoint = false) {
        if ((size_t) pos0 > hist.size()) {
            throw std::runtime_error("forward leaves a gap in the context");
        }
        // as the nodes do (model_part::roll_back): the states after pos0 belong to the dropped text
        ckpts.erase(std::remove_if(ckpts.begin(), ckpts.end(), [&](int32_t p) { return p > pos0; }), ckpts.end());
        hist.resize(pos0);
        hist.insert(hist.end(), tokens, tokens + n);

        forward_hdr h;
        h.session  = sid;
        h.step     = ++step;
        h.pos0     = pos0;
        h.n_tokens = n;
        h.kind     = payload::tokens;
        h.flags    = (out_all ? FORWARD_OUT_ALL : 0) | (checkpoint ? FORWARD_CHECKPOINT : 0);
        writer w;
        h.write(w);
        w.put_bytes(tokens, (size_t) n * sizeof(llama_token));
        if (!send_frame(*conns[0], msg::forward, w)) {
            throw chain_failure("node " + addrs[0] + " closed the connection");
        }

        std::vector<uint8_t> body = expect(conns.size() - 1, msg::result, 20 * (int) n);
        reader r(body.data(), body.size());
        result_msg res = result_msg::read(r);
        if (res.step != step || res.n_out != (out_all ? n : 1)) {
            throw std::runtime_error("result for step " + std::to_string(res.step) + ", expected " + std::to_string(step));
        }
        if (checkpoint) {
            const int32_t p = pos0 + (int32_t) n;
            ckpts.erase(std::remove(ckpts.begin(), ckpts.end(), p), ckpts.end());
            if (ckpts.size() >= model_part::max_checkpoints) {
                ckpts.erase(ckpts.begin());
            }
            ckpts.push_back(p);
        }
        if (out_all) {
            return res.top;
        }
        return std::vector<top_entry>(res.top.end() - res.n_top, res.top.end());
    }

    std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all = false,
            bool checkpoint = false) override {
        while (true) {
            try {
                return eval_once(tokens, n, pos0, out_all, checkpoint);
            } catch (const chain_failure & e) {
                if (n_recoveries >= o.max_recoveries) {
                    throw std::runtime_error(std::string("the chain failed again: ") + e.what());
                }
                recover(pos0, e.what());
            }
        }
    }

    // is the node alive and does it run these layers
    bool probe(const node_ref & n, int32_t layer_start, int32_t layer_end, const std::string & desc) {
        try {
            tcp_conn_ptr c = secure_connect(n.addr, cctx, n.pin, 1000);
            c->recv_timeout_ms = 1000;
            if (!send_frame(*c, msg::hello, writer())) {
                return false;
            }
            msg type;
            std::vector<uint8_t> body;
            if (!recv_frame(*c, type, body) || type != msg::hello_ok) {
                return false;
            }
            reader r(body.data(), body.size());
            const hello_ok_msg h = hello_ok_msg::read(r);
            c->close();
            return h.layer_start == layer_start && h.layer_end == layer_end && h.desc == desc;
        } catch (const std::exception &) {
            return false;
        }
    }

    // rebuilds the chain after a failure: dead nodes are replaced by spares of the same layers,
    // then the context before pos0 is read again in batches
    void recover(int32_t pos0, const std::string & why) {
        const double t0 = now_ms();
        n_recoveries++;
        AN_LOG("chain failure: %s - recovering\n", why.c_str());

        const std::vector<open_ok_msg> old_parts = parts;
        close_chain(false);

        for (int attempt = 0;; attempt++) {
            std::set<std::string> in_use;
            for (const auto & n : cur) {
                in_use.insert(n.addr);
            }
            // all nodes are checked at once: a hung node costs one probe timeout, not one per node
            std::vector<std::future<bool>> alive;
            for (size_t i = 0; i < cur.size(); i++) {
                const auto & p = old_parts[i];
                alive.push_back(std::async(std::launch::async, [this, n = cur[i], p]() { return probe(n, p.layer_start, p.layer_end, p.desc); }));
            }
            for (size_t i = 0; i < cur.size(); i++) {
                const auto & p = old_parts[i];
                if (alive[i].get()) {
                    continue;
                }
                bool replaced = false;
                for (const auto & sp : o.spares) {
                    if (!in_use.count(sp.addr) && probe(sp, p.layer_start, p.layer_end, p.desc)) {
                        AN_LOG("node %zu (%s) is down, replaced by %s\n", i, cur[i].addr.c_str(), sp.addr.c_str());
                        cur[i] = sp;
                        in_use.insert(sp.addr);
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) {
                    throw std::runtime_error("node " + cur[i].addr + " failed (" + why + ") and no spare node runs layers [" +
                        std::to_string(p.layer_start) + ", " + std::to_string(p.layer_end) + ")");
                }
            }
            if (can_issue_tokens()) {
                sid = random_session();
            }
            try {
                open_chain();
                break;
            } catch (const chain_failure & e) {
                close_chain(false);
                if (attempt >= 3) {
                    throw std::runtime_error(std::string("cannot rebuild the chain: ") + e.what());
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }

        // the new nodes read the context again; a failure here is handled by the caller's next attempt
        const std::vector<llama_token> pre(hist.begin(), hist.begin() + std::min<size_t>(pos0, hist.size()));
        hist.clear();
        ckpts.clear(); // the new nodes keep no states yet
        for (size_t i = 0; i < pre.size(); i += 512) {
            const uint32_t n = (uint32_t) std::min<size_t>(512, pre.size() - i);
            eval_once(pre.data() + i, n, (int32_t) i, false);
        }
        const double dt = now_ms() - t0;
        recover_ms    += dt;
        recover_max_ms = std::max(recover_max_ms, dt);
        AN_LOG("chain recovered in %.0f ms, %zu tokens of context read again\n", dt, pre.size());
    }

    // how many first tokens of a new prompt can stay in the memory of the chain: up to the longest common start, where
    // the nodes can go on (the end of the memory, or a checkpoint); at least one token of the prompt is left to read
    int32_t reusable(const std::vector<llama_token> & prompt) const {
        size_t common = 0;
        while (common < hist.size() && common < prompt.size() && hist[common] == prompt[common]) {
            common++;
        }
        int32_t best = common == hist.size() && common < prompt.size() ? (int32_t) common : 0;
        for (int32_t c : ckpts) {
            if (c <= (int32_t) common && (size_t) c < prompt.size()) {
                best = std::max(best, c);
            }
        }
        return best;
    }

    std::vector<stats_msg> stats() {
        std::vector<stats_msg> out;
        for (size_t i = 0; i < conns.size(); i++) {
            writer w;
            w.put(sid);
            send_frame(*conns[i], msg::stats, w);
            std::vector<uint8_t> body = expect(i, msg::stats_ok);
            reader r(body.data(), body.size());
            out.push_back(stats_msg::read(r));
        }
        return out;
    }
};

struct local_eval : evaluator {
    model_part part;
    uint32_t   n_top;

    local_eval(const llama_model * model, const model_opts & mo, uint32_t n_ctx, int32_t n_threads, uint32_t n_top)
        : part(model, mo, n_ctx, n_threads), n_top(n_top) {}

    std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all = false, bool = false) override {
        forward_hdr h;
        h.pos0     = pos0;
        h.n_tokens = n;
        h.kind     = payload::tokens;
        h.flags    = out_all ? FORWARD_OUT_ALL : 0;
        std::vector<float>     hidden;
        std::vector<top_entry> top;
        uint32_t n_out = 0;
        part.run(h, (const uint8_t *) tokens, hidden, top, n_top, n_out);
        return top;
    }
};

struct dump_step {
    llama_token            token;
    std::vector<top_entry> top;
};

static std::vector<dump_step> read_dump(const std::string & path) {
    std::ifstream f(u8_path(path));
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<dump_step> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        int step;
        dump_step d;
        ss >> step >> d.token;
        std::string item;
        while (ss >> item) {
            const size_t p = item.find(':');
            d.top.push_back({ std::stoi(item.substr(0, p)), std::stof(item.substr(p + 1)) });
        }
        out.push_back(std::move(d));
    }
    return out;
}

struct gen_opts {
    std::string prompt;
    bool        chat       = true;
    int32_t     n_gen      = 64;
    uint32_t    n_ctx      = 4096;
    uint32_t    n_top      = 8;
    int32_t     n_batch    = 512;
    bool        ignore_eos = false;
    std::string dump;
    std::string json;
    std::vector<llama_token> force; // follow these tokens instead of the greedy choice (from a reference dump)
    payload     hidden = payload::q8;
    // every generated token goes here instead of stdout; false stops the generation (the reader went away)
    std::function<bool(llama_token)> on_token;
    // sampling from the top-k the last node returns: temperature 0 is greedy
    float    temperature = 0.0f;
    float    top_p       = 1.0f;
    uint32_t seed        = 0;
    // a conversation that continues: the first n_cached tokens of the prompt are in the memory of the chain already
    // (the evaluator holds them); checkpoints: positions of the prompt where the chain keeps its state for later requests
    int32_t  n_cached    = 0;
    std::vector<int32_t> checkpoints;
    // the same places for the draft model on this PC (it keeps its own states, also before n_cached)
    std::vector<int32_t> marks;
};

// the next token from the top-k of one position (sorted by logit): greedy, or sampled with the temperature and top-p.
// With a draft model the same choice is made at every checked position, and a draft token is accepted only when the
// choice equals it: the text keeps the distribution of the model alone
static llama_token choose_token(const top_entry * row, uint32_t n_top, const gen_opts & g, std::mt19937 & rng) {
    if (g.temperature <= 0.0f || n_top <= 1) {
        return row[0].id;
    }
    std::vector<double> p(n_top);
    double sum = 0;
    for (uint32_t i = 0; i < n_top; i++) {
        p[i] = std::exp((row[i].logit - row[0].logit) / g.temperature);
        sum += p[i];
    }
    // nucleus: the smallest prefix with the probability top_p
    uint32_t n = n_top;
    if (g.top_p < 1.0f) {
        double acc = 0;
        for (uint32_t i = 0; i < n_top; i++) {
            acc += p[i] / sum;
            if (acc >= g.top_p) {
                n = i + 1;
                break;
            }
        }
    }
    std::discrete_distribution<uint32_t> d(p.begin(), p.begin() + n);
    return row[d(rng)].id;
}

static gen_opts gen_opts_from(const cli_args & a) {
    gen_opts g;
    g.prompt = a.get("-p", "--prompt", "");
    if (a.has("-f") || a.has("--file")) {
        g.prompt = read_file(a.get("-f", "--file", ""));
    }
    if (g.prompt.empty()) {
        g.prompt = "Explain in a few sentences how a distributed language model can run on many home computers.";
    }
    g.chat       = !a.has("--no-chat");
    g.n_gen      = a.get_int("-n", "--n-predict", 64);
    g.n_ctx      = (uint32_t) a.get_int("-c", "--ctx-size", 4096);
    g.n_top      = (uint32_t) a.get_int("--top", "", 8);
    g.ignore_eos = a.has("--ignore-eos");
    g.dump       = a.get("--dump", "", "");
    g.json       = a.get("--json", "", "");
    if (a.has("--force")) {
        for (const auto & d : read_dump(a.get("--force", "", ""))) {
            g.force.push_back(d.token);
        }
    }
    const std::string hidden = a.get("--hidden", "", "q8");
    if (hidden != "f16" && hidden != "f32" && hidden != "q8") {
        throw std::runtime_error("--hidden must be f16, f32 or q8");
    }
    g.hidden = hidden == "f32" ? payload::f32 : hidden == "q8" ? payload::q8 : payload::f16;
    return g;
}

static std::vector<llama_token> make_prompt(const llama_vocab * vocab, const llama_model * model, const gen_opts & g) {
    std::string text = g.prompt;
    if (g.chat) {
        const char * tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl != nullptr && strstr(tmpl, "<|im_start|>") != nullptr) {
            text = "<|im_start|>user\n" + g.prompt + "<|im_end|>\n<|im_start|>assistant\n";
        }
    }
    return common_tokenize(vocab, text, true, true);
}

struct gen_result {
    std::vector<llama_token> tokens;
    int32_t n_prompt   = 0;
    int32_t n_cached   = 0; // of them, in the memory of the chain already (gen_opts::n_cached)
    int32_t n_draft_cached = 0; // in the memory of the draft model already
    double  prefill_ms = 0;
    double  gen_ms     = 0;
    int32_t n_gen_eval = 0; // tokens produced in gen_ms (after the first one, which comes with the prompt)
    // speculative decoding
    int32_t n_rounds   = 0; // passes over the chain
    int32_t n_drafted  = 0;
    int32_t n_accepted = 0;
    double  draft_ms   = 0; // time of the draft model on the client
    double  verify_ms  = 0; // time of the passes over the chain that check the drafts
};

// reads the prompt after the n_cached tokens the chain has, in batches; a batch that ends at one of g.checkpoints keeps
// the state of the chain there. Returns the top-k after the last token
static std::vector<top_entry> prefill(evaluator & ev, const std::vector<llama_token> & prompt, const gen_opts & g, gen_result & res) {
    if (g.n_cached < 0 || (size_t) g.n_cached >= prompt.size()) {
        throw std::runtime_error("the cached part of the prompt must leave at least one token to read");
    }
    res.n_prompt = (int32_t) prompt.size();
    res.n_cached = g.n_cached;
    std::vector<top_entry> top;
    for (size_t i = (size_t) g.n_cached; i < prompt.size();) {
        size_t end = std::min(prompt.size(), i + (size_t) g.n_batch);
        for (int32_t c : g.checkpoints) {
            if (c > (int32_t) i && (size_t) c < end) {
                end = (size_t) c;
            }
        }
        const bool keep = std::find(g.checkpoints.begin(), g.checkpoints.end(), (int32_t) end) != g.checkpoints.end();
        top = ev.eval(prompt.data() + i, (uint32_t) (end - i), (int32_t) i, false, keep);
        i = end;
    }
    return top;
}

// before_gen is called after the prompt, when the timed generation starts
static gen_result generate(evaluator & ev, const llama_vocab * vocab, const std::vector<llama_token> & prompt, const gen_opts & g,
        const std::function<void()> & before_gen) {
    if ((uint32_t) (prompt.size() + g.n_gen) > g.n_ctx) {
        throw std::runtime_error("prompt + generation does not fit in the context, increase -c");
    }
    FILE * dump = nullptr;
    if (!g.dump.empty()) {
        dump = fopen_utf8(g.dump, "w");
        if (!dump) {
            throw std::runtime_error("cannot write " + g.dump);
        }
        fprintf(dump, "# agent-node dump v1 n_top=%u n_prompt=%zu\n", g.n_top, prompt.size());
    }

    gen_result res;
    std::mt19937 rng(g.seed);

    double t0 = now_ms();
    std::vector<top_entry> top = prefill(ev, prompt, g, res);
    res.prefill_ms = now_ms() - t0;

    if (before_gen) {
        before_gen();
    }

    int32_t pos = (int32_t) prompt.size();
    t0 = now_ms();
    for (int32_t i = 0; i < g.n_gen; i++) {
        // with --force the context follows the reference, so every step compares logits on the same text
        const llama_token tok = (size_t) i < g.force.size() ? g.force[i] : choose_token(top.data(), (uint32_t) top.size(), g, rng);
        res.tokens.push_back(tok);
        if (dump) {
            fprintf(dump, "%d %d", i, tok);
            for (const auto & e : top) {
                fprintf(dump, " %d:%.6f", e.id, e.logit);
            }
            fprintf(dump, "\n");
        }
        if (g.on_token) {
            if (!g.on_token(tok)) {
                break;
            }
        } else {
            printf("%s", common_token_to_piece(vocab, tok, false).c_str());
            fflush(stdout);
        }

        if (i + 1 == g.n_gen || (!g.ignore_eos && llama_vocab_is_eog(vocab, tok))) {
            break;
        }
        top = ev.eval(&tok, 1, pos++);
        res.n_gen_eval++;
    }
    res.gen_ms = now_ms() - t0;
    if (!g.on_token) {
        printf("\n");
    }
    if (dump) {
        fclose(dump);
    }
    return res;
}

// the draft model of speculative decoding, on the client
//
// it proposes tokens one at a time, so its recurrent snapshots (which cover only the last decode) cannot undo them:
// before every draft the recurrent state is saved (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, a few MB), after the check
// it is restored and the accepted tokens are decoded again in one batch
struct draft_model {
    llama_model_ptr      model;
    llama_context_ptr    ctx;
    int32_t              n_vocab = 0;
    std::vector<uint8_t> ckpt;
    std::vector<top_entry> top1 = std::vector<top_entry>(1);

    // the prompt of the last request (its tokens decoded in the prefill) and the states saved inside it, as the chain
    // keeps them (gen_opts::checkpoints): the next request of the conversation starts at the longest shared one
    std::vector<llama_token> seen;
    struct mark {
        int32_t              pos;
        std::vector<uint8_t> data;
    };
    std::vector<mark> marks;

    draft_model(const model_opts & mo, uint32_t n_ctx, int32_t n_threads) {
        model   = load_model(mo);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = 512;
        cp.n_ubatch        = 512;
        cp.n_seq_max       = 1;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.no_perf         = true;
        ctx.reset(llama_init_from_model(model.get(), cp));
        if (!ctx) {
            throw std::runtime_error("failed to create the context of the draft model");
        }
    }

    // a new text: forget the previous one (the local API serves many requests with one draft model)
    void reset() {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        seen.clear();
        marks.clear();
    }

    // a new prompt: back to the longest mark inside the start it shares with the last prompt; returns the position
    // the prefill continues at (0: from the start)
    int32_t start(const std::vector<llama_token> & prompt) {
        size_t common = 0;
        while (common < seen.size() && common < prompt.size() && seen[common] == prompt[common]) {
            common++;
        }
        marks.erase(std::remove_if(marks.begin(), marks.end(), [&](const mark & m) { return (size_t) m.pos > common; }), marks.end());
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        for (auto it = marks.rbegin(); it != marks.rend(); ++it) {
            if ((size_t) it->pos + 1 < prompt.size() &&
                    llama_state_seq_set_data_ext(ctx.get(), it->data.data(), it->data.size(), 0, fl) == it->data.size() &&
                    llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, it->pos, -1)) {
                const int32_t pos = it->pos;
                seen.resize(pos);
                return pos;
            }
        }
        reset();
        return 0;
    }

    // the tokens before pos are decoded: keep the state there (the 4 latest)
    void keep(int32_t pos) {
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        marks.erase(std::remove_if(marks.begin(), marks.end(), [&](const mark & m) { return m.pos == pos; }), marks.end());
        if (marks.size() >= 4) {
            marks.erase(marks.begin());
        }
        mark m{ pos, std::vector<uint8_t>(llama_state_seq_get_size_ext(ctx.get(), 0, fl)) };
        if (llama_state_seq_get_data_ext(ctx.get(), m.data.data(), m.data.size(), 0, fl) == m.data.size()) {
            marks.push_back(std::move(m));
        }
    }

    // decodes n tokens from pos0, returns the greedy next token
    llama_token decode(const llama_token * tokens, uint32_t n, int32_t pos0) {
        common_batch batch(ctx.get());
        for (uint32_t i = 0; i < n; i++) {
            batch.add(tokens[i], pos0 + (int32_t) i, 0, i + 1 == n);
        }
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("draft model: llama_process failed");
        }
        top_k(llama_get_logits_ith(ctx.get(), (int32_t) n - 1), n_vocab, 1, top1.data());
        return top1[0].id;
    }

    void save() {
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        ckpt.resize(llama_state_seq_get_size_ext(ctx.get(), 0, fl));
        if (!ckpt.empty() && llama_state_seq_get_data_ext(ctx.get(), ckpt.data(), ckpt.size(), 0, fl) != ckpt.size()) {
            throw std::runtime_error("draft model: cannot save the recurrent state");
        }
    }

    // back to the state saved at position pos (the tokens before pos are processed)
    void restore(int32_t pos) {
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        if (!ckpt.empty() && llama_state_seq_set_data_ext(ctx.get(), ckpt.data(), ckpt.size(), 0, fl) != ckpt.size()) {
            throw std::runtime_error("draft model: cannot restore the recurrent state");
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, pos, -1)) {
            throw std::runtime_error("draft model: cannot remove the rejected tokens");
        }
    }
};

// speculative decoding over the chain: the draft model proposes n_draft tokens, one pass over the chain checks
// them all, the longest matching prefix is accepted plus the token of the chain after it. Greedy, so the text is the
// one the chain would write alone (up to the numeric difference of checking several tokens in one batch)
// noise: for tests, the share of draft tokens replaced by another token (to exercise partial acceptance)
static gen_result generate_spec(chain_eval & ev, draft_model & dm, const llama_vocab * vocab, const std::vector<llama_token> & prompt,
        const gen_opts & g, int32_t n_draft, double noise, const std::function<void()> & before_gen) {
    std::mt19937 rng(42);
    std::mt19937 pick(g.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if ((uint32_t) (prompt.size() + g.n_gen + n_draft + 1) > g.n_ctx) {
        throw std::runtime_error("prompt + generation does not fit in the context, increase -c");
    }
    FILE * dump = nullptr;
    if (!g.dump.empty()) {
        dump = fopen_utf8(g.dump, "w");
        if (!dump) {
            throw std::runtime_error("cannot write " + g.dump);
        }
        fprintf(dump, "# agent-node dump v1 n_top=%u n_prompt=%zu draft=%d\n", g.n_top, prompt.size(), n_draft);
    }

    gen_result res;

    double t0 = now_ms();
    std::vector<top_entry> top = prefill(ev, prompt, g, res);
    res.prefill_ms = now_ms() - t0;

    // the draft model reads the prompt too, except its last token: every round starts by decoding the last token.
    // A conversation that goes on starts at a mark of the last request (the places the chain keeps its state at)
    t0 = now_ms();
    const size_t n_read = prompt.size() - 1;
    res.n_draft_cached = dm.start(prompt);
    for (size_t i = (size_t) res.n_draft_cached; i < n_read;) {
        size_t end = std::min(n_read, i + (size_t) g.n_batch);
        for (int32_t m : g.marks) {
            if (m > (int32_t) i && (size_t) m < end) {
                end = (size_t) m;
            }
        }
        dm.decode(prompt.data() + i, (uint32_t) (end - i), (int32_t) i);
        if (std::find(g.marks.begin(), g.marks.end(), (int32_t) end) != g.marks.end()) {
            dm.keep((int32_t) end);
        }
        i = end;
    }
    dm.seen.assign(prompt.begin(), prompt.begin() + (ptrdiff_t) n_read);
    res.draft_ms += now_ms() - t0;

    if (before_gen) {
        before_gen();
    }

    // emits one token, returns true when the generation is over
    int32_t n_out = 0;
    auto emit = [&](llama_token tok, const top_entry * row) {
        res.tokens.push_back(tok);
        if (dump) {
            fprintf(dump, "%d %d", n_out, tok);
            for (uint32_t k = 0; row && k < g.n_top; k++) {
                fprintf(dump, " %d:%.6f", row[k].id, row[k].logit);
            }
            fprintf(dump, "\n");
        }
        n_out++;
        if (g.on_token) {
            if (!g.on_token(tok)) {
                return true;
            }
        } else {
            printf("%s", common_token_to_piece(vocab, tok, false).c_str());
            fflush(stdout);
        }
        return n_out == g.n_gen || (!g.ignore_eos && llama_vocab_is_eog(vocab, tok));
    };

    // the first token comes from the prompt; the draft model has not seen the last prompt token yet
    llama_token last = choose_token(top.data(), (uint32_t) top.size(), g, pick);
    int32_t     pos  = (int32_t) prompt.size(); // position of `last` for the chain
    bool done = emit(last, top.data());

    // the draft model lags one token behind: it still has to process prompt.back() at pos - 1
    {
        const double td = now_ms();
        dm.decode(&prompt.back(), 1, pos - 1);
        res.draft_ms += now_ms() - td;
    }

    t0 = now_ms();
    std::vector<llama_token> batch;
    while (!done) {
        const int32_t k = std::min<int32_t>(n_draft, g.n_gen - n_out);

        // 1. the draft model proposes k tokens after `last`
        const double td = now_ms();
        dm.save();
        batch.assign(1, last);
        llama_token cur = last;
        for (int32_t j = 0; j < k; j++) {
            cur = dm.decode(&cur, 1, pos + j);
            if (noise > 0 && uni(rng) < noise) {
                cur = (cur + 1) % dm.n_vocab;
            }
            batch.push_back(cur);
        }
        res.draft_ms += now_ms() - td;

        // 2. one pass over the chain: the greedy choice of the model after every token of the batch
        const double tv = now_ms();
        const std::vector<top_entry> rows = ev.eval(batch.data(), (uint32_t) batch.size(), pos, true);
        res.verify_ms += now_ms() - tv;
        res.n_rounds++;
        res.n_drafted += k;

        // 3. accept the matching prefix, then the token of the chain after it; with sampling every position is
        //    sampled once, in order, and the first sample that differs from the draft ends the round
        int32_t a = 0;
        llama_token chosen = choose_token(&rows[0], g.n_top, g, pick);
        while (a < k && chosen == batch[a + 1]) {
            a++;
            chosen = choose_token(&rows[(size_t) a * g.n_top], g.n_top, g, pick);
        }
        res.n_accepted += a;
        for (int32_t j = 0; j < a && !done; j++) {
            done = emit(batch[j + 1], &rows[(size_t) j * g.n_top]);
        }
        if (done) {
            break;
        }
        const llama_token bonus = chosen;
        done = emit(bonus, &rows[(size_t) a * g.n_top]);

        // 4. the draft model goes back to `pos` and reads the accepted tokens in one batch;
        //    the nodes roll back by themselves when the next forward starts at pos + a + 1
        const double tr = now_ms();
        dm.restore(pos);
        dm.decode(batch.data(), (uint32_t) a + 1, pos);
        res.draft_ms += now_ms() - tr;

        last = bonus;
        pos += a + 1;
    }
    res.gen_ms     = now_ms() - t0;
    res.n_gen_eval = (int32_t) res.tokens.size() - 1;
    if (!g.on_token) {
        printf("\n");
    }
    if (dump) {
        fclose(dump);
    }
    return res;
}

static void print_speed(const gen_result & r) {
    AN_LOG("prompt: %d tokens in %.1f ms (%.1f tok/s)\n", r.n_prompt, r.prefill_ms, r.n_prompt * 1000.0 / std::max(r.prefill_ms, 1e-9));
    if (r.n_gen_eval > 0) {
        AN_LOG("generation: %d tokens in %.1f ms (%.2f tok/s, %.2f ms/token)\n", r.n_gen_eval, r.gen_ms,
            r.n_gen_eval * 1000.0 / r.gen_ms, r.gen_ms / r.n_gen_eval);
    }
    if (r.n_rounds > 0) {
        AN_LOG("speculative: %d passes over the chain, %.2f tokens per pass, %d of %d draft tokens accepted (%.0f%%), draft model %.1f ms\n",
            r.n_rounds, (double) r.tokens.size() / r.n_rounds, r.n_accepted, r.n_drafted,
            100.0 * r.n_accepted / std::max(1, r.n_drafted), r.draft_ms);
        AN_LOG("per pass: draft model %.1f ms, chain %.1f ms\n", r.draft_ms / r.n_rounds, r.verify_ms / r.n_rounds);
    }
}

static int cmd_client(const cli_args & a) {
    const gen_opts g = gen_opts_from(a);

    std::vector<std::string> addrs;
    {
        std::stringstream ss(a.get("--chain", "", ""));
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                addrs.push_back(item);
            }
        }
    }
    if (addrs.empty() == !a.has("--coordinator")) {
        throw std::runtime_error("either --chain host:port,host:port,... or --coordinator URL is required");
    }

    // the client only needs the tokenizer
    model_opts mo = model_opts_from(a);
    mo.vocab_only = true;
    llama_model_ptr model = load_model(mo);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    // a chain from the coordinator: nodes and spares with their pins and tokens for one session
    std::vector<node_ref> coord_nodes, coord_spares;
    uint64_t coord_session = 0;
    coord_account acc;
    std::string session_hex;
    if (a.has("--coordinator")) {
        acc = account_from(a);
        const coord_chain c = request_chain(acc, model_key(mo.path), a.get_int("--spares", "", 1), a.get_int("--token-ttl", "", 3600),
            a.get("--city", "", ""));
        coord_nodes   = c.nodes;
        coord_spares  = c.spares;
        coord_session = c.session;
        session_hex   = c.session_hex;
        for (const auto & n : coord_nodes) {
            addrs.push_back(n.addr);
        }
        AN_LOG("coordinator: chain of %zu nodes and %zu spares for session %s in %.0f ms (%d layer ranges without a spare)\n",
            coord_nodes.size(), coord_spares.size(), session_hex.c_str(), c.ms, c.no_spare);
    }

    const std::vector<llama_token> prompt = make_prompt(vocab, model.get(), g);

    // speculative decoding: a small draft model on the client, --draft-max tokens per pass over the chain
    std::unique_ptr<draft_model> dm;
    const int32_t n_draft = a.get_int("--draft-max", "", 6);
    if (a.has("--draft")) {
        if (n_draft < 1 || n_draft > 32) {
            throw std::runtime_error("--draft-max must be 1..32");
        }
        if (!g.force.empty()) {
            throw std::runtime_error("--force and --draft cannot be combined");
        }
        model_opts dmo;
        dmo.path         = a.get("--draft", "", "");
        dmo.n_gpu_layers = a.get_int("-ngld", "--draft-ngl", 999);
        dmo.device       = a.get("--device", "-dev", "");
        dm = std::make_unique<draft_model>(dmo, g.n_ctx, a.get_int("-t", "--threads", common_cpu_get_num_math()));
        if (dm->n_vocab != llama_vocab_n_tokens(vocab)) {
            throw std::runtime_error("the draft model has another vocabulary than the chain");
        }
    }

    // the nodes: addresses with, aligned by position, secret keys or coordinator tokens and certificate pins
    auto list = [&](const std::string & name) {
        std::vector<std::string> out;
        const std::string v = a.get(name, "", "");
        if (v.empty()) {
            return out;
        }
        std::stringstream ss(v);
        std::string item;
        while (std::getline(ss, item, ',')) {
            out.push_back(item);
        }
        if (!v.empty() && v.back() == ',') {
            out.push_back("");
        }
        return out;
    };
    auto refs = [&](const std::vector<std::string> & ad, const std::string & prefix) {
        const auto keys = list(prefix + "keys"), tokens = list(prefix + "tokens"), pins = list(prefix + "pins");
        for (const auto * l : { &keys, &tokens, &pins }) {
            if (!l->empty() && l->size() != ad.size()) {
                throw std::runtime_error(prefix + "keys, " + prefix + "tokens and " + prefix + "pins need one entry per node");
            }
        }
        std::vector<node_ref> out;
        for (size_t i = 0; i < ad.size(); i++) {
            out.push_back({ ad[i], keys.empty() ? "" : keys[i], tokens.empty() ? "" : tokens[i], pins.empty() ? "" : pins[i] });
        }
        return out;
    };
    std::vector<std::string> spare_addrs;
    for (const auto & sa : list("--spare")) {
        if (!sa.empty()) {
            spare_addrs.push_back(sa);
        }
    }

    chain_opts co;
    if (a.has("--coordinator")) {
        co.nodes   = coord_nodes;
        co.spares  = coord_spares;
        co.session = coord_session;
        co.tls     = true;
    } else {
        co.nodes  = refs(addrs, "--");
        co.spares = refs(spare_addrs, "--spare-");
        co.tls    = a.has("--tls");
    }
    co.timeout_ms = a.get_int("--timeout", "", 3000);
    co.n_ctx      = g.n_ctx;
    co.n_top      = g.n_top;
    co.hidden     = g.hidden;
    co.n_rollback = dm ? (uint32_t) n_draft : 0;
    co.token_ttl  = (uint64_t) a.get_int("--token-ttl", "", 3600);
    if (a.has("--session") && !a.has("--coordinator")) {
        co.session = std::stoull(a.get("--session", "", ""), nullptr, 16);
    }
    chain_eval chain(co);
    if (chain.parts[0].n_vocab != llama_vocab_n_tokens(vocab)) {
        throw std::runtime_error("the tokenizer does not match the model of the chain");
    }

    std::vector<stats_msg> st0;
    auto before_gen = [&]() { st0 = chain.stats(); };
    const double noise = std::stod(a.get("--draft-noise", "", "0"));
    const gen_result r = dm ? generate_spec(chain, *dm, vocab, prompt, g, n_draft, noise, before_gen)
                            : generate(chain, vocab, prompt, g, before_gen);
    const std::vector<stats_msg> st1 = chain.stats();

    print_speed(r);
    if (!session_hex.empty()) {
        long long charged = 0;
        const long long bal = report_session(acc, session_hex, r.n_prompt, (int32_t) r.tokens.size(), &charged);
        if (bal >= 0) {
            AN_LOG("points: %lld for this answer, %lld left\n", charged, bal);
        }
    }
    if (chain.n_recoveries > 0) {
        AN_LOG("recoveries: %d, the longest pause %.0f ms (counters of the nodes restarted)\n", chain.n_recoveries, chain.recover_max_ms);
        st0.assign(st1.size(), stats_msg());
    }

    // traffic and compute of the timed generation per generated token
    // (with a draft model the nodes also process the rejected draft tokens)
    std::ostringstream nodes_json;
    for (size_t i = 0; i < st1.size(); i++) {
        const double n   = std::max<double>(1, (double) r.n_gen_eval);
        const double in  = (st1[i].bytes_in  - st0[i].bytes_in)  / n;
        const double out = (st1[i].bytes_out - st0[i].bytes_out) / n;
        const double cms = (st1[i].compute_ms - st0[i].compute_ms) / n;
        const double oms = (st1[i].output_ms - st0[i].output_ms) / n;
        AN_LOG("node %zu: %.0f bytes in + %.0f bytes out per token, compute %.2f ms/token (of it reading outputs %.2f)\n", i, in, out, cms, oms);
        nodes_json << (i ? "," : "") << "{\"addr\":\"" << chain.addrs[i] << "\",\"layer_start\":" << chain.parts[i].layer_start
                   << ",\"layer_end\":" << chain.parts[i].layer_end << ",\"bytes_in_per_token\":" << in
                   << ",\"bytes_out_per_token\":" << out << ",\"compute_ms_per_token\":" << cms
                   << ",\"output_ms_per_token\":" << oms << "}";
    }

    if (!g.json.empty()) {
        std::ofstream f(u8_path(g.json));
        f << "{\"n_nodes\":" << addrs.size() << ",\"n_embd\":" << chain.parts[0].n_embd << ",\"n_prompt\":" << r.n_prompt
          << ",\"prefill_ms\":" << r.prefill_ms << ",\"n_gen\":" << r.n_gen_eval << ",\"gen_ms\":" << r.gen_ms
          << ",\"tg_tok_s\":" << (r.n_gen_eval > 0 ? r.n_gen_eval * 1000.0 / r.gen_ms : 0.0)
          << ",\"pp_tok_s\":" << r.n_prompt * 1000.0 / std::max(r.prefill_ms, 1e-9)
          << ",\"draft_max\":" << (dm ? n_draft : 0) << ",\"n_rounds\":" << r.n_rounds
          << ",\"n_drafted\":" << r.n_drafted << ",\"n_accepted\":" << r.n_accepted << ",\"draft_ms\":" << r.draft_ms << ",\"verify_ms\":" << r.verify_ms
          << ",\"n_recoveries\":" << chain.n_recoveries << ",\"recover_max_ms\":" << chain.recover_max_ms
          << ",\"tls\":" << (co.tls ? "true" : "false")
          << ",\"nodes\":[" << nodes_json.str() << "]}\n";
    }
    return 0;
}

//
// serve: the local API of the agent on the PC of the user (the app talks to it)
//
//   GET  /health
//   GET  /v1/models
//   POST /v1/chat/completions   OpenAI-style: {messages, max_tokens, stream, think}; the reasoning of the model goes to
//                               delta.reasoning_content (the app shows the thinking line there), the answer to delta.content
//   GET  /v1/account            balance and journal of the account of this PC (from the coordinator)
//
// every request takes a chain from the coordinator, runs over it (with the draft model when there is one) and reports
// the session, so the account pays for the answer. One request at a time: the draft model is one.
//

// the longest prefix of s that does not end inside a UTF-8 character
static size_t utf8_complete(const std::string & s) {
    size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; back++) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) {
            continue; // continuation byte
        }
        const size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        return back < len ? n - back : n;
    }
    return n;
}

struct serve_state {
    std::mutex                   mtx; // one generation at a time
    coord_account                acc;
    llama_model_ptr              vocab_model;
    const llama_vocab *          vocab = nullptr;
    common_chat_templates_ptr    tmpls; // the chat template of the model (from the GGUF): prompt, reasoning, tool calls
    std::string                  key;  // model key for the coordinator
    std::string                  name; // model name for /v1/models
    std::unique_ptr<draft_model> dm;
    int32_t                      n_draft = 6;
    uint32_t                     n_ctx   = 8192;
    int                          spares  = 1;
    std::string                  city;

    // the chain of the last request stays open: the next request of the conversation (a step of the agent) starts
    // at the checkpoint before the answer instead of reading the whole conversation again (--keep-chain-s, 0: off).
    // A session lives at most 8 minutes: the coordinator settles the sessions opened more than 10 minutes ago,
    // later reports of a settled session would not be charged
    std::unique_ptr<chain_eval>  chain;
    coord_chain                  chain_info;
    double                       chain_opened = 0, chain_used = 0;
    int                          keep_s = 180;
    static constexpr int         max_age_s = 480;
    int64_t                      billed_prompt = 0, billed_gen = 0; // of the session, for the report (a running total)
};

static json to_nl(const common_json & j) {
    return json::parse(j.dump());
}

// the OpenAI delta of one tool call
static json tool_call_delta(size_t index, const common_chat_tool_call & tc) {
    json d = { { "index", index }, { "function", { { "arguments", tc.arguments } } } };
    if (!tc.id.empty()) {
        d["id"]   = tc.id;
        d["type"] = "function";
    }
    if (!tc.name.empty()) {
        d["function"]["name"] = tc.name;
    }
    return d;
}

static int cmd_serve(const cli_args & a) {
    if (!a.has("--coordinator")) {
        throw std::runtime_error("serve needs --coordinator URL");
    }
    auto st = std::make_shared<serve_state>();
    model_opts mo = model_opts_from(a);
    mo.vocab_only = true;
    st->vocab_model = load_model(mo);
    st->vocab   = llama_model_get_vocab(st->vocab_model.get());
    // --chat-template FILE: a template instead of the one in the GGUF (tests, models with a broken template)
    st->tmpls   = common_chat_templates_init(st->vocab_model.get(), a.has("--chat-template") ? read_file(a.get("--chat-template", "", "")) : "");
    st->key     = model_key(mo.path);
    st->name    = st->key.substr(0, st->key.find(" #"));
    st->n_ctx   = (uint32_t) a.get_int("-c", "--ctx-size", 8192);
    st->spares  = a.get_int("--spares", "", 1);
    st->city    = a.get("--city", "", "");
    st->n_draft = a.get_int("--draft-max", "", 6);
    st->keep_s  = a.get_int("--keep-chain-s", "", 180);
    st->acc     = account_from(a);
    if (a.has("--draft")) {
        model_opts dmo;
        dmo.path         = a.get("--draft", "", "");
        dmo.n_gpu_layers = a.get_int("-ngld", "--draft-ngl", 999);
        dmo.device       = a.get("--device", "-dev", "");
        st->dm = std::make_unique<draft_model>(dmo, st->n_ctx, a.get_int("-t", "--threads", common_cpu_get_num_math()));
        if (st->dm->n_vocab != llama_vocab_n_tokens(st->vocab)) {
            throw std::runtime_error("the draft model has another vocabulary than the model");
        }
    }
    const payload hidden = gen_opts_from(a).hidden;

    httplib::Server http;
    // only the app may call the API: a web page in a browser on this PC must not spend the points of the account.
    // Requests from browsers carry Origin; the app's webview has one of these, curl and scripts have none
    const std::set<std::string> origins = { "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
                                            "http://localhost:1420", "http://127.0.0.1:1420" };
    http.set_pre_routing_handler([origins](const httplib::Request & req, httplib::Response & res) {
        const std::string origin = req.get_header_value("Origin");
        if (!origin.empty()) {
            if (!origins.count(origin)) {
                res.status = 403;
                res.set_content("{\"error\":{\"message\":\"this origin may not use the agent API\"}}", "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
            res.set_header("Access-Control-Allow-Origin", origin);
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    http.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
    http.Get("/v1/models", [st](const httplib::Request &, httplib::Response & res) {
        json j = { { "object", "list" }, { "data", json::array({ { { "id", st->name }, { "object", "model" }, { "owned_by", "agent-p2p" } } }) } };
        res.set_content(j.dump(), "application/json");
    });
    http.Get("/v1/account", [st](const httplib::Request &, httplib::Response & res) {
        if (st->acc.token.empty()) {
            res.status = 404;
            res.set_content("{\"error\":{\"message\":\"no account\"}}", "application/json");
            return;
        }
        const size_t p = st->acc.url.find("://");
        const size_t sl = st->acc.url.find('/', p + 3);
        httplib::Client cli(sl == std::string::npos ? st->acc.url : st->acc.url.substr(0, sl));
        cli.set_connection_timeout(5);
        const std::string path = (sl == std::string::npos ? "" : st->acc.url.substr(sl)) + "/v1/accounts/me";
        auto r = cli.Get(path, st->acc.headers());
        if (r && r->status == 401) {
            // the coordinator forgot the account (restarted with the in-memory store): a new one
            std::lock_guard<std::mutex> lock(st->mtx);
            if (st->acc.load_or_create(true)) {
                r = cli.Get(path, st->acc.headers());
            }
        }
        if (!r) {
            res.status = 502;
            res.set_content(json{ { "error", { { "message", "coordinator: " + httplib::to_string(r.error()) } } } }.dump(), "application/json");
            return;
        }
        res.status = r->status;
        res.set_content(r->body, "application/json");
    });

    http.Post("/v1/chat/completions", [st, hidden](const httplib::Request & req, httplib::Response & res) {
        auto fail = [&](int status, const std::string & msg) {
            res.status = status;
            res.set_content(json{ { "error", { { "message", msg } } } }.dump(), "application/json");
        };
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("messages") || !body["messages"].is_array()) {
            return fail(400, "messages are required");
        }
        // the prompt from the chat template of the model: reasoning on or off, tools in its own format
        auto inputs = std::make_shared<common_chat_templates_inputs>();
        std::shared_ptr<common_chat_session> cs;
        std::vector<llama_token> prompt;
        // where the conversation ends and the answer starts, and where the system message (instructions, tools) ends:
        // the chain keeps its state there, for the next step of the agent and for a new conversation
        int32_t boundary = 0, sys_end = 0;
        try {
            inputs->messages = common_chat_msgs_parse_oaicompat(common_json::parse(body["messages"].dump()));
            if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
                inputs->tools = common_chat_tools_parse_oaicompat(common_json::parse(body["tools"].dump()));
                if (body.contains("tool_choice") && body["tool_choice"].is_string()) {
                    inputs->tool_choice = common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>());
                }
                inputs->parallel_tool_calls = body.value("parallel_tool_calls", false);
            }
            inputs->enable_thinking  = body.value("think", true);
            inputs->reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            cs = std::make_shared<common_chat_session>(st->tmpls.get(), st->vocab, *inputs);
            auto tokenize = [&](const common_chat_prompt & pr) {
                std::vector<llama_token> out;
                bool first = true;
                for (const auto & part : pr.parts) {
                    // text of the user is not parsed for special tokens: a message cannot forge a turn of the template
                    const auto t = common_tokenize(st->vocab, part.text, first, !part.is_input);
                    out.insert(out.end(), t.begin(), t.end());
                    first = false;
                }
                return out;
            };
            prompt = tokenize(cs->prompt());
            // the same conversation without the start of the answer: the next request renders this answer as a message,
            // with the text after it; what both share is the part the next request can keep
            common_chat_templates_inputs past = *inputs;
            past.add_generation_prompt = false;
            const std::vector<llama_token> before = tokenize(common_chat_templates_apply(st->tmpls.get(), past).prompt);
            auto common = [&](const std::vector<llama_token> & x) {
                size_t n = 0;
                while (n < x.size() && n < prompt.size() && x[n] == prompt[n]) {
                    n++;
                }
                return (int32_t) n;
            };
            boundary = common(before);
            common_chat_templates_inputs sys = past;
            sys.messages.clear();
            for (const auto & m : inputs->messages) {
                if (m.role != "system") {
                    break;
                }
                sys.messages.push_back(m);
            }
            if (!sys.messages.empty() && sys.messages.size() < inputs->messages.size()) {
                try {
                    sys_end = common(tokenize(common_chat_templates_apply(st->tmpls.get(), sys).prompt));
                } catch (const std::exception &) {
                    // a template that needs a user message (Qwen3.5): the system message ends where the first user turn starts
                    common_chat_templates_inputs one = sys;
                    common_chat_msg u;
                    u.role    = "user";
                    u.content = "\x01";
                    one.messages.push_back(u);
                    sys_end = common(tokenize(common_chat_templates_apply(st->tmpls.get(), one).prompt));
                }
            }
        } catch (const std::exception & e) {
            return fail(400, std::string("bad chat: ") + e.what());
        }
        if (g_verbose) {
            AN_LOG("serve: prompt (%zu tokens):\n%s\n", prompt.size(), cs->prompt().str().c_str());
        }
        const int32_t room = (int32_t) st->n_ctx - (int32_t) prompt.size() - st->n_draft - 2;
        if (room < 16) {
            return fail(400, "the conversation does not fit in the context");
        }
        const int32_t max_tokens = std::clamp(body.value("max_tokens", 2048), 1, room);
        const float temperature = body.value("temperature", 0.7f);
        const float top_p       = body.value("top_p", 0.95f);
        const uint32_t seed     = body.contains("seed") ? body.value("seed", 0u) : (uint32_t) std::random_device{}();
        const bool stream       = body.value("stream", false);
        const std::string id = "chatcmpl-" + std::to_string((long long) (now_ms() * 1000));
        const long long created = (long long) std::time(nullptr);

        // runs the request; out(delta) gets OpenAI deltas (reasoning_content, content, tool_calls), false: stop.
        // final: the whole message
        auto run = [st, hidden, prompt, boundary, sys_end, max_tokens, temperature, top_p, seed, cs](
                       const std::function<bool(const json &)> & out, json & usage, std::string & finish, common_chat_msg & final_msg) {
            std::lock_guard<std::mutex> lock(st->mtx);
            const double t_start = now_ms();
            if (st->chain && (t_start - st->chain_used > st->keep_s * 1000.0 || t_start - st->chain_opened > st->max_age_s * 1000.0)) {
                st->chain.reset();
            }
            int32_t n_cached = st->chain ? st->chain->reusable(prompt) : 0;
            if (!st->chain) {
                st->chain_info = request_chain(st->acc, st->key, st->spares, 3600, st->city);
                chain_opts co;
                co.nodes      = st->chain_info.nodes;
                co.spares     = st->chain_info.spares;
                co.session    = st->chain_info.session;
                co.tls        = true;
                co.n_ctx      = st->n_ctx;
                co.n_top      = 16; // the candidates for sampling
                co.hidden     = hidden;
                co.n_rollback = st->dm ? (uint32_t) st->n_draft : 0;
                st->chain = std::make_unique<chain_eval>(co);
                st->chain_opened  = t_start;
                st->billed_prompt = st->billed_gen = 0;
                n_cached = 0;
                if (st->chain->parts[0].n_vocab != llama_vocab_n_tokens(st->vocab)) {
                    st->chain.reset();
                    throw std::runtime_error("the tokenizer does not match the model of the chain");
                }
            }
            chain_eval & chain = *st->chain;
            const coord_chain & c = st->chain_info;
            // a failed request leaves the chain in an unknown state: the next one opens a new chain
            struct drop_on_error {
                serve_state & st;
                bool ok = false;
                ~drop_on_error() {
                    if (!ok) {
                        st.chain.reset();
                    }
                }
            } guard{ *st };
            gen_opts g;
            g.n_cached    = n_cached;
            for (int32_t p : { sys_end, boundary }) {
                if (p > n_cached && st->keep_s > 0) {
                    g.checkpoints.push_back(p);
                }
                if (p > 0) {
                    g.marks.push_back(p);
                }
            }
            g.n_gen       = max_tokens;
            g.n_ctx       = st->n_ctx;
            g.n_top       = chain.n_top;
            g.hidden      = hidden;
            g.temperature = temperature;
            g.top_p       = top_p;
            g.seed        = seed;

            common_chat_msg prev;
            prev.role = "assistant";
            std::vector<std::string> ids;
            auto gen_id = []() { return "call_" + std::to_string((long long) (now_ms() * 1000)); };
            std::string pending, raw;
            bool stopped = false, ended = false;
            // the deltas between two parses of the message so far
            auto emit = [&](common_chat_msg msg) {
                msg.set_tool_call_ids(ids, gen_id);
                for (const auto & d : common_chat_msg_diff::compute_diffs(prev, msg)) {
                    json delta = json::object();
                    if (!d.reasoning_content_delta.empty()) {
                        delta["reasoning_content"] = d.reasoning_content_delta;
                    }
                    if (!d.content_delta.empty()) {
                        delta["content"] = d.content_delta;
                    }
                    if (d.tool_call_index != std::string::npos) {
                        delta["tool_calls"] = json::array({ tool_call_delta(d.tool_call_index, d.tool_call_delta) });
                    }
                    if (!delta.empty() && !out(delta)) {
                        return false;
                    }
                }
                prev = msg;
                return true;
            };
            g.on_token = [&](llama_token tok) {
                if (llama_vocab_is_eog(st->vocab, tok)) {
                    ended = true;
                    return true;
                }
                pending += common_token_to_piece(st->vocab, tok, true);
                const size_t n = utf8_complete(pending);
                if (n == 0) {
                    return true;
                }
                common_chat_input chunk;
                chunk.append(pending.substr(0, n), tok);
                raw += pending.substr(0, n);
                pending.erase(0, n);
                if (!emit(cs->feed(chunk))) {
                    stopped = true;
                    return false;
                }
                for (const auto & stop : cs->additional_stops()) {
                    if (!stop.empty() && raw.size() >= stop.size() && raw.compare(raw.size() - stop.size(), stop.size(), stop) == 0) {
                        ended = true;
                        return false;
                    }
                }
                return true;
            };
            const gen_result r = st->dm ? generate_spec(chain, *st->dm, st->vocab, prompt, g, st->n_draft, 0.0, nullptr)
                                        : generate(chain, st->vocab, prompt, g, nullptr);
            common_chat_input tail;
            if (!pending.empty()) {
                tail = common_chat_input(pending);
            }
            final_msg = cs->finish(tail);
            final_msg.set_tool_call_ids(ids, gen_id);
            if (!stopped) {
                emit(final_msg);
            }
            // the nodes read only the tokens after the cached start: the report is a running total of the session
            st->billed_prompt += r.n_prompt - r.n_cached;
            st->billed_gen    += (int64_t) r.tokens.size();
            long long charged = 0;
            const long long balance = report_session(st->acc, c.session_hex, (int32_t) st->billed_prompt, (int32_t) st->billed_gen, &charged);
            guard.ok        = true;
            st->chain_used  = now_ms();
            if (st->keep_s <= 0) {
                st->chain.reset();
            }
            finish = !final_msg.tool_calls.empty() ? "tool_calls" : ended ? "stop" : "length";
            usage = { { "prompt_tokens", r.n_prompt }, { "completion_tokens", (int) r.tokens.size() },
                      { "total_tokens", r.n_prompt + (int) r.tokens.size() },
                      { "prompt_tokens_details", { { "cached_tokens", r.n_cached } } } };
            usage["agent"] = { { "session", c.session_hex }, { "points", charged }, { "balance", balance },
                               { "tok_s", r.n_gen_eval > 0 ? r.n_gen_eval * 1000.0 / r.gen_ms : 0.0 }, { "nodes", c.nodes.size() } };
            AN_LOG("serve: %d + %zu tokens (%d of the prompt kept in the chain%s), %.2f tok/s, %zu tool calls, %lld points (balance %lld)\n",
                r.n_prompt, r.tokens.size(), r.n_cached, st->dm ? (", " + std::to_string(r.n_draft_cached) + " in the draft model").c_str() : "",
                r.n_gen_eval > 0 ? r.n_gen_eval * 1000.0 / r.gen_ms : 0.0, final_msg.tool_calls.size(), charged, balance);
        };

        if (!stream) {
            std::string finish;
            json usage;
            common_chat_msg msg;
            try {
                run([](const json &) { return true; }, usage, finish, msg);
            } catch (const std::exception & e) {
                return fail(std::string(e.what()).find("not enough points") != std::string::npos ? 402 : 503, e.what());
            }
            json m = to_nl(msg.to_json_oaicompat());
            m["role"] = "assistant";
            json j = { { "id", id }, { "object", "chat.completion" }, { "created", created }, { "model", st->name },
                       { "choices", json::array({ { { "index", 0 }, { "message", m }, { "finish_reason", finish } } }) },
                       { "usage", usage } };
            res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [st, run, id, created](size_t, httplib::DataSink & sink) {
            auto send = [&](const json & j) {
                const std::string line = "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                return sink.write(line.data(), line.size());
            };
            auto chunk = [&](const json & delta, const json & finish) {
                return json{ { "id", id }, { "object", "chat.completion.chunk" }, { "created", created }, { "model", st->name },
                             { "choices", json::array({ { { "index", 0 }, { "delta", delta }, { "finish_reason", finish } } }) } };
            };
            send(chunk({ { "role", "assistant" } }, nullptr));
            std::string finish;
            json usage;
            common_chat_msg msg;
            try {
                run([&](const json & delta) { return send(chunk(delta, nullptr)); }, usage, finish, msg);
                json last = chunk(json::object(), finish);
                last["usage"] = usage;
                send(last);
            } catch (const std::exception & e) {
                send({ { "error", { { "message", e.what() } } } });
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    });

    const std::string host = a.get("--host", "", "127.0.0.1");
    const int port = a.get_int("--port", "", 8090);
    if (!http.bind_to_port(host, port)) {
        throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    }
    printf("agent-node serve ready on http://%s:%d (model %s%s)\n", host.c_str(), port, st->name.c_str(), st->dm ? ", with a draft model" : "");
    fflush(stdout);
    http.listen_after_bind();
    return 0;
}

//
// verify: the checker of the network. It holds the whole model, takes a session to a node from the coordinator
// (the same as a client gets) and plays the client, the previous node and the next node of that session: the node
// sees ordinary work. The input is a short text (or, for a node in the middle, the hidden states of the whole model
// before the layers of the node, in int8 as on the network), the output is compared with the whole model.
// A node that returns made-up results is banned by the coordinator.
//

struct probe_ref {
    std::vector<llama_token> tokens;
    std::vector<float>       in;     // hidden states before layer a, per token (empty for a = 0)
    std::vector<float>       out;    // hidden states after layer b - 1 (empty for the last part)
    std::vector<float>       logits; // per token (the last part)
    int32_t                  n_embd = 0, n_vocab = 0;
};

static probe_ref probe_reference(llama_model * model, const std::vector<llama_token> & tokens, int32_t a, int32_t b) {
    probe_ref ref;
    ref.tokens  = tokens;
    ref.n_embd  = llama_model_n_embd(model);
    ref.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_layer = llama_model_n_layer(model);
    struct cap_t {
        std::string want_in, want_out;
        std::vector<float> * in;
        std::vector<float> * out;
    } cap{ a > 0 ? "l_out-" + std::to_string(a - 1) : "", b < n_layer ? "l_out-" + std::to_string(b - 1) : "", &ref.in, &ref.out };
    auto cb = [](ggml_tensor * t, bool ask, void * ud) -> bool {
        auto & c = *(cap_t *) ud;
        const bool hit = t->name == c.want_in || t->name == c.want_out;
        if (ask) {
            return hit;
        }
        if (hit && t->type == GGML_TYPE_F32) {
            std::vector<float> & dst = t->name == c.want_in ? *c.in : *c.out;
            dst.resize((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
        }
        return true;
    };
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx              = (uint32_t) tokens.size() + 16;
    cp.n_batch            = (uint32_t) tokens.size();
    cp.n_ubatch           = (uint32_t) tokens.size();
    cp.n_seq_max          = 1;
    cp.no_perf            = true;
    cp.cb_eval            = cb;
    cp.cb_eval_user_data  = &cap;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("verify: cannot create a context of the whole model");
    }
    common_batch batch(ctx.get());
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.add(tokens[i], (int32_t) i, 0, true);
    }
    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        throw std::runtime_error("verify: the whole model failed on the probe");
    }
    if (b == n_layer) {
        ref.logits.resize(tokens.size() * (size_t) ref.n_vocab);
        for (size_t i = 0; i < tokens.size(); i++) {
            memcpy(ref.logits.data() + i * ref.n_vocab, llama_get_logits_ith(ctx.get(), (int32_t) i), sizeof(float) * ref.n_vocab);
        }
    }
    if ((a > 0 && ref.in.size() != tokens.size() * ref.n_embd) || (b < n_layer && ref.out.size() != tokens.size() * ref.n_embd)) {
        throw std::runtime_error("verify: the hidden states of the whole model were not captured");
    }
    return ref;
}

static double cosine(const float * x, const float * y, int32_t n, double * rel_err) {
    double xy = 0, xx = 0, yy = 0, dd = 0;
    for (int32_t i = 0; i < n; i++) {
        xy += (double) x[i] * y[i];
        xx += (double) x[i] * x[i];
        yy += (double) y[i] * y[i];
        dd += ((double) x[i] - y[i]) * ((double) x[i] - y[i]);
    }
    if (rel_err) {
        *rel_err = std::sqrt(dd / std::max(yy, 1e-30));
    }
    return xy / std::sqrt(std::max(xx * yy, 1e-30));
}

// texts of the probes: ordinary requests, so that a probe looks like work
static const char * PROBE_TEXTS[] = {
    "Помоги написать письмо коллеге о переносе встречи на четверг.",
    "Explain the difference between a process and a thread in simple words.",
    "Как приготовить сырники, чтобы они не разваливались?",
    "Write a Python function that checks whether a string is a palindrome.",
    "Почему небо голубое? Объясни коротко.",
    "Summarize the main idea of the theory of relativity in three sentences.",
    "Составь план тренировок на неделю для начинающего бегуна.",
    "What are the pros and cons of remote work for a small team?",
};

struct probe_result {
    bool        passed = false;
    std::string detail;
};

static probe_result run_probe(llama_model * model, const json & pr, void * cctx, void * sctx, const std::string & bind_host,
                              const std::string & listen_host, int listen_port,
                              const std::string & fingerprint, std::mt19937 & rng) {
    const json & node = pr.at("node");
    const std::string addr  = node.at("addr").get<std::string>();
    const std::string pin   = node.at("pin").get<std::string>();
    const std::string token = node.at("token").get<std::string>();
    const int32_t a = node.at("layer_start").get<int32_t>(), b = node.at("layer_end").get<int32_t>();
    const int32_t n_layer = llama_model_n_layer(model);
    const uint64_t session = std::stoull(pr.at("session").get<std::string>(), nullptr, 16);
    const bool first = a == 0, last = b == n_layer;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the probe: a chat prompt plus a few "generated" tokens, as a client would send them
    const std::string text = PROBE_TEXTS[rng() % (sizeof(PROBE_TEXTS) / sizeof(PROBE_TEXTS[0]))];
    std::vector<llama_token> tokens = common_tokenize(vocab, "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n", true, true);
    const int n_steps = 4;
    {
        // the continuation of the whole model, greedy: the steps of a generation
        probe_ref r0 = probe_reference(model, tokens, 0, n_layer);
        for (int i = 0; i < n_steps; i++) {
            const float * lg = r0.logits.data() + (tokens.size() - 1) * r0.n_vocab;
            tokens.push_back((llama_token) (std::max_element(lg, lg + r0.n_vocab) - lg));
            r0 = probe_reference(model, tokens, 0, n_layer);
        }
    }
    const probe_ref ref = probe_reference(model, tokens, a, b);
    const int32_t n_embd = ref.n_embd;
    const size_t n_prefill = tokens.size() - n_steps;

    // the next node of the session: this checker, listening for the node to attach
    std::vector<float> got;  // hidden states the node sent on, per token
    std::mutex got_mtx;
    std::atomic<bool> listen_err{ false };
    std::unique_ptr<tcp_listener> ln;
    std::thread next_thread;
    if (!last) {
        ln = std::make_unique<tcp_listener>(bind_host, listen_port);
        next_thread = std::thread([&]() {
            try {
                tcp_conn_ptr c = ln->accept();
                if (!c) {
                    listen_err = true;
                    return;
                }
#ifdef AGENT_HAVE_TLS
                if (!c->tls_handshake((SSL_CTX *) sctx, true).empty()) {
                    listen_err = true;
                    return;
                }
#endif
                msg type;
                std::vector<uint8_t> body;
                while (recv_frame(*c, type, body)) {
                    if (type != msg::forward) {
                        continue; // attach
                    }
                    reader r(body.data(), body.size());
                    const forward_hdr h = forward_hdr::read(r);
                    const uint8_t * data = r.get_bytes(payload_size(h.kind, h.n_tokens, n_embd));
                    std::vector<float> rows((size_t) h.n_tokens * n_embd);
                    if (h.kind == payload::q8) {
                        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(data, rows.data(), (int64_t) rows.size());
                    } else if (h.kind == payload::f16) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, rows.data(), (int64_t) rows.size());
                    } else if (h.kind == payload::f32) {
                        memcpy(rows.data(), data, rows.size() * sizeof(float));
                    }
                    std::lock_guard<std::mutex> lock(got_mtx);
                    got.insert(got.end(), rows.begin(), rows.end());
                }
            } catch (const std::exception &) {
                listen_err = true;
            }
        });
    }
    struct joiner {
        std::thread & t;
        std::unique_ptr<tcp_listener> & l;
        ~joiner() {
            if (t.joinable()) {
                l.reset(); // closes the socket, accept returns
                t.detach();
            }
        }
    } join_guard{ next_thread, ln };

    // the client of the session
    tcp_conn_ptr client = secure_connect(addr, cctx, pin);
    open_msg om;
    om.session = session;
    om.n_ctx   = 1024;
    om.n_top   = 8;
    om.hidden  = payload::q8;
    om.token   = token;
    if (!last) {
        om.next       = listen_host + ":" + std::to_string(listen_port);
        om.next_token = std::to_string(rng()) + std::to_string(rng());
        om.next_pin   = fingerprint;
    }
    if (!send_frame(*client, msg::open, om.write())) {
        throw std::runtime_error("the node closed the connection");
    }
    auto expect = [](tcp_conn & c, msg want) {
        msg type;
        std::vector<uint8_t> body;
        if (!recv_frame(c, type, body)) {
            throw std::runtime_error("the node closed the connection");
        }
        if (type == msg::error) {
            reader r(body.data(), body.size());
            throw std::runtime_error("the node: " + r.get_str());
        }
        if (type != want) {
            throw std::runtime_error("unexpected frame from the node");
        }
        return body;
    };
    expect(*client, msg::open_ok);

    // the previous node of the session (or the client itself for the first part)
    tcp_conn_ptr in = client;
    if (!first) {
        in = secure_connect(addr, cctx, pin);
        attach_msg am;
        am.session = session;
        am.token   = token;
        send_frame(*in, msg::attach, am.write());
    }
    auto send_forward = [&](size_t pos0, size_t n, uint32_t step) {
        forward_hdr h;
        h.session  = session;
        h.step     = step;
        h.pos0     = (int32_t) pos0;
        h.n_tokens = (uint32_t) n;
        h.kind     = first ? payload::tokens : payload::q8;
        writer w;
        h.write(w);
        if (first) {
            w.put_bytes(tokens.data() + pos0, n * sizeof(llama_token));
        } else {
            ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(ref.in.data() + pos0 * n_embd,
                w.grow(payload_size(payload::q8, (uint32_t) n, n_embd)), (int64_t) n * n_embd);
        }
        if (!send_frame(*in, msg::forward, w)) {
            throw std::runtime_error("the node closed the connection");
        }
    };
    std::vector<std::vector<top_entry>> tops; // the last part: top-k per checked position
    std::vector<size_t> top_pos;
    uint32_t step = 0;
    send_forward(0, n_prefill, step++);
    if (last) {
        auto body = expect(*client, msg::result);
        reader r(body.data(), body.size());
        const result_msg rm = result_msg::read(r);
        tops.push_back(rm.top);
        top_pos.push_back(n_prefill - 1);
    }
    for (size_t i = n_prefill; i < tokens.size(); i++) {
        send_forward(i, 1, step++);
        if (last) {
            auto body = expect(*client, msg::result);
            reader r(body.data(), body.size());
            const result_msg rm = result_msg::read(r);
            tops.push_back(rm.top);
            top_pos.push_back(i);
        }
    }
    if (!last) {
        // wait for the outputs of all tokens
        for (int i = 0; i < 300; i++) {
            {
                std::lock_guard<std::mutex> lock(got_mtx);
                if (got.size() >= tokens.size() * n_embd) {
                    break;
                }
            }
            if (listen_err) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    writer cw;
    cw.put<uint64_t>(session);
    send_frame(*client, msg::close, cw);
    client->close();
    if (in != client) {
        in->close();
    }

    probe_result res;
    char buf[256];
    if (!last) {
        std::lock_guard<std::mutex> lock(got_mtx);
        if (got.size() < tokens.size() * n_embd) {
            res.detail = "the node sent " + std::to_string(got.size() / n_embd) + " of " + std::to_string(tokens.size()) + " outputs";
            return res;
        }
        double min_cos = 1, max_err = 0;
        for (size_t t = 0; t < tokens.size(); t++) {
            double err = 0;
            const double c = cosine(got.data() + t * n_embd, ref.out.data() + t * n_embd, n_embd, &err);
            min_cos = std::min(min_cos, c);
            max_err = std::max(max_err, err);
        }
        res.passed = min_cos >= 0.98 && max_err <= 0.25;
        snprintf(buf, sizeof(buf), "layers %d:%d, %zu tokens, cosine >= %.5f, relative error <= %.4f", a, b, tokens.size(), min_cos, max_err);
    } else {
        int same = 0;
        double max_rel = 0;
        for (size_t k = 0; k < tops.size(); k++) {
            const float * lg = ref.logits.data() + top_pos[k] * ref.n_vocab;
            const float mx = *std::max_element(lg, lg + ref.n_vocab);
            float scale = 1e-6f;
            for (int32_t v = 0; v < ref.n_vocab; v++) {
                scale = std::max(scale, std::fabs(lg[v]));
            }
            const llama_token best = (llama_token) (std::max_element(lg, lg + ref.n_vocab) - lg);
            same += !tops[k].empty() && tops[k][0].id == best;
            for (const auto & e : tops[k]) {
                if (e.id >= 0 && e.id < ref.n_vocab) {
                    max_rel = std::max(max_rel, (double) std::fabs(e.logit - lg[e.id]) / scale);
                }
            }
            (void) mx;
        }
        res.passed = same * 4 >= (int) tops.size() * 3 && max_rel <= 0.1;
        snprintf(buf, sizeof(buf), "layers %d:%d (last), same top-1 %d of %zu, logit error <= %.4f", a, b, same, tops.size(), max_rel);
    }
    res.detail = buf;
    return res;
}

static int cmd_verify(const cli_args & a) {
    const std::string url = a.get("--coordinator", "", "");
    std::string vtok = a.get("--verifier-token", "", "");
    if (vtok.empty()) {
        if (const char * e = getenv("VERIFIER_TOKEN")) {
            vtok = e;
        }
    }
    if (url.empty() || vtok.empty()) {
        throw std::runtime_error("verify needs --coordinator URL and --verifier-token (or env VERIFIER_TOKEN)");
    }
#ifndef AGENT_HAVE_TLS
    require_tls("verify");
    return 1;
#else
    model_opts mo = model_opts_from(a);
    mo.layer_start = 0;
    mo.layer_end   = 0;
    llama_model_ptr model = load_model(mo);
    const std::string key = model_key(mo.path);
    tls_identity identity;
    identity.load_or_create(a.get("--identity", "", "verifier.pem"));
    SSL_CTX * sctx = tls_server_ctx(identity);
    SSL_CTX * cctx = tls_client_ctx();
    const std::string host = a.get("--listen-host", "", "127.0.0.1");
    const int port = a.get_int("--listen-port", "", 50190);
    // in a container the checker listens on 0.0.0.0 but the nodes reach it at the public host
    const std::string bind = a.get("--bind-host", "", host);
    const int every = a.get_int("--every", "", 60);
    const int rounds = a.get_int("--rounds", "", 0); // 0: forever
    std::mt19937 rng(std::random_device{}());
    const httplib::Headers auth = { { "Authorization", "Bearer " + vtok } };
    printf("agent-node verify: model %s, checks every %d s, the nodes reach this checker at %s:%d\n", key.c_str(), every, host.c_str(), port);
    fflush(stdout);
    for (int round = 0; rounds == 0 || round < rounds; round++) {
        if (round > 0) {
            std::this_thread::sleep_for(std::chrono::seconds(every));
        }
        try {
            const http_reply pr = coord_post(url, "/v1/verify/probe", { { "model", key } }, auth);
            if (pr.status == 404) {
                AN_LOG("verify: no node online\n");
                continue;
            }
            if (pr.status != 200) {
                throw std::runtime_error("probe refused: " + pr.error());
            }
            probe_result res;
            try {
                res = run_probe(model.get(), pr.body, cctx, sctx, bind, host, port, identity.fingerprint(), rng);
            } catch (const std::exception & e) {
                // a node that breaks the session is not caught: it may be a network failure; the next probe repeats
                AN_LOG("verify: node %s: %s (no verdict)\n", pr.body["node"].value("addr", "").c_str(), e.what());
                continue;
            }
            const http_reply rr = coord_post(url, "/v1/verify/result",
                { { "session", pr.body["session"] }, { "passed", res.passed }, { "detail", res.detail } }, auth);
            printf("agent-node verify: node %s %s: %s%s\n", pr.body["node"].value("id", "").c_str(), res.passed ? "passed" : "FAILED",
                res.detail.c_str(), rr.status == 204 ? "" : (" (the coordinator: " + rr.error() + ")").c_str());
            fflush(stdout);
        } catch (const std::exception & e) {
            AN_LOG("warning: verify: %s\n", e.what());
        }
    }
    SSL_CTX_free(sctx);
    SSL_CTX_free(cctx);
    return 0;
#endif
}

static int cmd_reference(const cli_args & a) {
    const gen_opts g = gen_opts_from(a);
    model_opts mo = model_opts_from(a);
    mo.layer_start = 0;
    mo.layer_end   = 0;
    llama_model_ptr model = load_model(mo);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    local_eval ev(model.get(), mo, g.n_ctx, a.get_int("-t", "--threads", common_cpu_get_num_math()), g.n_top);
    const gen_result r = generate(ev, vocab, make_prompt(vocab, model.get(), g), g, nullptr);
    print_speed(r);
    return 0;
}

//
// compare two dumps: the chain against the reference
//

static int cmd_compare(const cli_args & a) {
    if (a.positional.size() != 2) {
        throw std::runtime_error("usage: agent-node compare chain.txt reference.txt [--tokens 128] [--tol 0.03] [--top1 0.95] [--same-tokens]");
    }
    const auto x = read_dump(a.positional[0]);
    const auto y = read_dump(a.positional[1]);
    const size_t want      = (size_t) a.get_int("--tokens", "", 128);
    const double tol       = std::stod(a.get("--tol", "", "0.03"));
    const double top1_min  = std::stod(a.get("--top1", "", "0.95"));
    const bool   same_toks = a.has("--same-tokens"); // free generation: require identical greedy tokens

    const size_t n = std::min({ x.size(), y.size(), want });
    size_t n_same = 0;
    while (n_same < n && x[n_same].token == y[n_same].token) {
        n_same++;
    }

    // the logits are comparable while the contexts are equal: up to and including the first different token
    // (with --force on the chain side the contexts are equal at every step)
    double max_diff = 0, max_abs = 0, sum_diff = 0;
    size_t n_diff = 0, n_top1 = 0;
    const size_t n_cmp = std::min(n, n_same + 1);
    for (size_t i = 0; i < n_cmp; i++) {
        n_top1 += !x[i].top.empty() && !y[i].top.empty() && x[i].top[0].id == y[i].top[0].id;
        for (const auto & ey : y[i].top) {
            max_abs = std::max(max_abs, (double) std::fabs(ey.logit));
            for (const auto & ex : x[i].top) {
                if (ex.id == ey.id) {
                    const double d = std::fabs(ex.logit - ey.logit);
                    max_diff = std::max(max_diff, d);
                    sum_diff += d;
                    n_diff++;
                }
            }
        }
    }
    const double rel  = max_abs > 0 ? max_diff / max_abs : 0;
    const double top1 = n_cmp > 0 ? (double) n_top1 / n_cmp : 0;

    const bool tokens_ok = !same_toks || (n_same == n && n == want);
    const bool cover_ok  = n_cmp == want; // every step was compared on the same context
    const bool logits_ok = rel <= tol && top1 >= top1_min;

    printf("tokens: %zu of %zu equal (want %zu)%s\n", n_same, n, want,
        n_same < n ? (" - first difference at token " + std::to_string(n_same)).c_str() : "");
    printf("steps compared on the same context: %zu, same top-1 token: %zu (%.1f%%, need %.0f%%)\n",
        n_cmp, n_top1, 100.0 * top1, 100.0 * top1_min);
    printf("logits: max |diff| = %.4f, mean |diff| = %.4f, max |logit| = %.2f, relative = %.2e (tolerance %.0e)\n",
        max_diff, n_diff ? sum_diff / n_diff : 0.0, max_abs, rel, tol);
    if (!cover_ok && !same_toks) {
        printf("note: the contexts diverged before token %zu, run the chain with --force REFERENCE_DUMP to compare every step\n", want);
    }
    const bool ok = tokens_ok && cover_ok && logits_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// a new secret key of a node (hex); the coordinator keeps it to issue access tokens
static int cmd_keygen(const cli_args &) {
    require_tls("keygen");
#ifdef AGENT_HAVE_TLS
    printf("%s\n", random_key_hex().c_str());
#endif
    return 0;
}

// the TLS identity of a node: created if the file does not exist; prints the certificate fingerprint
static int cmd_identity(const cli_args & a) {
    require_tls("identity");
#ifdef AGENT_HAVE_TLS
    const std::string path = a.get("--identity", "", "");
    if (path.empty()) {
        throw std::runtime_error("--identity FILE is required");
    }
    tls_identity id;
    id.load_or_create(path);
    printf("%s\n", id.fingerprint().c_str());
#else
    (void) a;
#endif
    return 0;
}

// an access token for a node and a session, as the coordinator issues it
static int cmd_token(const cli_args & a) {
    require_tls("token");
#ifdef AGENT_HAVE_TLS
    const std::string key = a.get("--key", "", "");
    const std::string ses = a.get("--session", "", "");
    if (key.empty() || ses.empty()) {
        throw std::runtime_error("--key HEX and --session HEX are required");
    }
    printf("%s\n", token_issue(key, std::stoull(ses, nullptr, 16), (uint64_t) a.get_int("--ttl", "", 3600)).c_str());
#else
    (void) a;
#endif
    return 0;
}

// prints the shape of the model as one JSON line, for the scripts that split it into layer ranges
// reads only the GGUF metadata, no weights
// identity of the model file for the coordinator, the same for a node and for the client (which loads only the vocabulary,
// so llama_model_desc is not known there): name and shape from the GGUF metadata plus a hash of the tensor names, types
// and sizes, which differs between quantizations; reads only the metadata
static std::string model_key(const std::string & path) {
    gguf_ptr g = read_local_header(path);
    if (!g) {
        throw std::runtime_error("cannot read the GGUF metadata of " + path);
    }
    return model_key_of({ g.get() });
}

// bench-batch: what a batch of several sessions costs on this device against one session. One decode step of the
// layers of the node (--layers, as the node holds them) for 1, 2, 4, 8 sequences at once, each after -c tokens
// of context. If a step of 4 sequences costs about as much as a step of one, nodes should batch the sessions of
// several users (docs/Параллельные сессии.md). The inputs are random: the tokens of different users route to
// different experts of an MoE model, as random hidden states do. --check also decodes each sequence alone and
// compares the outputs: a batch must give what the sequences give one by one.
static int cmd_bench_batch(const cli_args & a) {
    const model_opts mo = model_opts_from(a);
    std::vector<int> seqs;
    {
        std::stringstream ss(a.get("--seqs", "", "1,2,4,8"));
        for (std::string x; std::getline(ss, x, ',');) {
            if (!x.empty()) {
                seqs.push_back(std::max(1, std::stoi(x)));
            }
        }
    }
    if (seqs.empty()) {
        throw std::runtime_error("--seqs must list the batch sizes, e.g. 1,2,4,8");
    }
    const int n_hist    = a.get_int("-c", "--ctx-size", 512);  // context of each sequence before the measured steps
    const int n_steps   = a.get_int("-n", "--n-predict", 32);
    const int n_threads = a.get_int("-t", "--threads", common_cpu_get_num_math());
    const int max_seq   = *std::max_element(seqs.begin(), seqs.end());
    const bool check    = a.has("--check");

    llama_model_ptr model = load_model(mo);
    const int  n_layer = llama_model_n_layer(model.get());
    const int  n_embd  = llama_model_n_embd(model.get());
    const int  n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const int  first   = std::clamp(mo.layer_start, 0, n_layer);
    const int  last    = mo.layer_end <= 0 ? n_layer : std::min(mo.layer_end, n_layer);
    const bool is_first = first == 0, is_last = last == n_layer;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) (max_seq * (n_hist + n_steps + 8));
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = (uint32_t) max_seq;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf         = true;
    // to compare with --gpu-experts: a cache on the GPU for the experts in RAM (llama.cpp, experimental)
    cp.moe_cache_size  = (size_t) a.get_int("--moe-cache-mib", "", 0) << 20;
    if (!is_last) {
        cp.embeddings   = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    if (!ctx) {
        throw std::runtime_error("failed to create the context");
    }

    std::mt19937 rng(1);
    std::normal_distribution<float> val(0.0f, 1.0f);
    std::vector<float> embd((size_t) 512 * n_embd);
    for (float & x : embd) {
        x = val(rng);
    }
    common_batch batch(ctx.get());
    // the input of user u at position p: the same whatever the batch, so that --check can decode it alone
    // (into sequence s) and get the same output
    auto add = [&](int u, llama_pos p, llama_seq_id s, bool out) {
        const uint32_t h = (uint32_t) u * 2654435761u ^ (uint32_t) p * 40503u;
        if (is_first) {
            batch.add((llama_token) (h % (uint32_t) n_vocab), p, s, out);
        } else {
            const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
            batch.add_embd({ embd.data() + (size_t) (h % 512) * n_embd, 1, (size_t) n_embd }, pos, s, out);
        }
    };
    auto process = [&]() {
        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("llama_process failed");
        }
        llama_synchronize(ctx.get());
    };

    // the history of users 0..n-1, user u into sequence u (or all into sequence seq when alone)
    auto prefill = [&](int u0, int n, bool alone) {
        for (int u = u0; u < u0 + n; u++) {
            for (int p0 = 0; p0 < n_hist; p0 += 512) {
                batch.clear();
                const int m = std::min(512, n_hist - p0);
                for (int i = 0; i < m; i++) {
                    add(u, p0 + i, alone ? 0 : u, i + 1 == m);
                }
                process();
            }
        }
    };
    // the output of batch entry i: the hidden state, or the logits at the last layer
    const int n_out = is_last ? n_vocab : n_embd;
    auto output = [&](int i) {
        const float * o = is_last ? llama_get_logits_ith(ctx.get(), i) : llama_get_embeddings_ith(ctx.get(), i);
        if (o == nullptr) {
            throw std::runtime_error("no output for batch entry " + std::to_string(i));
        }
        return std::vector<float>(o, o + n_out);
    };

    AN_LOG("bench-batch: layers [%d, %d) of %d, %d tokens of context per sequence, %d steps\n", first, last, n_layer, n_hist, n_steps);
    json rows = json::array();
    double ms1 = 0;
    printf("%6s %12s %14s %10s\n", "seqs", "ms/step", "tokens/s", "vs 1");
    for (int n : seqs) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        prefill(0, n, false);
        double total = 0;
        for (int t = -2; t < n_steps; t++) { // two steps to warm up
            batch.clear();
            for (int s = 0; s < n; s++) {
                add(s, n_hist + 2 + t, s, true);
            }
            const double t0 = now_ms();
            process();
            if (t >= 0) {
                total += now_ms() - t0;
            }
        }
        const double ms = total / n_steps;
        if (n == seqs[0]) {
            ms1 = ms / n; // per sequence of the first size, normally 1
        }
        json row = { { "seqs", n }, { "ms_per_step", ms }, { "tok_s", n * 1000.0 / ms } };
        std::string verdict;
        if (check && n > 1) {
            // the outputs of the last measured step, then each user alone with the same history and input
            std::vector<std::vector<float>> got;
            for (int u = 0; u < n; u++) {
                got.push_back(output(u));
            }
            double max_diff = 0, max_ref = 0;
            int top_same = 0;
            for (int u = 0; u < n; u++) {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                prefill(u, 1, true);
                for (int t = -2; t < n_steps; t++) {
                    batch.clear();
                    add(u, n_hist + 2 + t, 0, true);
                    process();
                }
                const std::vector<float> ref = output(0);
                for (int k = 0; k < n_out; k++) {
                    max_diff = std::max(max_diff, (double) std::fabs(got[u][k] - ref[k]));
                    max_ref  = std::max(max_ref, (double) std::fabs(ref[k]));
                }
                top_same += std::max_element(got[u].begin(), got[u].end()) - got[u].begin() ==
                            std::max_element(ref.begin(), ref.end()) - ref.begin();
            }
            row["max_diff"] = max_diff;
            row["max_ref"]  = max_ref;
            char buf[128];
            snprintf(buf, sizeof(buf), "  check: max |diff| %.4g of max |value| %.4g%s", max_diff, max_ref,
                is_last ? (", top-1 same " + std::to_string(top_same) + "/" + std::to_string(n)).c_str() : "");
            verdict = buf;
        }
        printf("%6d %12.2f %14.1f %9.2fx%s\n", n, ms, n * 1000.0 / ms, n * ms1 / ms, verdict.c_str());
        rows.push_back(row);
    }
    const std::string out = a.get("--json", "", "");
    if (!out.empty()) {
        std::ofstream(out) << json({ { "model", model_desc(model.get()) }, { "layer_start", first }, { "layer_end", last }, { "n_layer", n_layer },
                                     { "ctx_per_seq", n_hist }, { "steps", n_steps }, { "cpu_moe", mo.cpu_moe }, { "rows", rows } })
                                   .dump(1)
                               << "\n";
    }
    return 0;
}

static int cmd_info(const cli_args & a) {
    const std::string path = a.get("-m", "--model", "");
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        throw std::runtime_error("cannot read the GGUF metadata of " + path);
    }
    auto get_str = [&](const std::string & key) -> std::string {
        const int64_t id = gguf_find_key(g, key.c_str());
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "";
    };
    auto get_u32 = [&](const std::string & key) -> long long {
        const int64_t id = gguf_find_key(g, key.c_str());
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_UINT32 ? (long long) gguf_get_val_u32(g, id) : -1;
    };
    const std::string arch = get_str("general.architecture");
    printf("{\"arch\":\"%s\",\"n_layer\":%lld,\"n_embd\":%lld,\"full_attention_interval\":%lld,\"model_key\":%s}\n", arch.c_str(),
        get_u32(arch + ".block_count"), get_u32(arch + ".embedding_length"), get_u32(arch + ".full_attention_interval"),
        json(model_key(path)).dump().c_str());
    gguf_free(g);
    return 0;
}

// slice: writes the slice of a model with a layer range (or only the metadata, for the vocabulary of a client),
// from a URL (HTTP ranges, only the bytes of the slice are downloaded) or a local file
static int cmd_slice(const cli_args & a) {
    const std::string from = a.get("--from", "", "");
    const std::string out  = a.get("--out", "", "");
    if (from.empty() || out.empty()) {
        throw std::runtime_error("slice needs --from URL|PATH and --out FILE");
    }
    model_source ms;
    open_model_source(ms, from, a.get("--mirror", "", ""));
    int32_t first = 0, last = 0;
    if (!a.has("--vocab-only")) {
        const std::string layers = a.get("--layers", "", "");
        const size_t p = layers.find(':');
        if (p == std::string::npos) {
            throw std::runtime_error("slice needs --layers START:END or --vocab-only");
        }
        first = std::stoi(layers.substr(0, p));
        last  = p + 1 < layers.size() ? std::stoi(layers.substr(p + 1)) : ms.n_layer;
    }
    write_slice(ms, first, last, out, download_printer());
    printf("{\"model_key\":%s,\"n_layer\":%d,\"layer_start\":%d,\"layer_end\":%d,\"bytes\":%llu}\n", json(ms.key).dump().c_str(),
        ms.n_layer, first, last, (unsigned long long) std::filesystem::file_size(slice_path(out)));
    return 0;
}

// devices: the compute devices ggml sees, as JSON, so that scripts can pick the right one
// (for example the discrete card and not the integrated one when both are Vulkan devices)
// a check of the direct transport without a model (CI, a quick look at a network):
//   agent-node p2p-echo serve  --rendezvous HOST:PORT --secret S   echoes every frame back
//   agent-node p2p-echo client --rendezvous HOST:PORT --secret S [--bytes N]
// the secret is the RELAY_SECRET of the rendezvous server (the coordinator hands out tickets made with it)
static int cmd_p2p_echo(const cli_args & a, const std::string & mode) {
#ifndef AGENT_HAVE_TLS
    require_tls("p2p-echo");
    return 1;
#else
    const std::string rv = a.get("--rendezvous", "", "127.0.0.1:4001");
    const std::string secret = a.get("--secret", "", "");
    const std::string id = std::string(64, 'e');
    auto & ep = p2p_endpoint::get();
    if (!p2p_dial) {
        throw std::runtime_error("p2p is off");
    }
    auto read_chunk = [](tcp_conn & c, std::vector<uint8_t> & buf) {
        uint32_t n = 0;
        if (!c.recv_all(&n, 4) || n > (1u << 20)) {
            return false;
        }
        buf.resize(n);
        return c.recv_all(buf.data(), n);
    };
    auto send_chunk = [](tcp_conn & c, const uint8_t * p, uint32_t n) {
        std::lock_guard<std::mutex> lock(c.send_mutex);
        return c.send_all(&n, 4) && c.send_all(p, n);
    };
    if (mode == "serve") {
        const long long exp = (long long) unix_now() + 3600;
        const std::string ticket = hmac_hex(std::vector<uint8_t>(secret.begin(), secret.end()),
                                            "agent-relay/1|" + id + "|" + std::to_string(exp));
        ep.listen(rv, id, exp, ticket, [&](tcp_conn_ptr c) {
            std::vector<uint8_t> buf;
            while (read_chunk(*c, buf) && send_chunk(*c, buf.data(), (uint32_t) buf.size())) {
            }
        });
        printf("p2p-echo: serving as %s through %s\n", id.substr(0, 16).c_str(), rv.c_str());
        fflush(stdout);
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    const size_t total = (size_t) std::stoull(a.get("--bytes", "", "8000000"));
    std::string err;
    const double t0 = now_ms();
    tcp_conn_ptr c = ep.connect(rv, id, err);
    if (!c) {
        printf("p2p-echo: no direct connection: %s\n", err.c_str());
        return 1;
    }
    printf("p2p-echo: connected in %.0f ms via %s\n", now_ms() - t0, ep.last_path.c_str());
    std::vector<uint8_t> small(64), back;
    const double r0 = now_ms();
    for (int i = 0; i < 200; i++) {
        small[0] = (uint8_t) i;
        if (!send_chunk(*c, small.data(), 64) || !read_chunk(*c, back) || back != small) {
            printf("p2p-echo: round trip %d failed\n", i);
            return 1;
        }
    }
    printf("p2p-echo: round trip of 64 bytes %.2f ms\n", (now_ms() - r0) / 200);
    std::vector<uint8_t> data(total);
    std::mt19937 rng(7);
    for (auto & b : data) {
        b = (uint8_t) rng();
    }
    const double b0 = now_ms();
    const size_t chunk = 60000;
    std::thread w([&]() {
        for (size_t off = 0; off < total; off += chunk) {
            if (!send_chunk(*c, data.data() + off, (uint32_t) std::min(chunk, total - off))) {
                return;
            }
        }
    });
    c->recv_timeout_ms = 30000;
    size_t got = 0;
    bool ok = true;
    while (ok && got < total) {
        ok = read_chunk(*c, back) && memcmp(back.data(), data.data() + got, back.size()) == 0;
        got += back.size();
    }
    w.join();
    const double s = (now_ms() - b0) / 1000;
    printf("p2p-echo: %zu bytes there and back %s in %.2f s (%.1f Mbit/s each way)\n", total, ok ? "intact" : "BROKEN", s,
        total * 8 / s / 1e6);
    c->close();
    return ok ? 0 : 1;
#endif
}

static int cmd_devices() {
    json out = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        const char * type = "other";
        switch (props.type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   type = "cpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type = "gpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  type = "igpu";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "accel"; break;
            default: break;
        }
        out.push_back({
            { "name", props.name ? props.name : "" },
            { "description", props.description ? props.description : "" },
            { "type", type },
            { "memory_total", (uint64_t) props.memory_total },
            { "memory_free", (uint64_t) props.memory_free },
        });
    }
    printf("%s\n", out.dump().c_str());
    return 0;
}

static void watch_parent(unsigned long pid) {
#ifdef _WIN32
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD) pid);
    if (h == nullptr) {
        exit(3); // the app is already gone
    }
    std::thread([h]() {
        WaitForSingleObject(h, INFINITE);
        _exit(0);
    }).detach();
#else
    std::thread([pid]() {
        while (getppid() == (pid_t) pid) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        _exit(0);
    }).detach();
#endif
}

static void usage() {
    fprintf(stderr,
        "usage:\n"
        "  agent-node node      -m model.gguf --layers START:END [--host 127.0.0.1] [--port 50152] [-ngl 999] [--device CUDA0] [-t N]\n"
        "  agent-node client    -m model.gguf --chain HOST:PORT,HOST:PORT,... [-p TEXT | -f FILE] [-n 64] [-c 4096]\n"
        "                       [--top 8] [--ignore-eos] [--no-chat] [--dump FILE] [--json FILE]\n"
        "                       [--hidden q8|f16|f32] [--force REFERENCE_DUMP]\n"
        "                       [--draft DRAFT.gguf [--draft-max 6] [-ngld 999]]   speculative decoding\n"
        "  agent-node reference -m model.gguf [-p TEXT | -f FILE] [-n 64] [-c 4096] [-ngl 999] [--device CUDA0] [--dump FILE]\n"
        "  agent-node serve     -m model.gguf --coordinator URL [--port 8090] [--draft DRAFT.gguf] [-c 8192] [--account FILE]\n"
        "                       [--keep-chain-s 180]  the local API of the agent (OpenAI-style /v1/chat/completions, reasoning in\n"
        "                       reasoning_content); the chain stays open between requests, the next one reads only the new text\n"
        "  agent-node verify    -m whole-model.gguf --coordinator URL --verifier-token T [--every 60] [--listen-host H --listen-port 50190 --bind-host 0.0.0.0]\n"
        "                       the checker: runs probes through random nodes and reports the verdicts (needs VERIFIER_TOKEN)\n"
        "  agent-node compare   CHAIN_DUMP REFERENCE_DUMP [--tokens 128] [--tol 0.03] [--top1 0.95] [--same-tokens]\n"
        "  agent-node info      -m model.gguf\n"
        "  agent-node bench-batch -m model.gguf [--layers START:END] [--cpu-moe [--gpu-experts N | --moe-cache-mib M]] [--seqs 1,2,4,8]\n"
        "                       [-c 512] [-n 32] [--check] [--json F]\n"
        "                       one decode step of the layers for several sequences at once: what batching sessions gives\n"
        "  agent-node slice     --from URL|model.gguf (--layers START:END | --vocab-only) --out slice.gguf [--mirror URL]\n"
        "                       only the weights of the range (downloaded by HTTP ranges), with the metadata of the model\n"
        "  agent-node devices                                  the compute devices (JSON: name, type, memory)\n"
        "  agent-node keygen                                   a secret key of a node (for access tokens)\n"
        "  agent-node identity  --identity node.pem            create/read the TLS identity, print its fingerprint\n"
        "  agent-node token     --key HEX --session HEX [--ttl 3600]\n"
        "\n"
        "security: node [--key HEX | env AGENT_NODE_KEY] [--tls --identity node.pem]\n"
        "          client [--tls] [--keys K,K,.. | --tokens T,T,.. --session HEX] [--pins FP,FP,..]\n"
        "failover: client --spare HOST:PORT,.. [--spare-keys ..] [--spare-pins ..] [--timeout 3000]\n"
        "sessions: node [--max-sessions 4] [--seq-ctx 16384] [--seq-rollback 6] [--vram-margin-mb 256] [--batch-max N]\n"
        "               one context for the sessions, fewer if the GPU memory runs short; the forwards waiting at the moment\n"
        "               run in one pass (--batch-max 1: one at a time)\n"
        "gpu:      node [--gpu-clocks auto|off]   hold the NVIDIA GPU clocks while there are sessions (needs administrator)\n"
        "               [--keep-warm 0] [--keep-warm-mb 24]   or keep the GPU awake with small jobs between tokens (ms)\n"
        "coordinator: node --coordinator URL [--public-addr HOST:PORT] [--city NAME]   (needs --tls and --key)\n"
        "                  [--relay off|auto|only]  behind NAT: reachable through the relay of the network\n"
        "                  [--p2p on|off]  behind NAT: direct UDP connections through the rendezvous server first (on)\n"
        "      node|verify [--cpu-moe [--ram-gb N] [--moe-cache-mib M | --gpu-experts N]]  the experts of an MoE model in RAM,\n"
        "                  the rest on the GPU; a cache of M MB of them on the GPU (with --model-url: what the free memory holds),\n"
        "                  or the experts of the last N layers on the GPU\n"
        "             node --model-url URL [--models-dir models] [--disk-gb 25] [--vram-gb N | --vram-reserve-mb 1024] [--mirror URL]\n"
        "                  instead of -m and --layers: the coordinator assigns the layers, the node downloads only them\n"
        "             client --coordinator URL [--spares 1] [--city NAME]              (instead of --chain)\n"
        "\n"
        "the client needs the same model file only for its tokenizer, it loads no weights\n");
}

#ifdef _WIN32
// argv comes in the ANSI code page: take the command line as UTF-16 and convert it to UTF-8
// so that non-ASCII paths and prompts survive, as common_params_parse does
static std::vector<std::string> utf8_args() {
    std::vector<std::string> out;
    int wargc = 0;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr) {
        return out;
    }
    for (int i = 0; i < wargc; i++) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string a(n > 0 ? n - 1 : 0, '\0');
        if (n > 0) {
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, a.data(), n, nullptr, nullptr);
        }
        out.push_back(std::move(a));
    }
    LocalFree(wargv);
    return out;
}
#endif

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    std::vector<std::string> args_utf8 = utf8_args();
    std::vector<char *>      argv_utf8;
    if ((int) args_utf8.size() == argc) {
        for (auto & s : args_utf8) {
            argv_utf8.push_back(s.data());
        }
        argv_utf8.push_back(nullptr);
        argv = argv_utf8.data();
    }
#endif
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    try {
        cli_args a(argc, argv, 2);
        // started by the app: exit together with it, even when it crashes (a left-over node or API would hold the port)
        if (a.has("--parent-pid")) {
            watch_parent(std::stoul(a.get("--parent-pid", "", "0")));
        }
        if (cmd == "compare") {
            return cmd_compare(a);
        }
        if (cmd == "keygen") {
            return cmd_keygen(a);
        }
        if (cmd == "identity") {
            return cmd_identity(a);
        }
        if (cmd == "token") {
            return cmd_token(a);
        }
        if (cmd == "info") {
            return cmd_info(a);
        }
        if (cmd == "slice") {
            return cmd_slice(a);
        }

        llama_log_set(log_callback, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        net_init();
        // direct UDP connections to nodes behind NAT, the TCP relay stays the fallback (--p2p off: relay only)
        if ((cmd == "node" || cmd == "client" || cmd == "serve" || cmd == "verify" || cmd == "p2p-echo") && a.get("--p2p", "", "on") != "off") {
            p2p_init();
        }

        if (cmd == "p2p-echo") {
            return cmd_p2p_echo(a, argc > 2 ? argv[2] : "");
        }
        if (cmd == "devices") {
            return cmd_devices();
        }
        if (cmd == "node") {
            return cmd_node(a);
        }
        if (cmd == "client") {
            return cmd_client(a);
        }
        if (cmd == "reference") {
            return cmd_reference(a);
        }
        if (cmd == "serve") {
            return cmd_serve(a);
        }
        if (cmd == "verify") {
            return cmd_verify(a);
        }
        if (cmd == "bench-batch") {
            return cmd_bench_batch(a);
        }
        usage();
        return 2;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
