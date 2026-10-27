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
               name == "--vocab-only";
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
            "--account", "--parent-pid", "--chat-template", "--from", "--out", "--vocab-only", "--model-url", "--models-dir", "--disk-gb", "--vram-gb", "--vram-reserve-mb", "--mirror",
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
};

static llama_model_ptr load_model(const model_opts & mo) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = mo.n_gpu_layers;
    mp.layer_start  = mo.layer_start;
    mp.layer_end    = mo.layer_end;
    mp.vocab_only   = mo.vocab_only;

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

struct model_part {
    const llama_model * model = nullptr;
    llama_context_ptr   ctx;
    int32_t first = 0, last = 0, n_layer = 0, n_embd = 0, n_vocab = 0;
    int32_t n_batch = 512;
    int32_t n_past  = 0; // tokens in the memory: the next forward continues at this position
    double  last_output_ms = 0; // time of reading the outputs in the last run()

    std::vector<float> hidden_in;

    // n_rollback: how many trailing tokens of a forward can be rolled back (speculative decoding)
    model_part(const llama_model * model, const model_opts & mo, uint32_t n_ctx, int32_t n_threads, uint32_t n_rollback = 0) : model(model) {
        n_layer = llama_model_n_layer(model);
        n_embd  = llama_model_n_embd(model);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        first   = std::clamp(mo.layer_start, 0, n_layer);
        last    = mo.layer_end <= 0 ? n_layer : std::min(mo.layer_end, n_layer);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = n_batch;
        cp.n_ubatch        = n_batch;
        cp.n_seq_max       = 1;
        // recurrent layers (Gated DeltaNet) keep snapshots of their state to undo rejected draft tokens
        cp.n_rs_seq        = n_rollback;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.no_perf         = true;
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

    // runs the tokens or the hidden states of one forward frame
    // not the last part: hidden_out gets n_tokens rows of n_embd, the last part: top gets n_out rows of n_top
    void run(const forward_hdr & h, const uint8_t * data,
            std::vector<float> & hidden_out, std::vector<top_entry> & top, uint32_t n_top, uint32_t & n_out) {
        if (h.n_tokens == 0 || (int32_t) h.n_tokens > n_batch) {
            throw std::runtime_error("forward must carry 1.." + std::to_string(n_batch) + " tokens");
        }
        const bool out_all = h.flags & FORWARD_OUT_ALL;

        // a forward that starts before the end of the previous one drops the rejected draft tokens
        if (h.pos0 > n_past) {
            throw std::runtime_error("forward at position " + std::to_string(h.pos0) + " leaves a gap after " + std::to_string(n_past));
        }
        if (h.pos0 < n_past) {
            if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, h.pos0, -1)) {
                throw std::runtime_error("cannot roll back from " + std::to_string(n_past) + " to " + std::to_string(h.pos0) +
                    " (at most n_rollback tokens of the last forward)");
            }
            n_past = h.pos0;
        }

        common_batch batch(ctx.get());
        if (h.kind == payload::tokens) {
            if (!is_first()) {
                throw std::runtime_error("this node runs layers from " + std::to_string(first) + " and takes hidden states, not tokens");
            }
            const int32_t * ids = (const int32_t *) data;
            for (uint32_t i = 0; i < h.n_tokens; i++) {
                const bool out = !is_last() || out_all || i + 1 == h.n_tokens;
                batch.add(ids[i], h.pos0 + (int32_t) i, 0, out);
            }
        } else if (h.kind == payload::f16 || h.kind == payload::f32 || h.kind == payload::q8) {
            if (is_first()) {
                throw std::runtime_error("the first node takes tokens, not hidden states");
            }
            hidden_in.resize((size_t) h.n_tokens * n_embd);
            if (h.kind == payload::f16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, hidden_in.data(), (int64_t) hidden_in.size());
            } else if (h.kind == payload::q8) {
                ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(data, hidden_in.data(), (int64_t) hidden_in.size());
            } else {
                memcpy(hidden_in.data(), data, hidden_in.size() * sizeof(float));
            }
            for (uint32_t i = 0; i < h.n_tokens; i++) {
                const llama_pos p = h.pos0 + (llama_pos) i;
                // text positions, for M-RoPE they expand to [p, p, p, 0]
                const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                const bool out = !is_last() || out_all || i + 1 == h.n_tokens;
                batch.add_embd({ hidden_in.data() + (size_t) i * n_embd, 1, (size_t) n_embd }, pos, 0, out);
            }
        } else {
            throw std::runtime_error("unknown payload kind");
        }

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("llama_process failed at position " + std::to_string(h.pos0));
        }
        n_past = h.pos0 + (int32_t) h.n_tokens;
        const double t_out = now_ms();
        struct out_timer {
            double t0; double & dst;
            ~out_timer() { dst = now_ms() - t0; }
        } timer { t_out, last_output_ms };

        if (!is_last()) {
            hidden_out.resize((size_t) h.n_tokens * n_embd);
            for (uint32_t i = 0; i < h.n_tokens; i++) {
                const float * e = llama_get_embeddings_ith(ctx.get(), (int32_t) i);
                if (e == nullptr) {
                    throw std::runtime_error("no hidden state for token " + std::to_string(i));
                }
                memcpy(hidden_out.data() + (size_t) i * n_embd, e, (size_t) n_embd * sizeof(float));
            }
            n_out = 0;
            return;
        }

        n_out = out_all ? h.n_tokens : 1;
        top.resize((size_t) n_out * n_top);
        for (uint32_t o = 0; o < n_out; o++) {
            const int32_t i = out_all ? (int32_t) o : (int32_t) h.n_tokens - 1;
            const float * logits = llama_get_logits_ith(ctx.get(), i);
            if (logits == nullptr) {
                throw std::runtime_error("no logits for token " + std::to_string(i));
            }
            top_k(logits, n_vocab, n_top, top.data() + (size_t) o * n_top);
        }
    }
};

//
// node: serves one part of the model to clients and to the previous node
//

struct session {
    uint64_t                    id = 0;
    std::unique_ptr<model_part> part;
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
        s->part  = std::make_unique<model_part>(model.get(), mo, m.n_ctx, n_threads, std::min<uint32_t>(m.n_rollback, 32));

        if (s->part->is_last() != m.next.empty()) {
            throw std::runtime_error(s->part->is_last() ? "this node runs the last layers, it has no next node"
                                                        : "this node does not run the last layers, it needs a next node");
        }
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
        ok.layer_start = s->part->first;
        ok.layer_end   = s->part->last;
        ok.n_layer     = s->part->n_layer;
        ok.n_embd      = s->part->n_embd;
        ok.n_vocab     = s->part->n_vocab;
        ok.desc        = model_desc(model.get());
        {
            std::lock_guard<std::mutex> lock(mtx);
            sessions[m.session] = s;
        }
        authed.insert(m.session);
        AN_LOG("session %" PRIu64 ": layers [%d, %d), n_ctx = %u, next = %s\n",
            m.session, ok.layer_start, ok.layer_end, m.n_ctx, m.next.empty() ? "(client)" : m.next.c_str());
        send_frame(*conn, msg::open_ok, ok.write());
    }

    void on_forward(const std::vector<uint8_t> & body) {
        reader r(body.data(), body.size());
        forward_hdr h = forward_hdr::read(r);
        auto s = find(h.session);

        std::lock_guard<std::mutex> lock(s->mtx);
        model_part & p = *s->part;
        const size_t need = payload_size(h.kind, h.n_tokens, p.n_embd);
        const uint8_t * data = r.get_bytes(need);

        s->st.bytes_in += 5 + body.size();
        s->st.n_tokens += h.n_tokens;
        s->st.n_forward++;
        count_work(h.session, h.n_tokens);

        static thread_local std::vector<float>     hidden;
        static thread_local std::vector<top_entry> top;
        uint32_t n_out = 0;

        const double t0 = now_ms();
        {
            struct busy_guard {
                node_server & srv;
                busy_guard(node_server & srv) : srv(srv) { srv.busy++; }
                ~busy_guard() { srv.last_end_us = (int64_t) (now_ms() * 1000); srv.busy--; }
            } guard(*this);
            p.run(h, data, hidden, top, s->n_top, n_out);
        }
        s->st.compute_ms += now_ms() - t0;
        s->st.output_ms  += p.last_output_ms;

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
static void prepare_assigned_slice(const cli_args & a, model_opts & mo, const std::string & fingerprint) {
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
}

static std::string model_key(const std::string & path);

// one node of the chain as the client knows it
struct node_ref {
    std::string addr;
    std::string key;   // secret key of the node, to issue tokens itself (tests; normally the coordinator issues them)
    std::string token; // access token issued by the coordinator for the session
    std::string pin;   // expected certificate fingerprint ("" = not checked)
};

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
        const http_reply r = coord_post(url, "/v1/accounts", json::object());
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

static int cmd_node(const cli_args & a) {
    node_server srv;
    const bool assigned = a.has("--model-url");
    if (assigned && (a.has("-m") || a.has("--model") || a.has("--layers"))) {
        throw std::runtime_error("--model-url takes the place of -m and --layers: the coordinator assigns the layers");
    }
    if (assigned) {
        srv.mo.n_gpu_layers = a.get_int("-ngl", "--n-gpu-layers", 999);
        srv.mo.device       = a.get("--device", "-dev", "");
    } else {
        srv.mo = model_opts_from(a);
    }
    srv.n_threads = a.get_int("-t", "--threads", common_cpu_get_num_math());

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
        prepare_assigned_slice(a, srv.mo, fingerprint);
    }

    srv.model = load_model(srv.mo);
    {
        const int32_t n_layer = llama_model_n_layer(srv.model.get());
        const int32_t first   = std::clamp(srv.mo.layer_start, 0, n_layer);
        const int32_t last    = srv.mo.layer_end <= 0 ? n_layer : std::min(srv.mo.layer_end, n_layer);
        AN_LOG("node: %s, layers [%d, %d) of %d, listening on %s:%d\n", model_desc(srv.model.get()).c_str(),
            first, last, n_layer, host.c_str(), port);
    }

    // for the coordinator, which prefers fast nodes; before the ready line, so that a measurement does not meet clients
    double ms_per_layer = 0;
    if (a.has("--coordinator")) {
        ms_per_layer = measure_ms_per_layer(srv.model.get(), srv.mo, srv.n_threads);
        AN_LOG("node: %.3f ms per layer per token\n", ms_per_layer);
    }

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
        const std::string pub = a.get("--public-addr", "", (host == "0.0.0.0" || host == "::" ? "" : host) + ":" + std::to_string(port));
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
        link.work = [&srv]() { return srv.work_json(); };
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
    // runs n tokens from position pos0, returns the top-k of the last token, or of every token with out_all
    virtual std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all = false) = 0;
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

    std::vector<top_entry> eval_once(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all) {
        if ((size_t) pos0 > hist.size()) {
            throw std::runtime_error("forward leaves a gap in the context");
        }
        hist.resize(pos0);
        hist.insert(hist.end(), tokens, tokens + n);

        forward_hdr h;
        h.session  = sid;
        h.step     = ++step;
        h.pos0     = pos0;
        h.n_tokens = n;
        h.kind     = payload::tokens;
        h.flags    = out_all ? FORWARD_OUT_ALL : 0;
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
        if (out_all) {
            return res.top;
        }
        return std::vector<top_entry>(res.top.end() - res.n_top, res.top.end());
    }

    std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all = false) override {
        while (true) {
            try {
                return eval_once(tokens, n, pos0, out_all);
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
        for (size_t i = 0; i < pre.size(); i += 512) {
            const uint32_t n = (uint32_t) std::min<size_t>(512, pre.size() - i);
            eval_once(pre.data() + i, n, (int32_t) i, false);
        }
        const double dt = now_ms() - t0;
        recover_ms    += dt;
        recover_max_ms = std::max(recover_max_ms, dt);
        AN_LOG("chain recovered in %.0f ms, %zu tokens of context read again\n", dt, pre.size());
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

    std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0, bool out_all = false) override {
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
    res.n_prompt = (int32_t) prompt.size();
    std::mt19937 rng(g.seed);

    double t0 = now_ms();
    std::vector<top_entry> top;
    for (size_t i = 0; i < prompt.size(); i += g.n_batch) {
        const uint32_t n = (uint32_t) std::min<size_t>(g.n_batch, prompt.size() - i);
        top = ev.eval(prompt.data() + i, n, (int32_t) i);
    }
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
    res.n_prompt = (int32_t) prompt.size();
    dm.reset();

    double t0 = now_ms();
    std::vector<top_entry> top;
    for (size_t i = 0; i < prompt.size(); i += g.n_batch) {
        const uint32_t n = (uint32_t) std::min<size_t>(g.n_batch, prompt.size() - i);
        top = ev.eval(prompt.data() + i, n, (int32_t) i);
    }
    res.prefill_ms = now_ms() - t0;

    // the draft model reads the prompt too, except its last token: every round starts by decoding the last token
    t0 = now_ms();
    for (size_t i = 0; i + 1 < prompt.size(); i += g.n_batch) {
        const uint32_t n = (uint32_t) std::min<size_t>(g.n_batch, prompt.size() - 1 - i);
        dm.decode(prompt.data() + i, n, (int32_t) i);
    }
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
            bool first = true;
            for (const auto & part : cs->prompt().parts) {
                // text of the user is not parsed for special tokens: a message cannot forge a turn of the template
                const auto t = common_tokenize(st->vocab, part.text, first, !part.is_input);
                prompt.insert(prompt.end(), t.begin(), t.end());
                first = false;
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
        auto run = [st, hidden, prompt, max_tokens, temperature, top_p, seed, cs](
                       const std::function<bool(const json &)> & out, json & usage, std::string & finish, common_chat_msg & final_msg) {
            std::lock_guard<std::mutex> lock(st->mtx);
            const coord_chain c = request_chain(st->acc, st->key, st->spares, 3600, st->city);
            chain_opts co;
            co.nodes      = c.nodes;
            co.spares     = c.spares;
            co.session    = c.session;
            co.tls        = true;
            co.n_ctx      = st->n_ctx;
            co.n_top      = 16; // the candidates for sampling
            co.hidden     = hidden;
            co.n_rollback = st->dm ? (uint32_t) st->n_draft : 0;
            chain_eval chain(co);
            if (chain.parts[0].n_vocab != llama_vocab_n_tokens(st->vocab)) {
                throw std::runtime_error("the tokenizer does not match the model of the chain");
            }
            gen_opts g;
            g.n_gen       = max_tokens;
            g.n_ctx       = st->n_ctx;
            g.n_top       = co.n_top;
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
            long long charged = 0;
            const long long balance = report_session(st->acc, c.session_hex, r.n_prompt, (int32_t) r.tokens.size(), &charged);
            finish = !final_msg.tool_calls.empty() ? "tool_calls" : ended ? "stop" : "length";
            usage = { { "prompt_tokens", r.n_prompt }, { "completion_tokens", (int) r.tokens.size() },
                      { "total_tokens", r.n_prompt + (int) r.tokens.size() } };
            usage["agent"] = { { "session", c.session_hex }, { "points", charged }, { "balance", balance },
                               { "tok_s", r.n_gen_eval > 0 ? r.n_gen_eval * 1000.0 / r.gen_ms : 0.0 }, { "nodes", c.nodes.size() } };
            AN_LOG("serve: %d + %zu tokens, %.2f tok/s, %zu tool calls, %lld points (balance %lld)\n", r.n_prompt, r.tokens.size(),
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
        "                       the local API of the agent (OpenAI-style /v1/chat/completions, reasoning in reasoning_content)\n"
        "  agent-node compare   CHAIN_DUMP REFERENCE_DUMP [--tokens 128] [--tol 0.03] [--top1 0.95] [--same-tokens]\n"
        "  agent-node info      -m model.gguf\n"
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
        "gpu:      node [--gpu-clocks auto|off]   hold the NVIDIA GPU clocks while there are sessions (needs administrator)\n"
        "               [--keep-warm 0] [--keep-warm-mb 24]   or keep the GPU awake with small jobs between tokens (ms)\n"
        "coordinator: node --coordinator URL [--public-addr HOST:PORT] [--city NAME]   (needs --tls and --key)\n"
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
        usage();
        return 2;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
