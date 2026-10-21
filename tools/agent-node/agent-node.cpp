// agent-node: one part of a model in a chain of nodes, plus the client that drives the chain
//
//   agent-node node      -m model.gguf --layers 0:12 --port 50152
//   agent-node client    -m model.gguf --chain 127.0.0.1:50152,127.0.0.1:50153 -p "Hello" -n 64
//   agent-node reference -m model.gguf -p "Hello" -n 64 --dump ref.txt
//   agent-node compare   chain.txt ref.txt
//
// the client sends tokens to the first node, every node sends its hidden states (fp16) straight to the next one,
// the last node sends the top-k logits back to the client: one transfer per node per token

#include "protocol.h"

#ifdef _WIN32
#    include <windows.h>
#    include <shellapi.h>
#endif

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
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
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
        return name == "--ignore-eos" || name == "--no-chat" || name == "-v" || name == "--verbose";
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

static void top_k(const float * logits, int32_t n_vocab, uint32_t k, top_entry * out) {
    k = std::min<uint32_t>(k, (uint32_t) n_vocab);
    std::vector<int32_t> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    auto gt = [&](int32_t a, int32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
    std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), gt);
    std::sort(idx.begin(), idx.begin() + k, gt);
    for (uint32_t i = 0; i < k; i++) {
        out[i] = { idx[i], logits[idx[i]] };
    }
}

struct model_part {
    const llama_model * model = nullptr;
    llama_context_ptr   ctx;
    int32_t first = 0, last = 0, n_layer = 0, n_embd = 0, n_vocab = 0;
    int32_t n_batch = 512;

    std::vector<float> hidden_in;

    model_part(const llama_model * model, const model_opts & mo, uint32_t n_ctx, int32_t n_threads) : model(model) {
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
        } else if (h.kind == payload::f16) {
            if (is_first()) {
                throw std::runtime_error("the first node takes tokens, not hidden states");
            }
            hidden_in.resize((size_t) h.n_tokens * n_embd);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, hidden_in.data(), (int64_t) hidden_in.size());
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
    tcp_conn_ptr                owner; // the client that opened the session, gets the results of the last node
    tcp_conn_ptr                next;  // the next node, null for the last node
    std::mutex                  mtx;
    stats_msg                   st;
};

struct node_server {
    llama_model_ptr model;
    model_opts      mo;
    int32_t         n_threads = 4;
    uint32_t        max_sessions = 4;

    std::mutex                                     mtx;
    std::map<uint64_t, std::shared_ptr<session>>   sessions;

    std::shared_ptr<session> find(uint64_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = sessions.find(id);
        if (it == sessions.end()) {
            throw std::runtime_error("unknown session " + std::to_string(id));
        }
        return it->second;
    }

    void on_open(const tcp_conn_ptr & conn, reader & r) {
        const open_msg m = open_msg::read(r);
        if (m.version != PROTOCOL_VERSION) {
            throw std::runtime_error("protocol version " + std::to_string(m.version) + ", this node speaks " + std::to_string(PROTOCOL_VERSION));
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (sessions.count(m.session)) {
                throw std::runtime_error("session already open");
            }
            if (sessions.size() >= max_sessions) {
                throw std::runtime_error("too many sessions");
            }
        }
        auto s = std::make_shared<session>();
        s->id    = m.session;
        s->n_top = std::clamp<uint32_t>(m.n_top, 1, 256);
        s->owner = conn;
        s->part  = std::make_unique<model_part>(model.get(), mo, m.n_ctx, n_threads);

        if (s->part->is_last() != m.next.empty()) {
            throw std::runtime_error(s->part->is_last() ? "this node runs the last layers, it has no next node"
                                                        : "this node does not run the last layers, it needs a next node");
        }
        if (!m.next.empty()) {
            s->next = tcp_connect(m.next);
            writer w;
            w.put(m.session);
            if (!send_frame(*s->next, msg::attach, w)) {
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
        const size_t need = h.kind == payload::tokens ? (size_t) h.n_tokens * 4 : (size_t) h.n_tokens * p.n_embd * 2;
        const uint8_t * data = r.get_bytes(need);

        s->st.bytes_in += 5 + body.size();
        s->st.n_tokens += h.n_tokens;
        s->st.n_forward++;

        static thread_local std::vector<float>     hidden;
        static thread_local std::vector<top_entry> top;
        uint32_t n_out = 0;

        const double t0 = now_ms();
        p.run(h, data, hidden, top, s->n_top, n_out);
        s->st.compute_ms += now_ms() - t0;

        if (!p.is_last()) {
            forward_hdr out = h;
            out.kind = payload::f16;
            writer w;
            out.write(w);
            ggml_fp32_to_fp16_row(hidden.data(), (ggml_fp16_t *) w.grow(hidden.size() * 2), (int64_t) hidden.size());
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
        msg type;
        std::vector<uint8_t> body;
        try {
            while (recv_frame(*conn, type, body)) {
                try {
                    reader r(body.data(), body.size());
                    switch (type) {
                        case msg::open:    on_open(conn, r); break;
                        case msg::attach:  break; // the previous node: forward frames follow
                        case msg::forward: on_forward(body); break;
                        case msg::stats: {
                            auto s = find(r.get<uint64_t>());
                            stats_msg st;
                            {
                                std::lock_guard<std::mutex> lock(s->mtx);
                                st = s->st;
                            }
                            send_frame(*conn, msg::stats_ok, st.write());
                        } break;
                        case msg::close: {
                            const uint64_t id = r.get<uint64_t>();
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
        drop_owned_by(conn);
    }
};

static int cmd_node(const cli_args & a) {
    node_server srv;
    srv.mo        = model_opts_from(a);
    srv.n_threads = a.get_int("-t", "--threads", common_cpu_get_num_math());

    const std::string host = a.get("--host", "", "127.0.0.1");
    const int         port = a.get_int("--port", "", 50152);

    srv.model = load_model(srv.mo);
    {
        const int32_t n_layer = llama_model_n_layer(srv.model.get());
        const int32_t first   = std::clamp(srv.mo.layer_start, 0, n_layer);
        const int32_t last    = srv.mo.layer_end <= 0 ? n_layer : std::min(srv.mo.layer_end, n_layer);
        AN_LOG("node: %s, layers [%d, %d) of %d, listening on %s:%d\n", model_desc(srv.model.get()).c_str(),
            first, last, n_layer, host.c_str(), port);
    }

    tcp_listener ln(host, port);
    // the scripts wait for this line
    printf("agent-node ready on %s:%d\n", host.c_str(), port);
    fflush(stdout);

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
    // runs n tokens from position pos0, returns the top-k of the last token
    virtual std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0) = 0;
};

struct chain_eval : evaluator {
    std::vector<std::string>  addrs;
    std::vector<tcp_conn_ptr> conns;
    std::vector<open_ok_msg>  parts;
    uint64_t                  sid = 0;
    uint32_t                  step = 0;
    uint32_t                  n_top = 8;

    chain_eval(const std::vector<std::string> & addrs, uint32_t n_ctx, uint32_t n_top) : addrs(addrs), n_top(n_top) {
        std::random_device rd;
        sid = ((uint64_t) rd() << 32) | rd();

        for (const auto & ad : addrs) {
            conns.push_back(tcp_connect(ad));
        }
        parts.resize(addrs.size());
        // open from the last node, so that every node can attach to an open session of the next one
        for (size_t i = addrs.size(); i-- > 0;) {
            open_msg m;
            m.session = sid;
            m.n_ctx   = n_ctx;
            m.n_top   = n_top;
            m.next    = i + 1 < addrs.size() ? addrs[i + 1] : "";
            send_frame(*conns[i], msg::open, m.write());

            std::vector<uint8_t> body = expect(i, msg::open_ok);
            reader r(body.data(), body.size());
            parts[i] = open_ok_msg::read(r);
        }

        for (size_t i = 0; i < parts.size(); i++) {
            const auto & p = parts[i];
            AN_LOG("node %zu (%s): layers [%d, %d) of %d\n", i, addrs[i].c_str(), p.layer_start, p.layer_end, p.n_layer);
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

    ~chain_eval() override {
        for (auto & c : conns) {
            writer w;
            w.put(sid);
            send_frame(*c, msg::close, w);
        }
    }

    std::vector<uint8_t> expect(size_t i, msg want) {
        msg type;
        std::vector<uint8_t> body;
        if (!recv_frame(*conns[i], type, body)) {
            throw std::runtime_error("node " + addrs[i] + " closed the connection");
        }
        if (type == msg::error) {
            reader r(body.data(), body.size());
            throw std::runtime_error("node " + addrs[i] + ": " + r.get_str());
        }
        if (type != want) {
            throw std::runtime_error("node " + addrs[i] + ": unexpected frame " + std::to_string((int) type));
        }
        return body;
    }

    std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0) override {
        forward_hdr h;
        h.session  = sid;
        h.step     = ++step;
        h.pos0     = pos0;
        h.n_tokens = n;
        h.kind     = payload::tokens;
        writer w;
        h.write(w);
        w.put_bytes(tokens, (size_t) n * sizeof(llama_token));
        if (!send_frame(*conns[0], msg::forward, w)) {
            throw std::runtime_error("node " + addrs[0] + " closed the connection");
        }

        std::vector<uint8_t> body = expect(conns.size() - 1, msg::result);
        reader r(body.data(), body.size());
        result_msg res = result_msg::read(r);
        if (res.step != step || res.n_out < 1) {
            throw std::runtime_error("result for step " + std::to_string(res.step) + ", expected " + std::to_string(step));
        }
        return std::vector<top_entry>(res.top.end() - res.n_top, res.top.end());
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

    std::vector<top_entry> eval(const llama_token * tokens, uint32_t n, int32_t pos0) override {
        forward_hdr h;
        h.pos0     = pos0;
        h.n_tokens = n;
        h.kind     = payload::tokens;
        std::vector<float>     hidden;
        std::vector<top_entry> top;
        uint32_t n_out = 0;
        part.run(h, (const uint8_t *) tokens, hidden, top, n_top, n_out);
        return top;
    }
};

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
};

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
    int32_t n_gen_eval = 0; // single-token forwards timed in gen_ms
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
        const llama_token tok = top[0].id;
        res.tokens.push_back(tok);
        if (dump) {
            fprintf(dump, "%d %d", i, tok);
            for (const auto & e : top) {
                fprintf(dump, " %d:%.6f", e.id, e.logit);
            }
            fprintf(dump, "\n");
        }
        printf("%s", common_token_to_piece(vocab, tok, false).c_str());
        fflush(stdout);

        if (i + 1 == g.n_gen || (!g.ignore_eos && llama_vocab_is_eog(vocab, tok))) {
            break;
        }
        top = ev.eval(&tok, 1, pos++);
        res.n_gen_eval++;
    }
    res.gen_ms = now_ms() - t0;
    printf("\n");
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
    if (addrs.empty()) {
        throw std::runtime_error("--chain host:port,host:port,... is required");
    }

    // the client only needs the tokenizer
    model_opts mo = model_opts_from(a);
    mo.vocab_only = true;
    llama_model_ptr model = load_model(mo);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    const std::vector<llama_token> prompt = make_prompt(vocab, model.get(), g);

    chain_eval chain(addrs, g.n_ctx, g.n_top);
    if (chain.parts[0].n_vocab != llama_vocab_n_tokens(vocab)) {
        throw std::runtime_error("the tokenizer does not match the model of the chain");
    }

    std::vector<stats_msg> st0;
    const gen_result r = generate(chain, vocab, prompt, g, [&]() { st0 = chain.stats(); });
    const std::vector<stats_msg> st1 = chain.stats();

    print_speed(r);

    // traffic of the timed generation: every forward carries one token
    std::ostringstream nodes_json;
    for (size_t i = 0; i < st1.size(); i++) {
        const double n   = std::max<double>(1, (double) (st1[i].n_tokens - st0[i].n_tokens));
        const double in  = (st1[i].bytes_in  - st0[i].bytes_in)  / n;
        const double out = (st1[i].bytes_out - st0[i].bytes_out) / n;
        const double cms = (st1[i].compute_ms - st0[i].compute_ms) / n;
        AN_LOG("node %zu: %.0f bytes in + %.0f bytes out per token, compute %.2f ms/token\n", i, in, out, cms);
        nodes_json << (i ? "," : "") << "{\"addr\":\"" << addrs[i] << "\",\"layer_start\":" << chain.parts[i].layer_start
                   << ",\"layer_end\":" << chain.parts[i].layer_end << ",\"bytes_in_per_token\":" << in
                   << ",\"bytes_out_per_token\":" << out << ",\"compute_ms_per_token\":" << cms << "}";
    }

    if (!g.json.empty()) {
        std::ofstream f(u8_path(g.json));
        f << "{\"n_nodes\":" << addrs.size() << ",\"n_embd\":" << chain.parts[0].n_embd << ",\"n_prompt\":" << r.n_prompt
          << ",\"prefill_ms\":" << r.prefill_ms << ",\"n_gen\":" << r.n_gen_eval << ",\"gen_ms\":" << r.gen_ms
          << ",\"tg_tok_s\":" << (r.n_gen_eval > 0 ? r.n_gen_eval * 1000.0 / r.gen_ms : 0.0)
          << ",\"pp_tok_s\":" << r.n_prompt * 1000.0 / std::max(r.prefill_ms, 1e-9)
          << ",\"nodes\":[" << nodes_json.str() << "]}\n";
    }
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

static int cmd_compare(const cli_args & a) {
    if (a.positional.size() != 2) {
        throw std::runtime_error("usage: agent-node compare chain.txt reference.txt [--tokens 128] [--tol 0.01]");
    }
    const auto x = read_dump(a.positional[0]);
    const auto y = read_dump(a.positional[1]);
    const size_t want = (size_t) a.get_int("--tokens", "", 128);
    const double tol  = std::stod(a.get("--tol", "", "0.01"));

    const size_t n = std::min({ x.size(), y.size(), want });
    size_t n_same = 0;
    while (n_same < n && x[n_same].token == y[n_same].token) {
        n_same++;
    }

    // the logits are comparable while the contexts are equal: up to and including the first different token
    double max_diff = 0, max_abs = 0;
    const size_t n_cmp = std::min(n, n_same + 1);
    for (size_t i = 0; i < n_cmp; i++) {
        for (const auto & ey : y[i].top) {
            max_abs = std::max(max_abs, (double) std::fabs(ey.logit));
            for (const auto & ex : x[i].top) {
                if (ex.id == ey.id) {
                    max_diff = std::max(max_diff, (double) std::fabs(ex.logit - ey.logit));
                }
            }
        }
    }
    const double rel = max_abs > 0 ? max_diff / max_abs : 0;
    const bool tokens_ok = n_same == n && n == want;
    const bool logits_ok = rel <= tol;

    printf("tokens: %zu of %zu equal (want %zu)%s\n", n_same, n, want,
        n_same < n ? (" - first difference at token " + std::to_string(n_same)).c_str() : "");
    printf("logits: max |diff| = %.4f, max |logit| = %.2f, relative = %.2e (tolerance %.0e)\n", max_diff, max_abs, rel, tol);
    printf("%s\n", tokens_ok && logits_ok ? "PASS" : "FAIL");
    return tokens_ok && logits_ok ? 0 : 1;
}

// prints the shape of the model as one JSON line, for the scripts that split it into layer ranges
// reads only the GGUF metadata, no weights
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
    printf("{\"arch\":\"%s\",\"n_layer\":%lld,\"n_embd\":%lld,\"full_attention_interval\":%lld}\n", arch.c_str(),
        get_u32(arch + ".block_count"), get_u32(arch + ".embedding_length"), get_u32(arch + ".full_attention_interval"));
    gguf_free(g);
    return 0;
}

static void usage() {
    fprintf(stderr,
        "usage:\n"
        "  agent-node node      -m model.gguf --layers START:END [--host 127.0.0.1] [--port 50152] [-ngl 999] [--device CUDA0] [-t N]\n"
        "  agent-node client    -m model.gguf --chain HOST:PORT,HOST:PORT,... [-p TEXT | -f FILE] [-n 64] [-c 4096]\n"
        "                       [--top 8] [--ignore-eos] [--no-chat] [--dump FILE] [--json FILE]\n"
        "  agent-node reference -m model.gguf [-p TEXT | -f FILE] [-n 64] [-c 4096] [-ngl 999] [--device CUDA0] [--dump FILE]\n"
        "  agent-node compare   CHAIN_DUMP REFERENCE_DUMP [--tokens 128] [--tol 0.01]\n"
        "  agent-node info      -m model.gguf\n"
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
        if (cmd == "compare") {
            return cmd_compare(a);
        }
        if (cmd == "info") {
            return cmd_info(a);
        }

        llama_log_set(log_callback, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        net_init();

        if (cmd == "node") {
            return cmd_node(a);
        }
        if (cmd == "client") {
            return cmd_client(a);
        }
        if (cmd == "reference") {
            return cmd_reference(a);
        }
        usage();
        return 2;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
