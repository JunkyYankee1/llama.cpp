// frames of the node transport, see docs/spec/node-transport.md in the agent repository
//
// frame = u32 body size | u8 type | body, all integers little-endian
// the hidden states travel as raw fp16 rows after the header fields, without copies into a container format
#pragma once

#include "net.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace agent {

static_assert(sizeof(float) == 4, "float must be 32-bit");

constexpr uint32_t PROTOCOL_VERSION = 1;
constexpr uint32_t MAX_FRAME_SIZE   = 256u << 20;

enum class msg : uint8_t {
    open     = 1,  // client -> node: create a session
    open_ok  = 2,  // node -> client: the session is ready
    attach   = 3,  // node -> next node: the following forward frames come over this connection
    forward  = 4,  // client -> first node, node -> next node: tokens or hidden states
    result   = 5,  // last node -> client: top-k logits
    close    = 6,  // client -> node: drop the session
    stats    = 7,  // client -> node: ask for the counters of a session
    stats_ok = 8,  // node -> client
    error    = 9,  // any direction: the request failed, body is a message
    hello    = 10, // anyone -> node: what do you run (no token needed; liveness and discovery of spare nodes)
    hello_ok = 11, // node -> anyone
};

enum class payload : uint8_t {
    tokens = 0, // int32 token ids
    f16    = 1, // fp16 hidden states, n_tokens rows of n_embd
    f32    = 2, // fp32 hidden states: exact, twice the traffic, for checks
    q8     = 3, // int8 hidden states as ggml Q8_0 blocks: an fp16 scale per 32 values, 34 bytes per 32 (n_embd % 32 == 0)
};

constexpr int32_t Q8_BLOCK       = 32;
constexpr size_t  Q8_BLOCK_BYTES = 34;

inline size_t payload_size(payload kind, uint32_t n_tokens, int32_t n_embd) {
    switch (kind) {
        case payload::tokens: return (size_t) n_tokens * 4;
        case payload::f16:    return (size_t) n_tokens * n_embd * 2;
        case payload::f32:    return (size_t) n_tokens * n_embd * 4;
        case payload::q8:     return (size_t) n_tokens * (n_embd / Q8_BLOCK) * Q8_BLOCK_BYTES;
    }
    throw std::runtime_error("unknown payload kind " + std::to_string((int) kind));
}

enum forward_flags : uint8_t {
    FORWARD_OUT_ALL = 1, // the last node returns logits for every token, not only the last one
};

class writer {
  public:
    template <typename T> writer & put(T v) {
        const size_t o = buf.size();
        buf.resize(o + sizeof(T));
        memcpy(buf.data() + o, &v, sizeof(T));
        return *this;
    }
    writer & put_str(const std::string & s) {
        put<uint32_t>((uint32_t) s.size());
        return put_bytes(s.data(), s.size());
    }
    writer & put_bytes(const void * data, size_t n) {
        const size_t o = buf.size();
        buf.resize(o + n);
        if (n > 0) {
            memcpy(buf.data() + o, data, n);
        }
        return *this;
    }
    // reserve n bytes at the end and return a pointer to them
    uint8_t * grow(size_t n) {
        const size_t o = buf.size();
        buf.resize(o + n);
        return buf.data() + o;
    }

    std::vector<uint8_t> buf;
};

class reader {
  public:
    reader(const uint8_t * data, size_t size) : p(data), end(data + size) {}

    template <typename T> T get() {
        need(sizeof(T));
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string get_str() {
        const uint32_t n = get<uint32_t>();
        need(n);
        std::string s((const char *) p, n);
        p += n;
        return s;
    }
    const uint8_t * get_bytes(size_t n) {
        need(n);
        const uint8_t * r = p;
        p += n;
        return r;
    }
    size_t left() const { return (size_t) (end - p); }

  private:
    void need(size_t n) const {
        if ((size_t) (end - p) < n) {
            throw std::runtime_error("truncated frame");
        }
    }
    const uint8_t * p;
    const uint8_t * end;
};

// one frame on the wire: 5 bytes of header + body
inline size_t frame_size(const writer & body) {
    return 5 + body.buf.size();
}

inline bool send_frame(tcp_conn & c, msg type, const writer & body) {
    uint8_t hdr[5];
    const uint32_t n = (uint32_t) body.buf.size();
    memcpy(hdr, &n, 4);
    hdr[4] = (uint8_t) type;

    std::lock_guard<std::mutex> lock(c.send_mutex);
    return c.send_all(hdr, sizeof(hdr)) && c.send_all(body.buf.data(), body.buf.size());
}

// returns false when the peer closed the connection
inline bool recv_frame(tcp_conn & c, msg & type, std::vector<uint8_t> & body) {
    uint8_t hdr[5];
    if (!c.recv_all(hdr, sizeof(hdr))) {
        return false;
    }
    uint32_t n;
    memcpy(&n, hdr, 4);
    if (n > MAX_FRAME_SIZE) {
        throw std::runtime_error("frame too large: " + std::to_string(n));
    }
    type = (msg) hdr[4];
    body.resize(n);
    return n == 0 || c.recv_all(body.data(), n);
}

inline bool send_error(tcp_conn & c, const std::string & text) {
    writer w;
    w.put_str(text);
    return send_frame(c, msg::error, w);
}

//
// message bodies
//

struct open_msg {
    uint32_t    version = PROTOCOL_VERSION;
    uint64_t    session = 0;
    uint32_t    n_ctx   = 4096;
    uint32_t    n_top   = 8;  // logits per output position returned by the last node
    std::string next;         // address of the next node, empty: this is the last node, results go back to the client
    payload     hidden  = payload::f16; // how the nodes of this session pass hidden states (optional field)
    uint32_t    n_rollback = 0;         // how many trailing tokens of a forward may be rolled back (speculative decoding, optional field)
    std::string token;                  // access token of this node for the session (optional field)
    std::string next_token;             // access token of the next node, presented by this node in attach
    std::string next_pin;               // expected certificate fingerprint of the next node ("" = not checked)

    writer write() const {
        writer w;
        w.put(version).put(session).put(n_ctx).put(n_top).put_str(next).put((uint8_t) hidden).put(n_rollback);
        w.put_str(token).put_str(next_token).put_str(next_pin);
        return w;
    }
    static open_msg read(reader & r) {
        open_msg m;
        m.version = r.get<uint32_t>();
        m.session = r.get<uint64_t>();
        m.n_ctx   = r.get<uint32_t>();
        m.n_top   = r.get<uint32_t>();
        m.next    = r.get_str();
        if (r.left() >= 1) {
            m.hidden = (payload) r.get<uint8_t>();
        }
        if (r.left() >= 4) {
            m.n_rollback = r.get<uint32_t>();
        }
        if (r.left() >= 12) {
            m.token      = r.get_str();
            m.next_token = r.get_str();
            m.next_pin   = r.get_str();
        }
        return m;
    }
};

struct open_ok_msg {
    int32_t     layer_start = 0;
    int32_t     layer_end   = 0;
    int32_t     n_layer     = 0;
    int32_t     n_embd      = 0;
    int32_t     n_vocab     = 0;
    std::string desc; // model description, the client checks that every node runs the same model

    writer write() const {
        writer w;
        w.put(layer_start).put(layer_end).put(n_layer).put(n_embd).put(n_vocab).put_str(desc);
        return w;
    }
    static open_ok_msg read(reader & r) {
        open_ok_msg m;
        m.layer_start = r.get<int32_t>();
        m.layer_end   = r.get<int32_t>();
        m.n_layer     = r.get<int32_t>();
        m.n_embd      = r.get<int32_t>();
        m.n_vocab     = r.get<int32_t>();
        m.desc        = r.get_str();
        return m;
    }
};

struct attach_msg {
    uint64_t    session = 0;
    std::string token; // access token of the receiving node for the session

    writer write() const {
        writer w;
        w.put(session).put_str(token);
        return w;
    }
    static attach_msg read(reader & r) {
        attach_msg m;
        m.session = r.get<uint64_t>();
        if (r.left() >= 4) {
            m.token = r.get_str();
        }
        return m;
    }
};

struct hello_ok_msg {
    uint32_t    version     = PROTOCOL_VERSION;
    int32_t     layer_start = 0;
    int32_t     layer_end   = 0;
    int32_t     n_layer     = 0;
    uint8_t     tls         = 0;
    uint8_t     auth        = 0; // the node requires access tokens
    std::string desc;

    writer write() const {
        writer w;
        w.put(version).put(layer_start).put(layer_end).put(n_layer).put(tls).put(auth).put_str(desc);
        return w;
    }
    static hello_ok_msg read(reader & r) {
        hello_ok_msg m;
        m.version     = r.get<uint32_t>();
        m.layer_start = r.get<int32_t>();
        m.layer_end   = r.get<int32_t>();
        m.n_layer     = r.get<int32_t>();
        m.tls         = r.get<uint8_t>();
        m.auth        = r.get<uint8_t>();
        m.desc        = r.get_str();
        return m;
    }
};

// header of a forward frame, the payload follows: n_tokens * 4 bytes of token ids or n_tokens * n_embd * 2 bytes of fp16
struct forward_hdr {
    uint64_t session  = 0;
    uint32_t step     = 0;  // incremented by the client for every forward
    int32_t  pos0     = 0;  // position of the first token, the following tokens are consecutive;
                            // pos0 before the end of the previous forward rolls the rejected draft tokens back
    uint32_t n_tokens = 0;
    payload  kind     = payload::tokens;
    uint8_t  flags    = 0;

    void write(writer & w) const {
        w.put(session).put(step).put(pos0).put(n_tokens).put((uint8_t) kind).put(flags);
    }
    static forward_hdr read(reader & r) {
        forward_hdr h;
        h.session  = r.get<uint64_t>();
        h.step     = r.get<uint32_t>();
        h.pos0     = r.get<int32_t>();
        h.n_tokens = r.get<uint32_t>();
        h.kind     = (payload) r.get<uint8_t>();
        h.flags    = r.get<uint8_t>();
        return h;
    }
};

struct top_entry {
    int32_t id;
    float   logit;
};

struct result_msg {
    uint64_t session = 0;
    uint32_t step    = 0;
    uint32_t n_out   = 0;
    uint32_t n_top   = 0;
    std::vector<top_entry> top; // n_out * n_top, sorted by logit, descending

    writer write() const {
        writer w;
        w.put(session).put(step).put(n_out).put(n_top);
        w.put_bytes(top.data(), top.size() * sizeof(top_entry));
        return w;
    }
    static result_msg read(reader & r) {
        result_msg m;
        m.session = r.get<uint64_t>();
        m.step    = r.get<uint32_t>();
        m.n_out   = r.get<uint32_t>();
        m.n_top   = r.get<uint32_t>();
        const size_t n = (size_t) m.n_out * m.n_top;
        m.top.resize(n);
        memcpy(m.top.data(), r.get_bytes(n * sizeof(top_entry)), n * sizeof(top_entry));
        return m;
    }
};

struct stats_msg {
    uint64_t bytes_in   = 0; // forward frames received for the session
    uint64_t bytes_out  = 0; // forward and result frames sent for the session
    uint64_t n_tokens   = 0;
    uint64_t n_forward  = 0;
    double   compute_ms = 0; // time spent in llama_process and in reading its outputs
    double   output_ms  = 0; // of it: reading the outputs (hidden states or logits and top-k)

    writer write() const {
        writer w;
        w.put(bytes_in).put(bytes_out).put(n_tokens).put(n_forward).put(compute_ms).put(output_ms);
        return w;
    }
    static stats_msg read(reader & r) {
        stats_msg m;
        m.bytes_in   = r.get<uint64_t>();
        m.bytes_out  = r.get<uint64_t>();
        m.n_tokens   = r.get<uint64_t>();
        m.n_forward  = r.get<uint64_t>();
        m.compute_ms = r.get<double>();
        if (r.left() >= 8) {
            m.output_ms = r.get<double>();
        }
        return m;
    }
};

} // namespace agent
