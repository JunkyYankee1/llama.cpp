// Direct connections between nodes behind NAT: UDP hole punching through the rendezvous server of the network
// (the UDP side of the relay port), KCP for reliable ordered delivery over UDP, and the usual TLS on top.
//
// How a connection is made (docs/spec/node-transport.md, T7):
//   - a node behind NAT sends "AGP1 REG <id> <expires> <ticket> <own LAN addresses>" to the rendezvous server every
//     15 s from its UDP socket; this keeps the mapping of its NAT open and tells the server where the node is seen
//   - a peer that wants the node sends "AGP1 CONN <id> <req> <own LAN addresses>" from its UDP socket; the server
//     answers it with the address of the node and a random nonce, and sends the node an INVITE with the address
//     of the peer and the same nonce
//   - both sides send "AGP1 PUNCH <nonce>" to every address of the other side; the first packet that gets
//     through opens the path (the outgoing packet of each side opened its own NAT for the other)
//   - KCP runs over that path; a local socket pair joins it to a normal tcp_conn, so TLS, frames, timeouts and
//     the rest of the code stay exactly as on TCP
// When punching fails (symmetric NAT, UDP blocked) the caller falls back to the TCP relay.
//
// Datagrams: control messages start with "AGP1 " (ASCII), KCP data with the byte 0x01. IPv4 only.
#pragma once

#include "net.h"
#include "kcp/ikcp.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <mstcpip.h> // SIO_UDP_CONNRESET
#endif

namespace agent {

inline int64_t p2p_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::string sa_str(const sockaddr_in & sa) {
    char ip[INET_ADDRSTRLEN] = {};
    inet_ntop(AF_INET, &sa.sin_addr, ip, sizeof(ip));
    return std::string(ip) + ":" + std::to_string(ntohs(sa.sin_port));
}

inline bool sa_parse(const std::string & s, sockaddr_in & sa) {
    const size_t p = s.rfind(':');
    if (p == std::string::npos) {
        return false;
    }
    sa = {};
    sa.sin_family = AF_INET;
    const int port = std::atoi(s.c_str() + p + 1);
    if (port <= 0 || port > 65535 || inet_pton(AF_INET, s.substr(0, p).c_str(), &sa.sin_addr) != 1) {
        return false;
    }
    sa.sin_port = htons((uint16_t) port);
    return true;
}

inline bool sa_resolve(const std::string & hostport, sockaddr_in & sa) {
    std::string host;
    int port = 0;
    split_addr(hostport, host, port);
    addrinfo hints = {};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo * res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) {
        return false;
    }
    sa = *(const sockaddr_in *) res->ai_addr;
    freeaddrinfo(res);
    return true;
}

inline bool sa_equal(const sockaddr_in & a, const sockaddr_in & b) {
    return a.sin_addr.s_addr == b.sin_addr.s_addr && a.sin_port == b.sin_port;
}

// two connected stream sockets in this process: the application uses one as a tcp_conn, the bridge the other
inline void local_pair(agent_socket_t & app, agent_socket_t & bridge) {
#ifdef _WIN32
    // no socketpair on Windows: a loopback TCP connection
    agent_socket_t ln = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa = {};
    sa.sin_family      = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int len = sizeof(sa);
    if (ln == AGENT_INVALID_SOCKET || ::bind(ln, (sockaddr *) &sa, sizeof(sa)) != 0 || ::listen(ln, 1) != 0 ||
        getsockname(ln, (sockaddr *) &sa, &len) != 0) {
        if (ln != AGENT_INVALID_SOCKET) {
            socket_close(ln);
        }
        throw std::runtime_error("p2p: cannot create a local socket pair");
    }
    app = ::socket(AF_INET, SOCK_STREAM, 0);
    if (app == AGENT_INVALID_SOCKET || ::connect(app, (sockaddr *) &sa, sizeof(sa)) != 0) {
        socket_close(ln);
        throw std::runtime_error("p2p: cannot create a local socket pair");
    }
    bridge = ::accept(ln, nullptr, nullptr);
    socket_close(ln);
    if (bridge == AGENT_INVALID_SOCKET) {
        socket_close(app);
        throw std::runtime_error("p2p: cannot create a local socket pair");
    }
    int one = 1;
    setsockopt(bridge, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
#else
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        throw std::runtime_error("p2p: cannot create a local socket pair");
    }
    app    = sv[0];
    bridge = sv[1];
#endif
    socket_set_nonblocking(bridge);
}

// the tcp_conn of the application side: TCP_NODELAY fails on a unix socket, which is harmless
inline tcp_conn_ptr local_conn(agent_socket_t app) {
    return std::make_shared<tcp_conn>(app);
}

class p2p_endpoint;

// one KCP stream to a peer, bridged to a local socket
struct p2p_session {
    p2p_endpoint * ep   = nullptr;
    IUINT32        conv = 0;
    sockaddr_in    peer = {};
    ikcpcb *       kcp  = nullptr;
    agent_socket_t local = AGENT_INVALID_SOCKET; // bridge side of the local pair
    std::mutex     m;
    std::string    to_local;          // received from the peer, not yet taken by the local socket
    std::atomic<int64_t> last_recv{0}; // ms, any packet from the peer
    std::atomic<int64_t> last_sent{0}; // ms, any packet to the peer
    std::atomic<bool> closed{false};    // over: the peer is gone or the session ended
    std::atomic<bool> peer_fin{false};  // the peer sent everything and closed
    bool           fin_sent = false;
    std::atomic<uint64_t> bytes_in{0}, bytes_out{0};

    ~p2p_session() {
        if (kcp) {
            ikcp_release(kcp);
        }
        if (local != AGENT_INVALID_SOCKET) {
            socket_close(local);
        }
    }

    // moves what KCP has received into to_local and on into the local socket; m must be held
    void drain() {
        char buf[65536];
        while (to_local.size() < (4u << 20)) {
            const int n = ikcp_recv(kcp, buf, sizeof(buf));
            if (n <= 0) {
                break;
            }
            to_local.append(buf, (size_t) n);
            bytes_in += (uint64_t) n;
        }
        while (!to_local.empty()) {
            const long r = (long) ::send(local, to_local.data(), (int) std::min<size_t>(to_local.size(), 1 << 20), 0);
            if (r <= 0) {
                break; // full (would block) or closed: the session loop sees it
            }
            to_local.erase(0, (size_t) r);
        }
    }
};

using p2p_session_ptr = std::shared_ptr<p2p_session>;

class p2p_endpoint {
  public:
    // tuning, read once from the environment (AGENT_P2P_*): defaults are for a home uplink of tens of Mbit/s
    int punch_ms   = 1500; // how long to try punching before falling back to the relay
    int snd_wnd    = 256;  // KCP windows in packets of about 1.2 KB
    int rcv_wnd    = 1024;
    int dead_ms    = 10000; // no packet from the peer this long: the session is over (keepalive every 2.5 s)
    int failed_ttl = 600; // s: after a failed punch to a node, go straight to the relay for this long
    bool debug     = false; // AGENT_P2P_DEBUG=1: every control datagram to stderr
    int low_ttl    = 3;     // first punches: past the own router(s), not as far as the router of the peer
    int low_ttl_ms = 200;
    int default_ttl = 64;

    static p2p_endpoint & get() {
        // never destroyed: its threads run until the process ends
        static p2p_endpoint * ep = new p2p_endpoint();
        return *ep;
    }

    // the process ends: tell every peer at once, its sessions end now instead of after dead_ms
    void goodbye() {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto & kv : sessions) {
            const std::string fin = "AGP1 FIN " + hex32(kv.first);
            send_ctl(kv.second->peer, fin);
            send_ctl(kv.second->peer, fin);
        }
    }

    bool ok() const { return sock != AGENT_INVALID_SOCKET; }
    int  port() const { return bound_port; }

    // connects to the node `id` that is registered at the rendezvous server `server` (host:port); returns the
    // application side of the stream, or nullptr with the reason in err
    tcp_conn_ptr connect(const std::string & server, const std::string & id, std::string & err) {
        if (!ok()) {
            err = "no UDP socket";
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = failed.find(id);
            if (it != failed.end() && it->second > p2p_now_ms()) {
                err = "a direct connection to this node failed recently";
                return nullptr;
            }
        }
        sockaddr_in srv;
        if (!sa_resolve(server, srv)) {
            err = "cannot resolve " + server;
            return nullptr;
        }
        const std::string req = random_hex(8);
        auto rq = std::make_shared<request>();
        {
            std::lock_guard<std::mutex> lock(mtx);
            requests[req] = rq;
        }
        const std::string conn = "AGP1 CONN " + id + " " + req + " " + local_list(srv);
        const int64_t t0 = p2p_now_ms();
        {
            std::unique_lock<std::mutex> lock(mtx);
            for (int attempt = 0; attempt < 3 && !rq->done; attempt++) {
                send_ctl(srv, conn);
                cv.wait_for(lock, std::chrono::milliseconds(400), [&]() { return rq->done; });
            }
            requests.erase(req);
        }
        if (!rq->done || !rq->error.empty()) {
            err = rq->done ? "rendezvous: " + rq->error : "the rendezvous server did not answer (UDP blocked?)";
            std::lock_guard<std::mutex> lock(mtx);
            failed[id] = p2p_now_ms() + (int64_t) failed_ttl * 1000;
            return nullptr;
        }
        // punch: the timer thread sends PUNCH to every address of the node until a packet of the node arrives
        auto pn = std::make_shared<punch>();
        pn->nonce     = rq->nonce;
        pn->conv      = conv_of(rq->nonce);
        pn->cands     = rq->cands;
        pn->outgoing  = true;
        pn->expires   = p2p_now_ms() + 15000;
        {
            std::lock_guard<std::mutex> lock(mtx);
            punches[pn->nonce] = pn;
        }
        send_punches(*pn);
        p2p_session_ptr ss;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(punch_ms), [&]() { return pn->session != nullptr; });
            ss = pn->session;
            if (!ss) {
                failed[id] = p2p_now_ms() + (int64_t) failed_ttl * 1000;
                punches.erase(pn->nonce);
            }
        }
        if (!ss) {
            err = "no direct path (tried " + std::to_string(pn->cands.size()) + " addresses)";
            return nullptr;
        }
        last_ms = (double) (p2p_now_ms() - t0);
        last_path = sa_str(ss->peer);
        return pn->app;
    }

    // a node: registers at the rendezvous server and hands the sessions of the peers to on_accept;
    // called again with a fresh ticket before the old one expires
    void listen(const std::string & server, const std::string & id, long long expires, const std::string & ticket,
                std::function<void(tcp_conn_ptr)> on_accept) {
        if (!ok()) {
            return;
        }
        sockaddr_in srv;
        if (!sa_resolve(server, srv)) {
            fprintf(stderr, "warning: p2p: cannot resolve the rendezvous server %s\n", server.c_str());
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            reg_server = srv;
            reg_line   = "AGP1 REG " + id + " " + std::to_string(expires) + " " + ticket + " " + local_list(srv);
            accept_cb  = std::move(on_accept);
            reg_next   = 0; // now
        }
    }

    // what the last outgoing connection took (for the logs of the client)
    double      last_ms = 0;
    std::string last_path;
    std::string public_addr; // as the rendezvous server sees this node

    std::atomic<uint64_t> n_direct{0};
    std::atomic<uint64_t> n_sessions{0};

  private:
    struct request {
        bool done = false;
        std::string error, nonce;
        std::vector<sockaddr_in> cands;
    };
    struct punch {
        std::string nonce;
        IUINT32     conv = 0;
        std::vector<sockaddr_in> cands;
        bool        outgoing = false;
        int64_t     expires = 0;
        int64_t     next_send = 0;
        int64_t     started = 0;
        p2p_session_ptr session; // set once a packet of the peer arrived
        tcp_conn_ptr    app;     // application side (outgoing)
    };

    agent_socket_t sock = AGENT_INVALID_SOCKET;
    int bound_port = 0;
    std::mutex mtx;
    std::condition_variable cv;
    std::map<std::string, std::shared_ptr<request>> requests;
    std::map<std::string, std::shared_ptr<punch>> punches; // by nonce
    std::map<IUINT32, p2p_session_ptr> sessions;           // by conv
    std::map<std::string, int64_t> failed;                 // node id -> retry the direct path after (ms)
    sockaddr_in reg_server = {};
    std::string reg_line;
    int64_t     reg_next = -1;
    std::function<void(tcp_conn_ptr)> accept_cb;
    std::mt19937_64 rng{std::random_device{}()};

    p2p_endpoint() {
        auto env_int = [](const char * name, int def) {
            const char * v = getenv(name);
            return v ? std::atoi(v) : def;
        };
        punch_ms   = env_int("AGENT_P2P_PUNCH_MS", punch_ms);
        snd_wnd    = env_int("AGENT_P2P_SND_WND", snd_wnd);
        rcv_wnd    = env_int("AGENT_P2P_RCV_WND", rcv_wnd);
        failed_ttl = env_int("AGENT_P2P_FAILED_TTL", failed_ttl);
        debug      = env_int("AGENT_P2P_DEBUG", 0) != 0;
        low_ttl    = env_int("AGENT_P2P_LOW_TTL", low_ttl);
        low_ttl_ms = env_int("AGENT_P2P_LOW_TTL_MS", low_ttl_ms);
        const int want_port = env_int("AGENT_P2P_PORT", 0);

        sock = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock == AGENT_INVALID_SOCKET) {
            return;
        }
        sockaddr_in sa = {};
        sa.sin_family      = AF_INET;
        sa.sin_addr.s_addr = htonl(INADDR_ANY);
        sa.sin_port        = htons((uint16_t) want_port);
        if (::bind(sock, (sockaddr *) &sa, sizeof(sa)) != 0) {
            fprintf(stderr, "warning: p2p: cannot bind UDP port %d\n", want_port);
            socket_close(sock);
            sock = AGENT_INVALID_SOCKET;
            return;
        }
        socklen_t len = sizeof(sa);
        getsockname(sock, (sockaddr *) &sa, &len);
        bound_port = ntohs(sa.sin_port);
        {
            int t = 0;
            socklen_t tl = sizeof(t);
            if (getsockopt(sock, IPPROTO_IP, IP_TTL, (char *) &t, &tl) == 0 && t > 0) {
                default_ttl = t;
            }
        }
#ifdef _WIN32
        // an ICMP "port unreachable" must not break recvfrom on Windows
        BOOL no = FALSE;
        DWORD ret = 0;
        WSAIoctl(sock, SIO_UDP_CONNRESET, &no, sizeof(no), nullptr, 0, &ret, nullptr, nullptr);
#endif
        int big = 4 << 20;
        setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (const char *) &big, sizeof(big));
        setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char *) &big, sizeof(big));
        std::thread([this]() { recv_loop(); }).detach();
        std::thread([this]() { timer_loop(); }).detach();
    }

    std::string random_hex(size_t nbytes) {
        std::lock_guard<std::mutex> lock(rng_mtx);
        std::string s;
        static const char * d = "0123456789abcdef";
        for (size_t i = 0; i < nbytes * 2; i++) {
            s += d[rng() & 15];
        }
        return s;
    }
    std::mutex rng_mtx;

    static IUINT32 conv_of(const std::string & nonce) {
        IUINT32 c = (IUINT32) std::stoul(nonce.substr(0, 8), nullptr, 16);
        return c == 0 ? 1 : c;
    }

    // the LAN address of this PC towards `to` (the interface the OS would use) with the port of the UDP socket
    std::string local_list(const sockaddr_in & to) {
        agent_socket_t s = ::socket(AF_INET, SOCK_DGRAM, 0);
        std::string out = "-";
        if (s == AGENT_INVALID_SOCKET) {
            return out;
        }
        if (::connect(s, (const sockaddr *) &to, sizeof(to)) == 0) { // UDP connect sends nothing
            sockaddr_in me = {};
            socklen_t len = sizeof(me);
            if (getsockname(s, (sockaddr *) &me, &len) == 0 && me.sin_addr.s_addr != 0) {
                me.sin_port = htons((uint16_t) bound_port);
                out = sa_str(me);
            }
        }
        socket_close(s);
        return out;
    }

    void send_raw(const sockaddr_in & to, const char * data, size_t n, int ttl = 0) {
        std::lock_guard<std::mutex> lock(send_mtx);
        if (ttl > 0) {
            setsockopt(sock, IPPROTO_IP, IP_TTL, (const char *) &ttl, sizeof(ttl));
        }
        ::sendto(sock, data, (int) n, 0, (const sockaddr *) &to, sizeof(to));
        if (ttl > 0) {
            setsockopt(sock, IPPROTO_IP, IP_TTL, (const char *) &default_ttl, sizeof(default_ttl));
        }
    }
    std::mutex send_mtx; // the TTL of the socket is changed around a single packet
    void send_ctl(const sockaddr_in & to, const std::string & msg) {
        if (debug) {
            fprintf(stderr, "p2p: -> %s %.120s\n", sa_str(to).c_str(), msg.c_str());
        }
        send_raw(to, msg.data(), msg.size());
    }

    // Both sides start with packets that die on the way (low TTL): they open the own NAT for the peer but do not
    // reach the NAT of the peer. A router that remembers a stray packet from outside would otherwise give the
    // outgoing packets of the peer another port, and the paths would never meet. After low_ttl_ms both sides
    // send normal packets, and each finds the NAT of the other already open.
    void send_punches(punch & pn) {
        const std::string msg = "AGP1 PUNCH " + pn.nonce;
        const int64_t now = p2p_now_ms();
        if (pn.started == 0) {
            pn.started = now;
        }
        const int ttl = low_ttl > 0 && now - pn.started < low_ttl_ms ? low_ttl : 0;
        for (const auto & c : pn.cands) {
            if (debug) {
                fprintf(stderr, "p2p: -> %s %s%s\n", sa_str(c).c_str(), msg.c_str(), ttl ? " (low TTL)" : "");
            }
            send_raw(c, msg.data(), msg.size(), ttl);
        }
        pn.next_send = now + 100;
    }

    static std::vector<sockaddr_in> parse_cands(const std::string & pub, const std::string & locals) {
        std::vector<sockaddr_in> out;
        sockaddr_in sa;
        if (sa_parse(pub, sa)) {
            out.push_back(sa);
        }
        std::stringstream ss(locals);
        std::string a;
        while (std::getline(ss, a, ',')) {
            if (sa_parse(a, sa)) {
                bool dup = false;
                for (const auto & o : out) {
                    dup = dup || sa_equal(o, sa);
                }
                if (!dup) {
                    out.push_back(sa);
                }
            }
        }
        return out;
    }

    static int kcp_output(const char * buf, int len, ikcpcb *, void * user) {
        auto * s = (p2p_session *) user;
        char pkt[1600];
        if (len + 1 > (int) sizeof(pkt)) {
            return -1;
        }
        pkt[0] = 0x01;
        memcpy(pkt + 1, buf, (size_t) len);
        s->ep->send_raw(s->peer, pkt, (size_t) len + 1);
        s->last_sent = p2p_now_ms();
        return 0;
    }

    // a packet of the peer arrived for this punch: from now on its address is the path. mtx must be held
    void establish(punch & pn, const sockaddr_in & from) {
        if (pn.session) {
            return;
        }
        auto s  = std::make_shared<p2p_session>();
        s->ep   = this;
        s->conv = pn.conv;
        s->peer = from;
        s->kcp  = ikcp_create(pn.conv, s.get());
        ikcp_setoutput(s->kcp, kcp_output);
        ikcp_setmtu(s->kcp, 1200);
        ikcp_wndsize(s->kcp, snd_wnd, rcv_wnd);
        ikcp_nodelay(s->kcp, 1, 10, 2, 1);
        s->kcp->stream = 1;
        s->last_recv = s->last_sent = p2p_now_ms();
        ikcp_update(s->kcp, (IUINT32) s->last_recv); // ikcp_flush does nothing before the first update
        agent_socket_t app;
        local_pair(app, s->local);
        pn.session = s;
        pn.app     = local_conn(app);
        sessions[pn.conv] = s;
        n_sessions++;
        std::thread([this, s]() { session_loop(s); }).detach();
        if (!pn.outgoing && accept_cb) {
            auto cb = accept_cb;
            tcp_conn_ptr c = pn.app;
            pn.app = nullptr;
            std::thread([cb, c]() { cb(c); }).detach();
        }
        cv.notify_all();
    }

    void recv_loop() {
        char buf[2048];
        while (true) {
            sockaddr_in from = {};
            socklen_t len = sizeof(from);
            const long n = (long) ::recvfrom(sock, buf, sizeof(buf), 0, (sockaddr *) &from, &len);
            if (n <= 0) {
                if (n < 0 && !socket_would_block() && !socket_interrupted()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue;
            }
            if (buf[0] == 0x01 && n >= 1 + 24) {
                on_data(buf + 1, n - 1, from);
            } else if (n > 5 && memcmp(buf, "AGP1 ", 5) == 0) {
                if (debug) {
                    fprintf(stderr, "p2p: <- %s %.*s\n", sa_str(from).c_str(), (int) std::min<long>(n, 120), buf);
                }
                on_ctl(std::string(buf, (size_t) n), from);
            }
        }
    }

    void on_data(const char * data, long n, const sockaddr_in & from) {
        IUINT32 conv;
        memcpy(&conv, data, 4); // KCP writes the conv little endian; both sides are little endian x86/arm
        p2p_session_ptr s;
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = sessions.find(conv);
            if (it == sessions.end()) {
                // data can overtake our PUNCHOK: a pending punch with this conv opens the session here
                for (auto & kv : punches) {
                    if (kv.second->conv == conv && !kv.second->session) {
                        establish(*kv.second, from);
                        it = sessions.find(conv);
                        break;
                    }
                }
                if (it == sessions.end()) {
                    return;
                }
            }
            s = it->second;
        }
        if (!sa_equal(s->peer, from)) {
            return;
        }
        std::lock_guard<std::mutex> lock(s->m);
        if (s->closed) {
            return;
        }
        s->last_recv = p2p_now_ms();
        ikcp_input(s->kcp, data, n);
        ikcp_flush(s->kcp); // the ACK leaves now, not at the next tick
        s->drain();
    }

    void on_ctl(const std::string & msg, const sockaddr_in & from) {
        std::vector<std::string> f;
        {
            std::stringstream ss(msg);
            std::string w;
            while (ss >> w) {
                f.push_back(w);
            }
        }
        if (f.size() < 2) {
            return;
        }
        const std::string & t = f[1];
        std::lock_guard<std::mutex> lock(mtx);
        try {
            on_ctl_locked(t, f, msg, from);
        } catch (const std::exception & e) {
            // a malformed datagram
            if (debug) {
                fprintf(stderr, "p2p: %s: %s\n", msg.substr(0, 40).c_str(), e.what());
            }
        }
    }

    void on_ctl_locked(const std::string & t, const std::vector<std::string> & f, const std::string & msg, const sockaddr_in & from) {
        if (t == "PUNCH" && f.size() == 3) {
            auto it = punches.find(f[2]);
            if (it != punches.end()) {
                send_ctl(from, "AGP1 PUNCHOK " + f[2]);
                establish(*it->second, from);
            }
        } else if (t == "PUNCHOK" && f.size() == 3) {
            auto it = punches.find(f[2]);
            if (it != punches.end()) {
                establish(*it->second, from);
            }
        } else if (t == "KA" && f.size() == 3) {
            auto it = sessions.find((IUINT32) std::stoul(f[2], nullptr, 16));
            if (it != sessions.end() && sa_equal(it->second->peer, from)) {
                it->second->last_recv = p2p_now_ms();
            }
        } else if (t == "FIN" && f.size() == 3) {
            auto it = sessions.find((IUINT32) std::stoul(f[2], nullptr, 16));
            if (it != sessions.end() && sa_equal(it->second->peer, from)) {
                it->second->peer_fin = true;
            }
        } else if (t == "PEER" && f.size() == 6) {
            auto it = requests.find(f[2]);
            if (it != requests.end() && !it->second->done) {
                it->second->nonce = f[3];
                it->second->cands = parse_cands(f[4], f[5]);
                it->second->done  = true;
                if (it->second->cands.empty() || f[3].size() != 32) {
                    it->second->error = "bad answer";
                }
                cv.notify_all();
            }
        } else if (t == "NOPEER" && f.size() >= 3) {
            auto it = requests.find(f[2]);
            if (it != requests.end()) {
                it->second->done  = true;
                it->second->error = f.size() > 3 ? f[3] : "no such node";
                cv.notify_all();
            }
        } else if (t == "INVITE" && f.size() == 5 && accept_cb && sa_equal(from, reg_server) && f[2].size() == 32) {
            if (punches.count(f[2]) == 0) {
                auto pn = std::make_shared<punch>();
                pn->nonce   = f[2];
                pn->conv    = conv_of(f[2]);
                pn->cands   = parse_cands(f[3], f[4]);
                pn->expires = p2p_now_ms() + 15000;
                punches[pn->nonce] = pn;
                send_punches(*pn);
            }
        } else if (t == "REGOK" && f.size() == 3 && sa_equal(from, reg_server)) {
            if (public_addr != f[2]) {
                public_addr = f[2];
                printf("agent-node: direct connections through the rendezvous server, seen as %s\n", f[2].c_str());
                fflush(stdout);
            }
        } else if (t == "REGERR" && sa_equal(from, reg_server)) {
            fprintf(stderr, "warning: p2p: the rendezvous server refused the registration: %s\n", msg.c_str() + 12);
        }
    }

    void timer_loop() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const int64_t now = p2p_now_ms();
            std::lock_guard<std::mutex> lock(mtx);
            if (reg_next >= 0 && now >= reg_next && !reg_line.empty()) {
                send_ctl(reg_server, reg_line);
                // often at first, until the server answered; then every 15 s keeps the NAT mapping open
                reg_next = now + (public_addr.empty() ? 1000 : 15000);
            }
            for (auto it = punches.begin(); it != punches.end();) {
                punch & pn = *it->second;
                if (now > pn.expires) {
                    it = punches.erase(it);
                    continue;
                }
                if (!pn.session && now >= pn.next_send) {
                    send_punches(pn);
                }
                ++it;
            }
            for (auto it = failed.begin(); it != failed.end();) {
                it = it->second < now ? failed.erase(it) : std::next(it);
            }
        }
    }

    // one per session: local socket -> KCP, timers, keepalive, end of the session
    void session_loop(p2p_session_ptr s) {
        char buf[65536];
        bool graceful = false;
        const std::string ka = "AGP1 KA " + hex32(s->conv);
        while (true) {
            int wait_ms;
            bool want_read, want_write;
            {
                std::lock_guard<std::mutex> lock(s->m);
                const IUINT32 now = (IUINT32) p2p_now_ms();
                wait_ms    = std::clamp((int) (ikcp_check(s->kcp, now) - now), 1, 10);
                // stop reading the application when KCP has a lot to send: back pressure instead of memory
                want_read  = !s->fin_sent && ikcp_waitsnd(s->kcp) < 4 * snd_wnd;
                want_write = !s->to_local.empty();
            }
#ifdef _WIN32
            WSAPOLLFD p = {};
            p.fd     = s->local;
            p.events = (short) ((want_read ? POLLRDNORM : 0) | (want_write ? POLLWRNORM : 0));
            const int r = p.events ? WSAPoll(&p, 1, wait_ms) : (std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms)), 0);
            const bool can_read  = r > 0 && (p.revents & (POLLRDNORM | POLLHUP | POLLERR));
#else
            pollfd p = {};
            p.fd     = s->local;
            p.events = (short) ((want_read ? POLLIN : 0) | (want_write ? POLLOUT : 0));
            const int r = ::poll(&p, 1, wait_ms);
            const bool can_read  = r > 0 && (p.revents & (POLLIN | POLLHUP | POLLERR));
#endif
            std::lock_guard<std::mutex> lock(s->m);
            const int64_t now = p2p_now_ms();
            bool local_eof = false;
            if (can_read && want_read) {
                while (ikcp_waitsnd(s->kcp) < 4 * snd_wnd) {
                    const long n = (long) ::recv(s->local, buf, sizeof(buf), 0);
                    if (n == 0) {
                        local_eof = true;
                        break;
                    }
                    if (n < 0) {
                        if (!socket_would_block() && !socket_interrupted()) {
                            local_eof = true;
                        }
                        break;
                    }
                    ikcp_send(s->kcp, buf, (int) n);
                    s->bytes_out += (uint64_t) n;
                }
                ikcp_flush(s->kcp);
            }
            ikcp_update(s->kcp, (IUINT32) now);
            s->drain();
            if (now - s->last_sent > 2500) {
                send_ctl(s->peer, ka);
                s->last_sent = now;
            }
            // the application closed its side: tell the peer once everything was delivered
            if (local_eof && !s->fin_sent) {
                s->fin_sent = true;
            }
            const bool delivered = s->fin_sent && ikcp_waitsnd(s->kcp) == 0;
            const bool dead = now - s->last_recv > dead_ms || s->kcp->state == (IUINT32) -1;
            // the peer closed: hand over what is left, then end of stream for the application
            const bool peer_done = s->peer_fin && s->to_local.empty() && ikcp_peeksize(s->kcp) < 0;
            if (delivered || dead || peer_done) {
                if (delivered) {
                    for (int i = 0; i < 3; i++) {
                        send_ctl(s->peer, "AGP1 FIN " + hex32(s->conv));
                    }
                }
                s->closed = true;
                if (peer_done) {
                    graceful = true;
                }
                break;
            }
        }
        if (graceful) {
            // EOF after the data; the application still may hold unread bytes: wait for it to close (at most 2 s)
#ifdef _WIN32
            shutdown(s->local, SD_SEND);
#else
            shutdown(s->local, SHUT_WR);
#endif
            for (int i = 0; i < 200; i++) {
                if (socket_wait(s->local, false, 10) > 0 && ::recv(s->local, buf, sizeof(buf), 0) == 0) {
                    break;
                }
            }
        }
        // closing the bridge side ends the application's reads; the session object goes with the last reference
#ifdef _WIN32
        shutdown(s->local, SD_BOTH);
#else
        shutdown(s->local, SHUT_RDWR);
#endif
        std::lock_guard<std::mutex> lock(mtx);
        sessions.erase(s->conv);
        for (auto & kv : punches) {
            if (kv.second->session == s) {
                kv.second->expires = 0;
            }
        }
    }

    static std::string hex32(IUINT32 v) {
        char b[9];
        snprintf(b, sizeof(b), "%08x", (unsigned) v);
        return b;
    }
};

// enables direct connections for tcp_connect("relay://..."); off with AGENT_P2P=off
inline void p2p_init() {
    const char * v = getenv("AGENT_P2P");
    if (v && std::string(v) == "off") {
        return;
    }
    if (!p2p_endpoint::get().ok()) {
        return;
    }
    p2p_dial = [](const std::string & server, const std::string & id, std::string & err) -> tcp_conn_ptr {
        return p2p_endpoint::get().connect(server, id, err);
    };
    std::atexit([]() { p2p_endpoint::get().goodbye(); });
}

} // namespace agent
