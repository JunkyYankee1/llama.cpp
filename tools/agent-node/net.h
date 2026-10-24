// TCP connections for Windows and POSIX, optionally over TLS 1.3
//
// sockets are non-blocking: every wait goes through poll(), so reads can time out and a closed connection
// wakes its reader. With TLS all calls on the SSL object of a connection are serialized by a mutex, while the
// waits happen outside of it: one thread may read while others write
#pragma once

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN // no wincrypt.h
#    endif
#    include <winsock2.h>
#    include <ws2tcpip.h>
// wincrypt.h, if some header pulled it in anyway, defines names that OpenSSL / BoringSSL use
#    undef X509_NAME
#    undef X509_EXTENSIONS
#    undef X509_CERT_PAIR
#    undef PKCS7_ISSUER_AND_SERIAL
#    undef PKCS7_SIGNER_INFO
#    undef OCSP_REQUEST
#    undef OCSP_RESPONSE
typedef SOCKET agent_socket_t;
#    define AGENT_INVALID_SOCKET INVALID_SOCKET
#else
#    include <arpa/inet.h>
#    include <fcntl.h>
#    include <netdb.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <poll.h>
#    include <sys/socket.h>
#    include <unistd.h>
typedef int agent_socket_t;
#    define AGENT_INVALID_SOCKET (-1)
#endif

// cpp-httplib brings OpenSSL (system) or BoringSSL (LLAMA_BUILD_BORINGSSL) and defines this
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
#    define AGENT_HAVE_TLS 1
#    include <openssl/err.h>
#    include <openssl/evp.h>
#    include <openssl/ssl.h>
#    include <openssl/x509.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace agent {

inline void net_init() {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        throw std::runtime_error("WSAStartup failed");
    }
#else
    // a closed peer must not kill the process
    signal(SIGPIPE, SIG_IGN);
#endif
}

inline void socket_close(agent_socket_t s) {
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
}

inline void socket_set_nonblocking(agent_socket_t s) {
#ifdef _WIN32
    u_long one = 1;
    ioctlsocket(s, FIONBIO, &one);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif
}

inline bool socket_would_block() {
#ifdef _WIN32
    const int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS;
#endif
}

inline bool socket_interrupted() {
#ifdef _WIN32
    return false;
#else
    return errno == EINTR;
#endif
}

// waits until the socket can be read (or written); timeout_ms < 0 waits forever
// returns > 0 when ready (or closed / failed: the next call tells), 0 on timeout
inline int socket_wait(agent_socket_t s, bool for_write, int timeout_ms) {
#ifdef _WIN32
    WSAPOLLFD p = {};
    p.fd     = s;
    p.events = for_write ? POLLWRNORM : POLLRDNORM;
    const int r = WSAPoll(&p, 1, timeout_ms);
#else
    pollfd p = {};
    p.fd     = s;
    p.events = for_write ? POLLOUT : POLLIN;
    int r;
    do {
        r = ::poll(&p, 1, timeout_ms);
    } while (r < 0 && errno == EINTR);
#endif
    return r < 0 ? 1 : r;
}

// "host:port" -> host, port
inline void split_addr(const std::string & addr, std::string & host, int & port) {
    const size_t p = addr.rfind(':');
    if (p == std::string::npos) {
        throw std::runtime_error("address must be host:port: " + addr);
    }
    host = addr.substr(0, p);
    port = std::stoi(addr.substr(p + 1));
}

#ifdef AGENT_HAVE_TLS
inline std::string tls_error_string() {
    char buf[256] = "unknown error";
    const unsigned long e = ERR_get_error();
    if (e != 0) {
        ERR_error_string_n(e, buf, sizeof(buf));
    }
    return buf;
}

inline std::string to_hex(const uint8_t * data, size_t n) {
    static const char * digits = "0123456789abcdef";
    std::string s(n * 2, '0');
    for (size_t i = 0; i < n; i++) {
        s[2 * i]     = digits[data[i] >> 4];
        s[2 * i + 1] = digits[data[i] & 15];
    }
    return s;
}

// SHA-256 of the DER certificate, lowercase hex
inline std::string cert_fingerprint(X509 * cert) {
    uint8_t md[EVP_MAX_MD_SIZE];
    unsigned int n = 0;
    if (cert == nullptr || X509_digest(cert, EVP_sha256(), md, &n) != 1) {
        return "";
    }
    return to_hex(md, n);
}
#endif

class tcp_conn {
  public:
    explicit tcp_conn(agent_socket_t s) : s(s) {
        // small frames must leave at once: one hidden state per token is the whole point
        int one = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
        socket_set_nonblocking(s);
    }
    ~tcp_conn() {
#ifdef AGENT_HAVE_TLS
        if (ssl) {
            SSL_free(ssl);
        }
#endif
        socket_close(s);
    }

    tcp_conn(const tcp_conn &) = delete;
    tcp_conn & operator=(const tcp_conn &) = delete;

    // stops the connection: blocked reads and writes return false, the socket is freed with the object
    void close() {
        if (!closed.exchange(true)) {
#ifdef _WIN32
            shutdown(s, SD_BOTH);
#else
            shutdown(s, SHUT_RDWR);
#endif
        }
    }
    bool is_closed() const { return closed; }

#ifdef AGENT_HAVE_TLS
    // TLS 1.3 handshake on the connected socket, returns an error message or ""
    std::string tls_handshake(SSL_CTX * ctx, bool server, int timeout_ms = 10000) {
        ssl = SSL_new(ctx);
        if (ssl == nullptr || SSL_set_fd(ssl, (int) s) != 1) {
            return "SSL_new: " + tls_error_string();
        }
        while (true) {
            int r, err;
            {
                std::lock_guard<std::mutex> lock(io);
                ERR_clear_error();
                r   = server ? SSL_accept(ssl) : SSL_connect(ssl);
                err = r == 1 ? SSL_ERROR_NONE : SSL_get_error(ssl, r);
            }
            if (r == 1) {
                return "";
            }
            if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
                if (socket_wait(s, err == SSL_ERROR_WANT_WRITE, timeout_ms) == 0) {
                    return "TLS handshake timed out";
                }
                continue;
            }
            return "TLS handshake failed: " + tls_error_string();
        }
    }

    std::string peer_fingerprint() {
        std::lock_guard<std::mutex> lock(io);
        X509 * cert = ssl ? SSL_get_peer_certificate(ssl) : nullptr;
        std::string fp = cert_fingerprint(cert);
        if (cert) {
            X509_free(cert);
        }
        return fp;
    }
#endif

    bool is_tls() const {
#ifdef AGENT_HAVE_TLS
        return ssl != nullptr;
#else
        return false;
#endif
    }

    bool send_all(const void * data, size_t n) {
        const char * p = (const char *) data;
        while (n > 0) {
            if (closed) {
                return false;
            }
            const int chunk = (int) std::min<size_t>(n, 1 << 20);
            long r;
            bool want_read = false;
#ifdef AGENT_HAVE_TLS
            if (ssl) {
                int err;
                {
                    std::lock_guard<std::mutex> lock(io);
                    ERR_clear_error();
                    r   = SSL_write(ssl, p, chunk);
                    err = r > 0 ? SSL_ERROR_NONE : SSL_get_error(ssl, (int) r);
                }
                if (r <= 0) {
                    if (err != SSL_ERROR_WANT_WRITE && err != SSL_ERROR_WANT_READ) {
                        return false;
                    }
                    want_read = err == SSL_ERROR_WANT_READ;
                    r = 0;
                }
            } else
#endif
            {
                r = (long) ::send(s, p, chunk, 0);
                if (r < 0) {
                    if (!socket_would_block() && !socket_interrupted()) {
                        return false;
                    }
                    r = 0;
                }
            }
            if (r == 0) {
                socket_wait(s, !want_read, -1);
                continue;
            }
            p += r;
            n -= (size_t) r;
            bytes_sent += (uint64_t) r;
        }
        return true;
    }

    // waits at most recv_timeout_ms for every chunk of data (< 0: forever); sets timed_out
    bool recv_all(void * data, size_t n) {
        char * p = (char *) data;
        while (n > 0) {
            if (closed) {
                return false;
            }
            const int chunk = (int) std::min<size_t>(n, 1 << 20);
            long r;
            bool want_write = false;
#ifdef AGENT_HAVE_TLS
            if (ssl) {
                int err;
                {
                    std::lock_guard<std::mutex> lock(io);
                    ERR_clear_error();
                    r   = SSL_read(ssl, p, chunk);
                    err = r > 0 ? SSL_ERROR_NONE : SSL_get_error(ssl, (int) r);
                }
                if (r <= 0) {
                    if (err != SSL_ERROR_WANT_READ && err != SSL_ERROR_WANT_WRITE) {
                        return false;
                    }
                    want_write = err == SSL_ERROR_WANT_WRITE;
                    r = -1;
                }
            } else
#endif
            {
                r = (long) ::recv(s, p, chunk, 0);
                if (r == 0) {
                    return false; // closed by the peer
                }
                if (r < 0) {
                    if (!socket_would_block() && !socket_interrupted()) {
                        return false;
                    }
                }
            }
            if (r < 0) {
                if (socket_wait(s, want_write, recv_timeout_ms) == 0) {
                    timed_out = true;
                    return false;
                }
                continue;
            }
            p += r;
            n -= (size_t) r;
            bytes_recv += (uint64_t) r;
        }
        return true;
    }

    std::mutex send_mutex; // frames from several threads must not interleave
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_recv{0};
    int  recv_timeout_ms = -1;
    bool timed_out       = false;

  private:
    agent_socket_t    s;
    std::atomic<bool> closed{false};
    std::mutex        io; // calls on the SSL object
#ifdef AGENT_HAVE_TLS
    SSL * ssl = nullptr;
#endif
};

using tcp_conn_ptr = std::shared_ptr<tcp_conn>;

// connects within timeout_ms
inline tcp_conn_ptr tcp_connect(const std::string & addr, int timeout_ms = 5000) {
    std::string host;
    int port = 0;
    split_addr(addr, host, port);

    addrinfo hints = {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) {
        throw std::runtime_error("cannot resolve " + addr);
    }
    agent_socket_t s = AGENT_INVALID_SOCKET;
    for (addrinfo * ai = res; ai != nullptr; ai = ai->ai_next) {
        s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == AGENT_INVALID_SOCKET) {
            continue;
        }
        socket_set_nonblocking(s);
        bool ok = ::connect(s, ai->ai_addr, (int) ai->ai_addrlen) == 0;
        if (!ok && socket_would_block() && socket_wait(s, true, timeout_ms) > 0) {
            int err = 0;
            socklen_t len = sizeof(err);
            ok = getsockopt(s, SOL_SOCKET, SO_ERROR, (char *) &err, &len) == 0 && err == 0;
        }
        if (ok) {
            break;
        }
        socket_close(s);
        s = AGENT_INVALID_SOCKET;
    }
    freeaddrinfo(res);
    if (s == AGENT_INVALID_SOCKET) {
        throw std::runtime_error("cannot connect to " + addr);
    }
    return std::make_shared<tcp_conn>(s);
}

class tcp_listener {
  public:
    tcp_listener(const std::string & host, int port) {
        s = ::socket(AF_INET, SOCK_STREAM, 0);
        if (s == AGENT_INVALID_SOCKET) {
            throw std::runtime_error("cannot create socket");
        }
        int one = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));

        sockaddr_in sa = {};
        sa.sin_family = AF_INET;
        sa.sin_port   = htons((uint16_t) port);
        if (inet_pton(AF_INET, host.c_str(), &sa.sin_addr) != 1) {
            socket_close(s);
            throw std::runtime_error("invalid IPv4 address " + host);
        }
        if (::bind(s, (sockaddr *) &sa, sizeof(sa)) != 0 || ::listen(s, 16) != 0) {
            socket_close(s);
            throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
        }
    }
    ~tcp_listener() { socket_close(s); }

    tcp_conn_ptr accept() {
        agent_socket_t c = ::accept(s, nullptr, nullptr);
        if (c == AGENT_INVALID_SOCKET) {
            return nullptr;
        }
        return std::make_shared<tcp_conn>(c);
    }

  private:
    agent_socket_t s;
};

} // namespace agent
