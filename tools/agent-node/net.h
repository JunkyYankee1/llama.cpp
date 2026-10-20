// minimal blocking TCP sockets for Windows and POSIX
#pragma once

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <winsock2.h>
#    include <ws2tcpip.h>
typedef SOCKET agent_socket_t;
#    define AGENT_INVALID_SOCKET INVALID_SOCKET
#else
#    include <arpa/inet.h>
#    include <netdb.h>
#    include <netinet/in.h>
#    include <netinet/tcp.h>
#    include <sys/socket.h>
#    include <unistd.h>
typedef int agent_socket_t;
#    define AGENT_INVALID_SOCKET (-1)
#endif

#include <algorithm>
#include <atomic>
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

// "host:port" -> host, port
inline void split_addr(const std::string & addr, std::string & host, int & port) {
    const size_t p = addr.rfind(':');
    if (p == std::string::npos) {
        throw std::runtime_error("address must be host:port: " + addr);
    }
    host = addr.substr(0, p);
    port = std::stoi(addr.substr(p + 1));
}

class tcp_conn {
  public:
    explicit tcp_conn(agent_socket_t s) : s(s) {
        // small frames must leave at once: one hidden state per token is the whole point
        int one = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
    }
    ~tcp_conn() { close(); }

    tcp_conn(const tcp_conn &) = delete;
    tcp_conn & operator=(const tcp_conn &) = delete;

    void close() {
        agent_socket_t old = s.exchange(AGENT_INVALID_SOCKET);
        if (old != AGENT_INVALID_SOCKET) {
#ifdef _WIN32
            shutdown(old, SD_BOTH);
#else
            shutdown(old, SHUT_RDWR);
#endif
            socket_close(old);
        }
    }

    bool send_all(const void * data, size_t n) {
        const char * p = (const char *) data;
        while (n > 0) {
            const int chunk = (int) std::min<size_t>(n, 1 << 30);
            const auto r = ::send(s, p, chunk, 0);
            if (r <= 0) {
                return false;
            }
            p += r;
            n -= (size_t) r;
            bytes_sent += (uint64_t) r;
        }
        return true;
    }

    bool recv_all(void * data, size_t n) {
        char * p = (char *) data;
        while (n > 0) {
            const int chunk = (int) std::min<size_t>(n, 1 << 30);
            const auto r = ::recv(s, p, chunk, 0);
            if (r <= 0) {
                return false;
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

  private:
    std::atomic<agent_socket_t> s;
};

using tcp_conn_ptr = std::shared_ptr<tcp_conn>;

inline tcp_conn_ptr tcp_connect(const std::string & addr) {
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
        if (::connect(s, ai->ai_addr, (int) ai->ai_addrlen) == 0) {
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
