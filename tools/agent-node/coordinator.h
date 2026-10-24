// talking to the coordinator over HTTP(S), see server/internal/coord in the agent repository
//
// node:   POST /v1/nodes                 registration: address, certificate fingerprint, secret key, layer range
//         POST /v1/nodes/{id}/heartbeat  every few seconds, signed with the key, with the number of open sessions
// client: POST /v1/chains                a chain for one session: addresses, pins and access tokens, plus spares
#pragma once

#include "secure.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
// wincrypt.h again (through httplib.h)
#    undef X509_NAME
#    undef X509_EXTENSIONS
#    undef X509_CERT_PAIR
#    undef PKCS7_ISSUER_AND_SERIAL
#    undef PKCS7_SIGNER_INFO
#    undef OCSP_REQUEST
#    undef OCSP_RESPONSE
#endif

#include <atomic>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace agent {

using json = nlohmann::ordered_json;

struct http_reply {
    int  status = 0;
    json body;

    std::string error() const {
        if (body.is_object() && body.contains("error") && body["error"].is_string()) {
            return body["error"].get<std::string>();
        }
        return "HTTP " + std::to_string(status);
    }
};

// url: http(s)://host[:port][/prefix]
inline http_reply coord_post(const std::string & url, const std::string & path, const json & body,
                             const httplib::Headers & headers = {}) {
    const size_t p = url.find("://");
    if (p == std::string::npos) {
        throw std::runtime_error("the coordinator URL must start with http:// or https://");
    }
    const size_t s = url.find('/', p + 3);
    const std::string base = s == std::string::npos ? url : url.substr(0, s);
    std::string prefix     = s == std::string::npos ? "" : url.substr(s);
    while (!prefix.empty() && prefix.back() == '/') {
        prefix.pop_back();
    }
    httplib::Client cli(base);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(15);
    cli.set_write_timeout(15);
    auto res = cli.Post(prefix + path, headers, body.dump(), "application/json");
    if (!res) {
        throw std::runtime_error("coordinator " + base + ": " + httplib::to_string(res.error()));
    }
    http_reply r;
    r.status = res->status;
    r.body   = res->body.empty() ? json::object() : json::parse(res->body, nullptr, false);
    return r;
}

// keeps the node registered: registers, then sends heartbeats; registers again when the coordinator forgot the node
// (a restart of a coordinator with the in-memory registry), retries with a growing pause when it is unreachable
struct coord_link {
    std::string              url;
    json                     reg;      // the registration body
    std::vector<uint8_t>     key;
    std::function<int()>     sessions; // open sessions now
    std::atomic<bool>        stop{ false };

    void run() {
        std::string id;
        int interval_s = 10;
        int backoff_s  = 1;
        bool warned    = false;
        while (!stop) {
            int wait_s = interval_s;
            try {
                if (id.empty()) {
                    const http_reply r = coord_post(url, "/v1/nodes", reg);
                    if (r.status != 200) {
                        throw std::runtime_error("registration refused: " + r.error());
                    }
                    id         = r.body.value("id", "");
                    interval_s = std::max(1, r.body.value("heartbeat_s", 10));
                    wait_s     = interval_s;
                    backoff_s  = 1;
                    warned     = false;
                    // the scripts may wait for this line
                    printf("agent-node registered with the coordinator as %s (%s)\n", id.c_str(), r.body.value("addr", "").c_str());
                    fflush(stdout);
                } else {
#ifdef AGENT_HAVE_TLS
                    const httplib::Headers h = { { "Authorization", "Node " + node_signature(key, id, 60) } };
#else
                    const httplib::Headers h;
#endif
                    const http_reply r = coord_post(url, "/v1/nodes/" + id + "/heartbeat", json{ { "sessions", sessions() } }, h);
                    if (r.status == 404) {
                        fprintf(stderr, "agent-node: the coordinator does not know node %s, registering again\n", id.c_str());
                        id.clear();
                        wait_s = 0;
                    } else if (r.status != 204 && r.status != 200) {
                        throw std::runtime_error("heartbeat refused: " + r.error());
                    }
                    backoff_s = 1;
                    warned    = false;
                }
            } catch (const std::exception & e) {
                if (!warned) {
                    fprintf(stderr, "agent-node: coordinator: %s (retrying)\n", e.what());
                    warned = true;
                }
                wait_s    = backoff_s;
                backoff_s = std::min(backoff_s * 2, 30);
            }
            for (int i = 0; i < wait_s * 10 && !stop; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
};

} // namespace agent
