// access tokens and TLS identities of the nodes
//
// token: the coordinator knows the secret key of every node and issues, per session and node,
//   token = "<expiry unix seconds>:<hex HMAC-SHA256(key, "agent-node/1|<session hex>|<expiry>")>"
// the node accepts a session (open) or a previous node (attach) only with a valid token for that session
//
// identity: every node has its own key pair and self-signed certificate (a PEM file); the client and the previous
// node check the SHA-256 fingerprint of the certificate (pinning), the coordinator distributes the fingerprints
#pragma once

#include "net.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef AGENT_HAVE_TLS
#    include <openssl/crypto.h>
#    include <openssl/hmac.h>
#    include <openssl/sha.h>
#    include <openssl/pem.h>
#    include <openssl/rand.h>
#endif

namespace agent {

inline void require_tls(const char * what) {
#ifndef AGENT_HAVE_TLS
    throw std::runtime_error(std::string(what) + " needs a build with OpenSSL or BoringSSL");
#else
    (void) what;
#endif
}

inline std::vector<uint8_t> from_hex(const std::string & s) {
    if (s.size() % 2 != 0) {
        throw std::runtime_error("odd number of hex digits");
    }
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::runtime_error("not a hex digit");
    };
    std::vector<uint8_t> out(s.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = (uint8_t) (val(s[2 * i]) * 16 + val(s[2 * i + 1]));
    }
    return out;
}

inline uint64_t unix_now() {
    using namespace std::chrono;
    return (uint64_t) duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

inline std::string session_hex(uint64_t session) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) session);
    return buf;
}

#ifdef AGENT_HAVE_TLS

inline std::string random_key_hex(size_t n = 32) {
    std::vector<uint8_t> b(n);
    if (RAND_bytes(b.data(), (int) n) != 1) {
        throw std::runtime_error("RAND_bytes failed");
    }
    return to_hex(b.data(), n);
}

inline std::string hmac_hex(const std::vector<uint8_t> & key, const std::string & m) {
    uint8_t md[EVP_MAX_MD_SIZE];
    unsigned int n = 0;
    if (HMAC(EVP_sha256(), key.data(), (int) key.size(), (const uint8_t *) m.data(), m.size(), md, &n) == nullptr) {
        throw std::runtime_error("HMAC failed");
    }
    return to_hex(md, n);
}

inline std::string token_mac(const std::vector<uint8_t> & key, uint64_t session, uint64_t expiry) {
    return hmac_hex(key, "agent-node/1|" + session_hex(session) + "|" + std::to_string(expiry));
}

// signature of a request of the node to the coordinator (heartbeat): "<expiry>:<hex HMAC(key, "agent-coord/1|<node id>|<expiry>")>"
inline std::string node_signature(const std::vector<uint8_t> & key, const std::string & node_id, uint64_t ttl_s) {
    const uint64_t expiry = unix_now() + ttl_s;
    return std::to_string(expiry) + ":" + hmac_hex(key, "agent-coord/1|" + node_id + "|" + std::to_string(expiry));
}

inline std::string token_issue(const std::string & key_hex, uint64_t session, uint64_t ttl_s) {
    const uint64_t expiry = unix_now() + ttl_s;
    return std::to_string(expiry) + ":" + token_mac(from_hex(key_hex), session, expiry);
}

// "" when the token is valid, otherwise the reason
inline std::string token_check(const std::vector<uint8_t> & key, uint64_t session, const std::string & token) {
    const size_t p = token.find(':');
    if (token.empty()) {
        return "no access token";
    }
    if (p == std::string::npos) {
        return "malformed access token";
    }
    uint64_t expiry = 0;
    try {
        expiry = std::stoull(token.substr(0, p));
    } catch (const std::exception &) {
        return "malformed access token";
    }
    if (expiry < unix_now()) {
        return "access token expired";
    }
    const std::string want = token_mac(key, session, expiry);
    const std::string got  = token.substr(p + 1);
    if (got.size() != want.size() || CRYPTO_memcmp(got.data(), want.data(), want.size()) != 0) {
        return "invalid access token";
    }
    return "";
}

// the key pair and certificate of a node, loaded from a PEM file or created and saved there
struct tls_identity {
    EVP_PKEY * pkey = nullptr;
    X509 *     cert = nullptr;

    tls_identity() = default;
    tls_identity(const tls_identity &) = delete;
    tls_identity & operator=(const tls_identity &) = delete;
    ~tls_identity() {
        if (cert) X509_free(cert);
        if (pkey) EVP_PKEY_free(pkey);
    }

    std::string fingerprint() const { return cert_fingerprint(cert); }

    void load_or_create(const std::string & path) {
        if (!path.empty()) {
            if (BIO * b = BIO_new_file(path.c_str(), "r")) {
                pkey = PEM_read_bio_PrivateKey(b, nullptr, nullptr, nullptr);
                cert = PEM_read_bio_X509(b, nullptr, nullptr, nullptr);
                BIO_free(b);
                if (!pkey || !cert) {
                    throw std::runtime_error("cannot read the identity " + path + ": " + tls_error_string());
                }
                return;
            }
        }
        create();
        if (!path.empty()) {
            BIO * b = BIO_new_file(path.c_str(), "w");
            if (!b || PEM_write_bio_PrivateKey(b, pkey, nullptr, nullptr, 0, nullptr, nullptr) != 1 ||
                PEM_write_bio_X509(b, cert) != 1) {
                if (b) BIO_free(b);
                throw std::runtime_error("cannot write the identity " + path);
            }
            BIO_free(b);
        }
    }

    void create() {
        EVP_PKEY_CTX * pc = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr);
        if (!pc || EVP_PKEY_keygen_init(pc) != 1 || EVP_PKEY_CTX_set_ec_paramgen_curve_nid(pc, NID_X9_62_prime256v1) != 1 ||
            EVP_PKEY_keygen(pc, &pkey) != 1) {
            if (pc) EVP_PKEY_CTX_free(pc);
            throw std::runtime_error("cannot create a key: " + tls_error_string());
        }
        EVP_PKEY_CTX_free(pc);

        cert = X509_new();
        uint8_t serial[8];
        RAND_bytes(serial, sizeof(serial));
        uint64_t sn = 0;
        memcpy(&sn, serial, sizeof(sn));
        X509_set_version(cert, 2);
        ASN1_INTEGER_set(X509_get_serialNumber(cert), (long) (sn & 0x7fffffff));
        X509_gmtime_adj(X509_getm_notBefore(cert), -3600);
        X509_gmtime_adj(X509_getm_notAfter(cert), 10L * 365 * 24 * 3600);
        X509_set_pubkey(cert, pkey);
        X509_NAME * name = X509_get_subject_name(cert);
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, (const unsigned char *) "agent-node", -1, -1, 0);
        X509_set_issuer_name(cert, name);
        if (X509_sign(cert, pkey, EVP_sha256()) == 0) {
            throw std::runtime_error("cannot sign the certificate: " + tls_error_string());
        }
    }
};

inline void tls_ctx_common(SSL_CTX * ctx) {
    SSL_CTX_set_min_proto_version(ctx, TLS1_3_VERSION);
    // no session tickets: the read path of a connection never has to write
    SSL_CTX_set_options(ctx, SSL_OP_NO_TICKET);
    SSL_CTX_set_mode(ctx, SSL_MODE_ENABLE_PARTIAL_WRITE | SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
}

inline SSL_CTX * tls_server_ctx(const tls_identity & id) {
    SSL_CTX * ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) {
        throw std::runtime_error("SSL_CTX_new: " + tls_error_string());
    }
    tls_ctx_common(ctx);
    if (SSL_CTX_use_certificate(ctx, id.cert) != 1 || SSL_CTX_use_PrivateKey(ctx, id.pkey) != 1) {
        throw std::runtime_error("cannot use the identity: " + tls_error_string());
    }
    return ctx;
}

// the certificates are self-signed: the peer is checked by its fingerprint (tls_check_pin), not by a CA
inline SSL_CTX * tls_client_ctx() {
    SSL_CTX * ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        throw std::runtime_error("SSL_CTX_new: " + tls_error_string());
    }
    tls_ctx_common(ctx);
    SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
    return ctx;
}

#endif // AGENT_HAVE_TLS

// connects to a node, over TLS when client_ctx is set; pin: the expected certificate fingerprint ("" = any)
inline tcp_conn_ptr secure_connect(const std::string & addr, void * client_ctx, const std::string & pin, int timeout_ms = 5000) {
    tcp_conn_ptr c = tcp_connect(addr, timeout_ms);
    if (client_ctx == nullptr) {
        if (!pin.empty()) {
            throw std::runtime_error("a certificate pin for " + addr + " needs TLS");
        }
        return c;
    }
#ifdef AGENT_HAVE_TLS
    const std::string err = c->tls_handshake((SSL_CTX *) client_ctx, false, timeout_ms);
    if (!err.empty()) {
        throw std::runtime_error(addr + ": " + err);
    }
    if (!pin.empty()) {
        const std::string fp = c->peer_fingerprint();
        if (fp != pin) {
            throw std::runtime_error(addr + ": certificate fingerprint " + fp + " does not match the expected " + pin);
        }
    }
    return c;
#else
    throw std::runtime_error("TLS needs a build with OpenSSL or BoringSSL");
#endif
}

} // namespace agent
