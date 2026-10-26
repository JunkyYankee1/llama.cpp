// a slice of a model: one GGUF file with only the weights of a layer range, downloaded from the full model by HTTP ranges
//
// a node of a chain needs only its layers (plus the token embeddings on the first part and the output head on the last),
// so it does not download the whole model: it reads the GGUF header of the full file (a few MB), picks the tensors of its
// range and fetches only their bytes. The slice keeps all the metadata of the model (vocabulary, hyperparameters) and adds
//   agent.model_key    the identity of the full model (the coordinator matches nodes by it)
//   agent.layer_start  the layer range the slice holds
//   agent.layer_end
// llama loads a slice like the full file with the same layer range: the tensors outside the range are skipped by name.
// The download resumes: the layout of the slice depends only on the header and the range, so the bytes already in the
// .part file stay valid.
#pragma once

#include "http.h" // common: an httplib client that follows redirects and honors HTTPS_PROXY

#include "ggml.h"
#include "gguf.h"

#ifdef _WIN32
#    undef X509_NAME
#    undef X509_EXTENSIONS
#    undef OCSP_REQUEST
#    undef OCSP_RESPONSE
#endif

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace agent {

// file paths are UTF-8 (on Windows the narrow API takes the ANSI code page)
inline std::filesystem::path slice_path(const std::string & path) {
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(path.begin(), path.end()));
#else
    return std::filesystem::u8path(path);
#endif
}

inline FILE * slice_fopen(const std::string & path, const char * mode) {
#ifdef _WIN32
    const std::wstring wmode(mode, mode + strlen(mode));
    return _wfopen(slice_path(path).c_str(), wmode.c_str());
#else
    return fopen(path.c_str(), mode);
#endif
}

//
// where the bytes of the full model come from: a URL (HTTP ranges) or a local file
//

struct byte_source {
    virtual ~byte_source() = default;
    // total size of the file
    virtual uint64_t size() = 0;
    // reads [off, off + n) and passes the bytes to sink in order; sink returns false to stop
    virtual void read(uint64_t off, uint64_t n, const std::function<bool(const char *, size_t)> & sink) = 0;
    virtual std::string name() const = 0;
};

struct file_source : byte_source {
    std::string path;
    FILE *      f = nullptr;
    uint64_t    n_bytes = 0;

    explicit file_source(const std::string & path) : path(path) {
        f = slice_fopen(path, "rb");
        if (!f) {
            throw std::runtime_error("cannot open " + path);
        }
        n_bytes = (uint64_t) std::filesystem::file_size(slice_path(path));
    }
    ~file_source() override {
        if (f) {
            fclose(f);
        }
    }
    uint64_t size() override { return n_bytes; }
    std::string name() const override { return path; }

    void read(uint64_t off, uint64_t n, const std::function<bool(const char *, size_t)> & sink) override {
#ifdef _WIN32
        _fseeki64(f, (int64_t) off, SEEK_SET);
#else
        fseeko(f, (off_t) off, SEEK_SET);
#endif
        std::vector<char> buf(1 << 20);
        while (n > 0) {
            const size_t want = (size_t) std::min<uint64_t>(n, buf.size());
            const size_t got  = fread(buf.data(), 1, want, f);
            if (got != want) {
                throw std::runtime_error("short read of " + path);
            }
            if (!sink(buf.data(), got)) {
                return;
            }
            n -= got;
        }
    }
};

struct http_source : byte_source {
    std::string url;
    uint64_t    n_bytes = 0;

    explicit http_source(const std::string & url) : url(url) {}
    std::string name() const override { return url; }

    uint64_t size() override {
        if (n_bytes == 0) {
            // the total size comes with the first range: "Content-Range: bytes 0-0/12345"
            read(0, 1, [](const char *, size_t) { return true; });
        }
        return n_bytes;
    }

    void read(uint64_t off, uint64_t n, const std::function<bool(const char *, size_t)> & sink) override {
        if (n == 0) {
            return;
        }
        auto [cli, parts] = common_http_client(url);
        cli.set_connection_timeout(15);
        cli.set_read_timeout(60);
        httplib::Headers headers = {
            { "Range", "bytes=" + std::to_string(off) + "-" + std::to_string(off + n - 1) },
            { "User-Agent", "agent-node" },
        };
        uint64_t got = 0;
        bool stopped = false;
        std::string bad;
        auto res = cli.Get(parts.path, headers,
            [&](const httplib::Response & r) {
                if (r.status == 206) {
                    const std::string cr = r.get_header_value("Content-Range");
                    const size_t slash = cr.rfind('/');
                    if (slash != std::string::npos && cr.compare(slash + 1, std::string::npos, "*") != 0) {
                        n_bytes = std::stoull(cr.substr(slash + 1));
                    }
                    return true;
                }
                if (r.status == 200 && off == 0) {
                    // a server without ranges: the whole file from the start, cut below
                    if (r.has_header("Content-Length")) {
                        n_bytes = std::stoull(r.get_header_value("Content-Length"));
                    }
                    return true;
                }
                bad = "HTTP " + std::to_string(r.status);
                return false;
            },
            [&](const char * data, size_t len) {
                const size_t take = (size_t) std::min<uint64_t>(len, n - got);
                if (take > 0 && !sink(data, take)) {
                    stopped = true;
                    return false;
                }
                got += take;
                return got < n;
            });
        if (stopped || got == n) {
            return;
        }
        if (!bad.empty()) {
            throw std::runtime_error(common_http_show_masked_url(parts) + ": " + bad);
        }
        if (!res) {
            throw std::runtime_error(common_http_show_masked_url(parts) + ": " + httplib::to_string(res.error()));
        }
        throw std::runtime_error(common_http_show_masked_url(parts) + ": the connection closed after " + std::to_string(got) +
            " of " + std::to_string(n) + " bytes");
    }
};

inline bool is_url(const std::string & s) {
    return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

inline std::unique_ptr<byte_source> open_source(const std::string & where) {
    if (is_url(where)) {
        return std::make_unique<http_source>(where);
    }
    return std::make_unique<file_source>(where);
}

// the files of a split model: "name-00001-of-00004.gguf" -> all four; anything else -> itself
inline std::vector<std::string> split_files(const std::string & first) {
    const std::string tail = ".gguf";
    // "-00001-of-00004.gguf" is 20 characters
    if (first.size() < 20 + 1 || first.compare(first.size() - tail.size(), tail.size(), tail) != 0) {
        return { first };
    }
    const size_t p = first.size() - 20;
    if (first[p] != '-' || first.compare(p + 6, 4, "-of-") != 0) {
        return { first };
    }
    int no = 0, count = 0;
    if (sscanf(first.c_str() + p, "-%5d-of-%5d.gguf", &no, &count) != 2 || no != 1 || count < 1) {
        return { first };
    }
    std::vector<std::string> out;
    for (int i = 1; i <= count; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, count);
        out.push_back(first.substr(0, p) + buf);
    }
    return out;
}

//
// the header of the full model
//

struct model_file {
    std::unique_ptr<byte_source> src;
    gguf_context *               g    = nullptr;
    ggml_context *               meta = nullptr; // tensors without data, for gguf_add_tensor

    model_file() = default;
    model_file(const model_file &) = delete;
    model_file & operator=(const model_file &) = delete;
    model_file(model_file && o) noexcept : src(std::move(o.src)), g(o.g), meta(o.meta) { o.g = nullptr; o.meta = nullptr; }
    ~model_file() {
        if (g) {
            gguf_free(g);
        }
        if (meta) {
            ggml_free(meta);
        }
    }
};

// reads only the header: the bytes are fetched in growing chunks while gguf asks for more
inline model_file read_header(const std::string & where) {
    model_file mf;
    mf.src = open_source(where);
    struct cache_t {
        byte_source *     src;
        std::vector<char> buf;
        uint64_t          total;
        std::string       err;
    } cache{ mf.src.get(), {}, mf.src->size(), {} };

    auto cb = [](void * ud, void * out, uint64_t offset, size_t len) -> size_t {
        auto & c = *(cache_t *) ud;
        const uint64_t end = std::min<uint64_t>(offset + len, c.total);
        if (end > c.buf.size()) {
            // grow by doubling, at least 4 MB per request: a header with a large vocabulary is tens of MB
            const uint64_t want = std::min<uint64_t>(c.total, std::max<uint64_t>({ end, (uint64_t) c.buf.size() * 2, 4u << 20 }));
            c.buf.reserve((size_t) want);
            // a broken connection keeps what came, the next attempt continues from there
            for (int attempt = 0; c.buf.size() < want; attempt++) {
                try {
                    c.src->read(c.buf.size(), want - c.buf.size(), [&](const char * d, size_t n) {
                        c.buf.insert(c.buf.end(), d, d + n);
                        return true;
                    });
                } catch (const std::exception & e) {
                    c.err = e.what();
                    if (attempt >= 8) {
                        return 0;
                    }
                    std::this_thread::sleep_for(std::chrono::seconds(1 + attempt));
                }
            }
        }
        if (offset >= c.buf.size()) {
            return 0;
        }
        const size_t n = (size_t) std::min<uint64_t>(len, c.buf.size() - offset);
        memcpy(out, c.buf.data() + offset, n);
        return n;
    };
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &mf.meta };
    mf.g = gguf_init_from_callback(cb, &cache, 0, cache.total, gp);
    if (mf.g == nullptr) {
        throw std::runtime_error("cannot read the GGUF header of " + mf.src->name() + (cache.err.empty() ? "" : ": " + cache.err));
    }
    return mf;
}

inline std::string gguf_str(const gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? gguf_get_val_str(g, id) : "";
}

inline long long gguf_int(const gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        return -1;
    }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (long long) gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (long long) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT16: return (long long) gguf_get_val_u16(g, id);
        case GGUF_TYPE_UINT64: return (long long) gguf_get_val_u64(g, id);
        default:               return -1;
    }
}

// identity of the model for the coordinator: name and shape from the metadata plus a hash of the tensor names, types and sizes
// (differs between quantizations); a slice carries the identity of its full model
inline std::string model_key_of(const std::vector<const gguf_context *> & files) {
    const gguf_context * g0 = files.at(0);
    const std::string stored = gguf_str(g0, "agent.model_key");
    if (!stored.empty()) {
        return stored;
    }
    uint64_t h = 1469598103934665603ull; // FNV-1a
    auto mix = [&](const void * p, size_t n) {
        for (size_t i = 0; i < n; i++) {
            h = (h ^ ((const uint8_t *) p)[i]) * 1099511628211ull;
        }
    };
    for (const gguf_context * g : files) {
        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            const char *   name = gguf_get_tensor_name(g, i);
            const int32_t  type = (int32_t) gguf_get_tensor_type(g, i);
            const uint64_t size = (uint64_t) gguf_get_tensor_size(g, i);
            mix(name, strlen(name) + 1);
            mix(&type, sizeof(type));
            mix(&size, sizeof(size));
        }
    }
    const std::string arch = gguf_str(g0, "general.architecture");
    char buf[512];
    snprintf(buf, sizeof(buf), "%s %s L%lld E%lld #%016llx", arch.c_str(), gguf_str(g0, "general.name").c_str(),
        gguf_int(g0, arch + ".block_count"), gguf_int(g0, arch + ".embedding_length"), (unsigned long long) h);
    return buf;
}

// which part of the model a tensor belongs to
struct tensor_role {
    int  layer  = -1;    // blk.N.*
    bool embd   = false; // token_embd.*
    bool output = false; // output.*, output_norm.*
};

inline tensor_role role_of(const char * name) {
    tensor_role r;
    int il = -1;
    if (sscanf(name, "blk.%d.", &il) == 1) {
        r.layer = il;
    } else if (strncmp(name, "token_embd.", 11) == 0) {
        r.embd = true;
    } else if (strncmp(name, "output.", 7) == 0 || strncmp(name, "output_norm.", 12) == 0) {
        r.output = true;
    }
    return r;
}

// the whole model: header(s) of its file(s)
struct model_source {
    std::vector<model_file> files;
    std::string             key;
    int32_t                 n_layer = 0;
    bool                    tied    = false; // no output.weight: the output head reuses token_embd
    std::vector<uint64_t>   layer_bytes;     // bytes of blk.N.* per layer
    uint64_t                embd_bytes   = 0; // token_embd
    uint64_t                output_bytes = 0; // output head and norm (token_embd again when tied)
    uint64_t                other_bytes  = 0; // the rest, every part loads it (e.g. rope frequencies)

    // where: a URL or a path of the full model; a split model by its first file
    void open(const std::string & where) {
        for (const std::string & f : split_files(where)) {
            files.push_back(read_header(f));
        }
        std::vector<const gguf_context *> gs;
        for (const auto & f : files) {
            gs.push_back(f.g);
        }
        key = model_key_of(gs);
        const gguf_context * g0 = files[0].g;
        if (!gguf_str(g0, "agent.model_key").empty()) {
            throw std::runtime_error(where + " is a slice, not a full model");
        }
        const std::string arch = gguf_str(g0, "general.architecture");
        n_layer = (int32_t) gguf_int(g0, arch + ".block_count");
        if (n_layer <= 0) {
            throw std::runtime_error(where + ": no " + arch + ".block_count in the metadata");
        }
        layer_bytes.assign((size_t) n_layer, 0);
        bool has_output = false;
        for (const auto & f : files) {
            for (int64_t i = 0; i < gguf_get_n_tensors(f.g); i++) {
                const char *   name = gguf_get_tensor_name(f.g, i);
                const uint64_t size = (uint64_t) gguf_get_tensor_size(f.g, i);
                const tensor_role r = role_of(name);
                if (r.layer >= 0 && r.layer < n_layer) {
                    layer_bytes[(size_t) r.layer] += size;
                } else if (r.embd) {
                    embd_bytes += size;
                } else if (r.output) {
                    output_bytes += size;
                    has_output = has_output || strcmp(name, "output.weight") == 0;
                } else {
                    other_bytes += size;
                }
            }
        }
        tied = !has_output;
        if (tied) {
            output_bytes += embd_bytes;
        }
    }

    // is the tensor in the slice [first, last)
    bool wanted(const char * name, int32_t first, int32_t last) const {
        const tensor_role r = role_of(name);
        if (r.layer >= 0) {
            return r.layer >= first && r.layer < last;
        }
        if (r.embd) {
            return first == 0 || (last == n_layer && tied);
        }
        if (r.output) {
            return last == n_layer;
        }
        return first < last; // the rest goes with every part (none in a vocabulary-only slice)
    }

    // bytes of the weights of a slice
    uint64_t slice_bytes(int32_t first, int32_t last) const {
        uint64_t n = 0;
        for (const auto & f : files) {
            for (int64_t i = 0; i < gguf_get_n_tensors(f.g); i++) {
                if (wanted(gguf_get_tensor_name(f.g, i), first, last)) {
                    n += (uint64_t) gguf_get_tensor_size(f.g, i);
                }
            }
        }
        return n;
    }
};

// the metadata of a GGUF file on disk (a slice or a full model), or null
struct gguf_ptr_deleter {
    void operator()(gguf_context * g) const { gguf_free(g); }
};
using gguf_ptr = std::unique_ptr<gguf_context, gguf_ptr_deleter>;

inline gguf_ptr read_local_header(const std::string & path) {
    FILE * f = slice_fopen(path, "rb");
    if (!f) {
        return nullptr;
    }
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file_ptr(f, gp);
    fclose(f);
    return gguf_ptr(g);
}

//
// writing a slice
//

struct slice_progress {
    uint64_t done  = 0; // bytes of the slice on disk
    uint64_t total = 0;
};

// writes the slice [first, last) of the model to path (through path + ".part", resuming it);
// first == last: a slice with the metadata only, enough for the vocabulary of a client.
// on_progress is called every few MB; attempts: how many times a broken download is retried
inline void write_slice(const model_source & ms, int32_t first, int32_t last, const std::string & path,
                        const std::function<void(const slice_progress &)> & on_progress = nullptr, int attempts = 30) {
    if (first < 0 || last > ms.n_layer || first > last) {
        throw std::runtime_error("bad layer range " + std::to_string(first) + ":" + std::to_string(last));
    }
    // metadata: everything of the model except the split keys and the alignment (the slice uses the default)
    gguf_context * out = gguf_init_empty();
    std::unique_ptr<gguf_context, gguf_ptr_deleter> out_guard(out);
    gguf_set_kv(out, ms.files[0].g);
    for (const char * k : { "split.no", "split.count", "split.tensors.count", "general.alignment" }) {
        gguf_remove_key(out, k);
    }
    gguf_set_val_str(out, "agent.model_key", ms.key.c_str());
    gguf_set_val_u32(out, "agent.layer_start", (uint32_t) first);
    gguf_set_val_u32(out, "agent.layer_end", (uint32_t) last);

    struct piece {
        size_t   file;     // which file of the model
        uint64_t src_off;  // absolute offset in that file
        uint64_t out_off;  // absolute offset in the slice
        uint64_t size;
    };
    std::vector<piece> pieces;
    for (size_t fi = 0; fi < ms.files.size(); fi++) {
        const model_file & f = ms.files[fi];
        for (int64_t i = 0; i < gguf_get_n_tensors(f.g); i++) {
            const char * name = gguf_get_tensor_name(f.g, i);
            if (!ms.wanted(name, first, last)) {
                continue;
            }
            ggml_tensor * t = ggml_get_tensor(f.meta, name);
            if (t == nullptr) {
                throw std::runtime_error(std::string("no metadata of tensor ") + name);
            }
            gguf_add_tensor(out, t);
            pieces.push_back({ fi, gguf_get_data_offset(f.g) + gguf_get_tensor_offset(f.g, i), 0, (uint64_t) gguf_get_tensor_size(f.g, i) });
        }
    }
    const uint64_t meta_size = gguf_get_meta_size(out);
    std::vector<char> meta((size_t) meta_size);
    gguf_get_meta_data(out, meta.data());
    for (int64_t i = 0; i < gguf_get_n_tensors(out); i++) {
        pieces[(size_t) i].out_off = meta_size + gguf_get_tensor_offset(out, i);
    }
    uint64_t total = meta_size;
    if (!pieces.empty()) {
        total = pieces.back().out_off + pieces.back().size;
    }

    const std::string part = path + ".part";
    // a .part with other metadata (another range, another model) cannot be continued
    uint64_t have = 0;
    if (std::filesystem::exists(slice_path(part))) {
        have = (uint64_t) std::filesystem::file_size(slice_path(part));
        bool same = have >= meta_size && have <= total;
        if (same) {
            FILE * f = slice_fopen(part, "rb");
            std::vector<char> old((size_t) meta_size);
            same = f && fread(old.data(), 1, old.size(), f) == old.size() && old == meta;
            if (f) {
                fclose(f);
            }
        }
        if (!same) {
            std::filesystem::remove(slice_path(part));
            have = 0;
        }
    }
    if (have == 0) {
        FILE * f = slice_fopen(part, "wb");
        if (!f || fwrite(meta.data(), 1, meta.size(), f) != meta.size()) {
            if (f) {
                fclose(f);
            }
            throw std::runtime_error("cannot write " + part);
        }
        fclose(f);
        have = meta_size;
    }

    slice_progress pr{ have, total };
    uint64_t last_report = 0;
    for (int attempt = 0;; attempt++) {
        FILE * f = slice_fopen(part, "ab");
        if (!f) {
            throw std::runtime_error("cannot write " + part);
        }
        uint64_t pos = (uint64_t) std::filesystem::file_size(slice_path(part));
        try {
            // pieces that are next to each other in the same file of the model go in one request:
            // the layers of a range usually are one run of bytes
            size_t i = 0;
            while (i < pieces.size()) {
                if (pieces[i].out_off + pieces[i].size <= pos) {
                    i++;
                    continue;
                }
                size_t j = i + 1;
                const uint64_t max_gap = 1u << 20;
                while (j < pieces.size() && pieces[j].file == pieces[i].file &&
                       pieces[j].src_off >= pieces[j - 1].src_off + pieces[j - 1].size &&
                       pieces[j].src_off - (pieces[j - 1].src_off + pieces[j - 1].size) <= max_gap) {
                    j++;
                }
                // bytes of the run: [from, to) in the file of the model; skip what is already written
                const uint64_t skip = pos > pieces[i].out_off ? pos - pieces[i].out_off : 0;
                const uint64_t from = pieces[i].src_off + skip;
                const uint64_t to   = pieces[j - 1].src_off + pieces[j - 1].size;
                size_t   k   = i;           // the piece the next byte belongs to
                uint64_t src = from;        // offset of the next byte in the file of the model
                std::vector<char> zeros;
                ms.files[pieces[i].file].src->read(from, to - from, [&](const char * d, size_t n) {
                    while (n > 0) {
                        while (k < j && src >= pieces[k].src_off + pieces[k].size) {
                            k++;
                        }
                        if (k == j) {
                            return false;
                        }
                        if (src < pieces[k].src_off) {
                            // a gap between tensors in the model
                            const size_t g = (size_t) std::min<uint64_t>(n, pieces[k].src_off - src);
                            d += g; n -= g; src += g;
                            continue;
                        }
                        // padding of the slice before the tensor
                        const uint64_t dst = pieces[k].out_off + (src - pieces[k].src_off);
                        if (pos < dst) {
                            zeros.assign((size_t) (dst - pos), 0);
                            if (fwrite(zeros.data(), 1, zeros.size(), f) != zeros.size()) {
                                throw std::runtime_error("cannot write " + part + " (is the disk full?)");
                            }
                            pos = dst;
                        }
                        const size_t w = (size_t) std::min<uint64_t>(n, pieces[k].src_off + pieces[k].size - src);
                        if (fwrite(d, 1, w, f) != w) {
                            throw std::runtime_error("cannot write " + part + " (is the disk full?)");
                        }
                        d += w; n -= w; src += w; pos += w;
                        pr.done = pos;
                        if (on_progress && pos - last_report >= (8u << 20)) {
                            last_report = pos;
                            on_progress(pr);
                        }
                    }
                    return true;
                });
                i = j;
            }
            fclose(f);
            f = nullptr;
            if (pos != total) {
                throw std::runtime_error("the slice has " + std::to_string(pos) + " of " + std::to_string(total) + " bytes");
            }
            break;
        } catch (const std::exception & e) {
            if (f) {
                fclose(f);
            }
            if (attempt + 1 >= attempts) {
                throw;
            }
            fprintf(stderr, "agent-node: download: %s, retrying in %d s\n", e.what(), std::min(30, 2 + attempt * 2));
            std::this_thread::sleep_for(std::chrono::seconds(std::min(30, 2 + attempt * 2)));
        }
    }
    pr.done = total;
    if (on_progress) {
        on_progress(pr);
    }
    std::error_code ec;
    std::filesystem::remove(slice_path(path), ec);
    std::filesystem::rename(slice_path(part), slice_path(path));
}

} // namespace agent
