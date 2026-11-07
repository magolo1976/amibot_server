#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <cstdio>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define LLMCOMMS_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

struct llmcomms_file;
struct llmcomms_mmap;
struct llmcomms_mlock;

using llmcomms_files  = std::vector<std::unique_ptr<llmcomms_file>>;
using llmcomms_mmaps  = std::vector<std::unique_ptr<llmcomms_mmap>>;
using llmcomms_mlocks = std::vector<std::unique_ptr<llmcomms_mlock>>;

struct llmcomms_file {
    llmcomms_file(const char * fname, const char * mode, bool use_direct_io = false);
    llmcomms_file(FILE * file);
    ~llmcomms_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llmcomms_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llmcomms_mmap(const llmcomms_mmap &) = delete;
    llmcomms_mmap(struct llmcomms_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llmcomms_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llmcomms_mlock {
    llmcomms_mlock();
    ~llmcomms_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llmcomms_memory_range {
    const void * addr;
    size_t size;
};

using llmcomms_memory_ranges = std::vector<llmcomms_memory_range>;

// Prefetch the host pages covering these memory ranges.
void llmcomms_prefetch(llmcomms_memory_ranges mr);

size_t llmcomms_path_max();
