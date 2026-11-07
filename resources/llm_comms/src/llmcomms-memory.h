#pragma once

#include "llmcomms-core.h"
#include "llmcomms-graph.h"

#include <map>
#include <memory>
#include <functional>
#include <vector>

struct llmcomms_ubatch;

class llmcomms_batch_allocr;

class llmcomms_io_write_i;
class llmcomms_io_read_i;

struct llmcomms_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llmcomms_context_type ctx_type;

    llmcomms_memory_t mem_other;

    // only simulate the memory allocations
    bool no_alloc;
};

// the tensor contexts and backend buffers of a memory instance
// with no_alloc, dummy buffers are attached instead of allocating the tensor data, so the memory use can still be measured
struct llmcomms_memory_bufs {
    explicit llmcomms_memory_bufs(bool no_alloc) : no_alloc(no_alloc) {}

    // allocate the tensors of ctx on buft, or attach a dummy buffer if no_alloc is set
    // returns the buffer, for logging
    ggml_backend_buffer_t emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name);

    // the memory used by the tensors: the buffer sizes, or the simulated sizes if no_alloc is set
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // allocate a single ctx buffer - for the memory instances that keep their own buffer layout
    static ggml_backend_buffer_ptr alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name);

    bool empty() const { return bufs.empty(); }

    auto begin()       { return bufs.begin(); }
    auto end()         { return bufs.end();   }
    auto begin() const { return bufs.begin(); }
    auto end()   const { return bufs.end();   }

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> bufs;

    const bool no_alloc;
};

enum llmcomms_memory_status {
    LLMCOMMS_MEMORY_STATUS_SUCCESS = 0,
    LLMCOMMS_MEMORY_STATUS_NO_UPDATE,
    LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE,
    LLMCOMMS_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llmcomms_memory_status llmcomms_memory_status_combine(llmcomms_memory_status s0, llmcomms_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llmcomms_memory_status_is_fail(llmcomms_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llmcomms_kv_cache_context
//   - llmcomms_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llmcomms_memory_i::apply()
struct llmcomms_memory_context_i {
    virtual ~llmcomms_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llmcomms_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llmcomms_memory_status get_status() const = 0;
};

using llmcomms_memory_context_ptr = std::unique_ptr<llmcomms_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llmcomms_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llmcomms_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llmcomms_memory_context_i::get_status() for the result
    // embd_all: keep all the tokens of a sequence in one ubatch (all tokens are output, or the attention is non-causal)
    virtual llmcomms_memory_context_ptr init_batch(
            llmcomms_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llmcomms_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLMCOMMS_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llmcomms_memory_context_ptr init_update(llmcomms_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1) = 0;
    virtual void seq_cp  (llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) = 0;
    virtual void seq_keep(llmcomms_seq_id seq_id) = 0;
    virtual void seq_add (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) = 0;
    virtual void seq_div (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, int d) = 0;

    virtual llmcomms_pos seq_pos_min(llmcomms_seq_id seq_id) const = 0;
    virtual llmcomms_pos seq_pos_max(llmcomms_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llmcomms_io_read_i  & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) = 0;
};

using llmcomms_memory_ptr = std::unique_ptr<llmcomms_memory_i>;
