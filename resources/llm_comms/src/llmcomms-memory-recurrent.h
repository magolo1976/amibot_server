#pragma once

#include "llmcomms-batch.h"
#include "llmcomms-graph.h"
#include "llmcomms-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llmcomms_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llmcomms_memory_recurrent_context_i
//       see the implementation of llmcomms_kv_cache_context_i for an example how to do it
class llmcomms_memory_recurrent : public llmcomms_memory_i {
public:
    llmcomms_memory_recurrent(
            const llmcomms_model & model,
                         bool   no_alloc,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llmcomms_memory_recurrent() = default;

    //
    // llmcomms_memory_i
    //

    llmcomms_memory_context_ptr init_batch(
            llmcomms_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llmcomms_memory_context_ptr init_full() override;

    llmcomms_memory_context_ptr init_update(llmcomms_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_cp  (llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_keep(llmcomms_seq_id seq_id)                                                          override;
    void seq_add (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) override;
    void seq_div (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, int d) override;

    llmcomms_pos seq_pos_min(llmcomms_seq_id seq_id) const override;
    llmcomms_pos seq_pos_max(llmcomms_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llmcomms_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llmcomms_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) const override;
    void state_read (llmcomms_io_read_i  & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llmcomms_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llmcomms_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        // largest rollback whose snapshot the last ubatch guarantees: a ubatch of n <= n_rs_seq + 1 tokens
        // writes snapshots 0..n-1, a longer one only guarantees the final state (ggml_gated_delta_net)
        uint32_t rs_max = 0;

        std::set<llmcomms_seq_id> seq_id;

        bool has_seq_id(const llmcomms_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llmcomms_model & model;
    const llmcomms_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the RS cache along with the allocated backend buffers:
    llmcomms_memory_bufs ctxs_bufs;

    // true if no layers - can happen if the layer filter removes all layers
    bool is_empty() const;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llmcomms_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llmcomms_seq_id seq_id = -1) const;
    void state_write_data(llmcomms_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const;

    bool state_read_meta(llmcomms_io_read_i & io, uint32_t cell_count, llmcomms_seq_id dest_seq_id = -1);
    bool state_read_data(llmcomms_io_read_i & io, uint32_t cell_count);

    void state_clear(llmcomms_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llmcomms_memory_recurrent_context : public llmcomms_memory_context_i {
public:
    // used for errors
    llmcomms_memory_recurrent_context(llmcomms_memory_status status);

    // used to create a full-cache or update context
    llmcomms_memory_recurrent_context(
            llmcomms_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llmcomms_memory_recurrent_context(
            llmcomms_memory_recurrent * mem,
            std::vector<llmcomms_ubatch> ubatches);

    virtual ~llmcomms_memory_recurrent_context();

    //
    // llmcomms_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llmcomms_memory_status  get_status() const override;
    const llmcomms_ubatch & get_ubatch() const override;

    //
    // llmcomms_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    int32_t s_copy(int i) const;

private:
    const llmcomms_memory_status status;

    llmcomms_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llmcomms_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
