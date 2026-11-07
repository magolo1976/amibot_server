#pragma once

#include "llmcomms-kv-cache.h"

#include <vector>

// llmcomms_kv_cache_msa

// uses two instances of llmcomms_kv_cache, one for K/V tensors, and one for the MSA indexer tensors
// both receive identical sequence operations and identical ubatches, so their cell layouts stay in synced.
// the context also exposes per-ubatch pos - cell translation maps populated from llmcomms_kv_cells via
// llmcomms_kv_cache::get_cells(), which the model graph uses to run MSA block selection in position space

class llmcomms_kv_cache_msa : public llmcomms_memory_i {
public:
    llmcomms_kv_cache_msa(
            const llmcomms_model & model,
                         bool   no_alloc,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_swa,
               llmcomms_swa_type   swa_type,
        const layer_filter_cb & filter,
        const layer_filter_cb & filter_idx,
        const  layer_reuse_cb & reuse);

    ~llmcomms_kv_cache_msa() = default;

    // llmcomms_memory_i

    llmcomms_memory_context_ptr init_batch(
            llmcomms_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llmcomms_memory_context_ptr init_full() override;

    llmcomms_memory_context_ptr init_update(llmcomms_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_cp  (llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) override;
    void seq_keep(llmcomms_seq_id seq_id)                                                          override;
    void seq_add (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) override;
    void seq_div (llmcomms_seq_id seq_id,                              llmcomms_pos p0, llmcomms_pos p1, int d) override;

    llmcomms_pos seq_pos_min(llmcomms_seq_id seq_id) const override;
    llmcomms_pos seq_pos_max(llmcomms_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) const override;
    void state_read (llmcomms_io_read_i  & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0) override;

    // llmcomms_kv_cache_msa specific API

    llmcomms_kv_cache * get_base() const;
    llmcomms_kv_cache * get_idx () const;

    uint32_t       get_n_seq_max() const { return n_seq_max; }
    uint32_t       get_n_swa()    const { return n_swa; }
    llmcomms_swa_type get_swa_type() const { return swa_type; }

private:
    // keep the indexer KV cache hparams instance here as llmcomms_kv_cache stores only a reference
    llmcomms_hparams hparams_idx;

    const uint32_t n_stream  = 1;
    const uint32_t n_seq_max = 1;

    const uint32_t       n_swa    = 0;
    const llmcomms_swa_type swa_type = LLMCOMMS_SWA_TYPE_NONE;

    std::unique_ptr<llmcomms_kv_cache> kv_base;
    std::unique_ptr<llmcomms_kv_cache> kv_idx;
};

class llmcomms_kv_cache_msa_context : public llmcomms_memory_context_i {
public:
    using slot_info_vec_t = llmcomms_kv_cache::slot_info_vec_t;

    // used for errors
    llmcomms_kv_cache_msa_context(llmcomms_memory_status status);

    // used to create a full-cache context
    llmcomms_kv_cache_msa_context(
            llmcomms_kv_cache_msa * kv);

    // used to create an update context
    llmcomms_kv_cache_msa_context(
            llmcomms_kv_cache_msa * kv,
            llmcomms_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llmcomms_kv_cache_msa_context(
            llmcomms_kv_cache_msa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_idx,
            std::vector<llmcomms_ubatch> ubatches);

    virtual ~llmcomms_kv_cache_msa_context();

    // llmcomms_memory_context_i

    bool next()  override;
    bool apply() override;

    llmcomms_memory_status  get_status() const override;
    const llmcomms_ubatch & get_ubatch() const override;

    // llmcomms_kv_cache_msa_context specific API

    const llmcomms_kv_cache_context * get_base() const;
    const llmcomms_kv_cache_context * get_idx () const;

    // max position currently present in the cache plus one, padded MSA blocks are defined over token positions
    // so the block-selection tensors are sized by this value rather than by the number of cells
    uint32_t get_n_pos() const;

    // position <-> cell translation maps, populated from the base cache cells
    // the model graph relates cache contents to token positions only through these per ubatch inputs
    // value for empty or other-sequence cells is 0 so consumers must mask them
    void set_input_cell_pos(ggml_tensor * dst, const llmcomms_ubatch * ubatch, int32_t div) const;
    // positions without a cell map to cell 0, consumers must mask them assumes one sequence per stream
    void set_input_pos_slot(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const;
    void set_input_pos_mask(ggml_tensor * dst, const llmcomms_ubatch * ubatch) const;

private:
    llmcomms_kv_cache_msa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llmcomms_ubatch> ubatches;

    const llmcomms_memory_context_ptr ctx_base;
    const llmcomms_memory_context_ptr ctx_idx;

    const llmcomms_memory_status status;
};
