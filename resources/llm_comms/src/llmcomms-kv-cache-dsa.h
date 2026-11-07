#pragma once

#include "llmcomms-kv-cache.h"

#include <vector>

//
// llmcomms_kv_cache_dsa
//

// utilizes two instances of llmcomms_kv_cache:
// - the first instance is for caching key tensors of the model,
// - the second instance is for caching lightning indexer key tensors

class llmcomms_kv_cache_dsa : public llmcomms_memory_i {
public:
    llmcomms_kv_cache_dsa(
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
        const layer_filter_cb & filter_mla,
        const layer_filter_cb & filter_lid,
        const  layer_reuse_cb & reuse);

    ~llmcomms_kv_cache_dsa() = default;

    //
    // llmcomms_memory_i
    //

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

    //
    // llmcomms_kv_cache_dsa specific API
    //

    llmcomms_kv_cache * get_mla() const;
    llmcomms_kv_cache * get_lid() const;

private:
    // we keep indexer KV cache hparams instance here as llmcomms_kv_cache stores only reference to it
    llmcomms_hparams hparams_lid;
    const uint32_t n_stream  = 1;

    std::unique_ptr<llmcomms_kv_cache> kv_mla;
    std::unique_ptr<llmcomms_kv_cache> kv_lid;
};

class llmcomms_kv_cache_dsa_context : public llmcomms_memory_context_i {
public:
    using slot_info_vec_t = llmcomms_kv_cache::slot_info_vec_t;

    // used for errors
    llmcomms_kv_cache_dsa_context(llmcomms_memory_status status);

    // used to create a full-cache context
    llmcomms_kv_cache_dsa_context(
            llmcomms_kv_cache_dsa * kv);

    // used to create an update context
    llmcomms_kv_cache_dsa_context(
            llmcomms_kv_cache_dsa * kv,
            llmcomms_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llmcomms_kv_cache_dsa_context(
            llmcomms_kv_cache_dsa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_ik,
            std::vector<llmcomms_ubatch> ubatches);

    virtual ~llmcomms_kv_cache_dsa_context();

    //
    // llmcomms_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llmcomms_memory_status  get_status() const override;
    const llmcomms_ubatch & get_ubatch() const override;

    //
    // llmcomms_kv_cache_dsa_context specific API
    //

    const llmcomms_kv_cache_context * get_mla() const;
    const llmcomms_kv_cache_context * get_lid()  const;

private:
    //llmcomms_kv_cache_dsa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llmcomms_ubatch> ubatches;

    const llmcomms_memory_context_ptr ctx_mla;
    const llmcomms_memory_context_ptr ctx_lid;

    const llmcomms_memory_status status;
};
