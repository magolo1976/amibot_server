#pragma once

#include "llmcomms-kv-cache.h"

#include <vector>

//
// llmcomms_kv_cache_iswa
//

// utilizes two instances of llmcomms_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class llmcomms_kv_cache_iswa : public llmcomms_memory_i {
public:
    llmcomms_kv_cache_iswa(
            const llmcomms_model & model,
                         bool   no_alloc,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
               llmcomms_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share);

    llmcomms_kv_cache_iswa(
            const llmcomms_model & model,
                         bool   no_alloc,
          const llmcomms_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
               llmcomms_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share);

    ~llmcomms_kv_cache_iswa() = default;

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
    // llmcomms_kv_cache_iswa specific API
    //

    llmcomms_kv_cache * get_base() const;
    llmcomms_kv_cache * get_swa () const;

private:
    const bool unified;

    std::unique_ptr<llmcomms_kv_cache> kv_base;
    std::unique_ptr<llmcomms_kv_cache> kv_swa;
};

class llmcomms_kv_cache_iswa_context : public llmcomms_memory_context_i {
public:
    using slot_info_vec_t = llmcomms_kv_cache::slot_info_vec_t;

    // used for errors
    llmcomms_kv_cache_iswa_context(llmcomms_memory_status status);

    // used to create a full-cache context
    llmcomms_kv_cache_iswa_context(
            llmcomms_kv_cache_iswa * kv);

    // used to create an update context
    llmcomms_kv_cache_iswa_context(
            llmcomms_kv_cache_iswa * kv,
            llmcomms_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llmcomms_kv_cache_iswa_context(
            llmcomms_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<llmcomms_ubatch> ubatches);

    virtual ~llmcomms_kv_cache_iswa_context();

    //
    // llmcomms_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llmcomms_memory_status  get_status() const override;
    const llmcomms_ubatch & get_ubatch() const override;

    //
    // llmcomms_kv_cache_iswa_context specific API
    //

    const llmcomms_kv_cache_context * get_base() const;
    const llmcomms_kv_cache_context * get_swa()  const;

private:
    //llmcomms_kv_cache_iswa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llmcomms_ubatch> ubatches;

    const llmcomms_memory_context_ptr ctx_base;
    const llmcomms_memory_context_ptr ctx_swa;

    const llmcomms_memory_status status;
};
