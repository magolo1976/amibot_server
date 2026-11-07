#pragma once

#include "llmcomms-batch.h"
#include "llmcomms-graph.h"
#include "llmcomms-kv-cache.h"
#include "llmcomms-memory.h"
#include "llmcomms-memory-recurrent.h"

#include <memory>
#include <vector>

//
// llmcomms_memory_hybrid
//

// utilizes instances of llmcomms_memory_recurrent and llmcomms_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class llmcomms_memory_hybrid : public llmcomms_memory_i {
public:
    llmcomms_memory_hybrid(
        const llmcomms_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llmcomms_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    ~llmcomms_memory_hybrid() = default;

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
    void state_read (llmcomms_io_read_i  & io, llmcomms_seq_id seq_id = -1, llmcomms_state_seq_flags flags = 0)       override;

    //
    // llmcomms_memory_hybrid specific API
    //

    llmcomms_kv_cache * get_mem_attn() const;
    llmcomms_memory_recurrent * get_mem_recr() const;

private:
    const llmcomms_hparams & hparams;

    const std::unique_ptr<llmcomms_kv_cache> mem_attn;
    const std::unique_ptr<llmcomms_memory_recurrent> mem_recr;
};

class llmcomms_memory_hybrid_context : public llmcomms_memory_context_i {
public:
    using slot_info_vec_t = llmcomms_kv_cache::slot_info_vec_t;

    // init failure
    explicit llmcomms_memory_hybrid_context(llmcomms_memory_status status);

    // init full
    explicit llmcomms_memory_hybrid_context(llmcomms_memory_hybrid * mem);

    // init update
    explicit llmcomms_memory_hybrid_context(
        llmcomms_memory_hybrid * mem,
              llmcomms_context * lctx,
                       bool   optimize);

    // init success
    llmcomms_memory_hybrid_context(
              llmcomms_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llmcomms_ubatch>   ubatches);

    ~llmcomms_memory_hybrid_context() = default;

    bool next()  override;
    bool apply() override;

    llmcomms_memory_status  get_status() const override;
    const llmcomms_ubatch & get_ubatch() const override;

    //
    // llmcomms_memory_hybrid_context
    //

    const llmcomms_kv_cache_context * get_attn() const;
    const llmcomms_memory_recurrent_context * get_recr() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llmcomms_ubatch> ubatches;

    const llmcomms_memory_context_ptr ctx_attn;
    const llmcomms_memory_context_ptr ctx_recr;

    const llmcomms_memory_status status;
};
