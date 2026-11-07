#include "llmcomms-memory-hybrid-iswa.h"

#include "llmcomms-impl.h"
#include "llmcomms-model.h"
#include "llmcomms-context.h"

//
// llmcomms_memory_hybrid_iswa
//

llmcomms_memory_hybrid_iswa::llmcomms_memory_hybrid_iswa(
        const llmcomms_model & model,
                      bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
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
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llmcomms_kv_cache_iswa(
        model,
        no_alloc,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llmcomms_memory_recurrent(
        model,
        no_alloc,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

llmcomms_memory_context_ptr llmcomms_memory_hybrid_iswa::init_batch(llmcomms_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llmcomms_ubatch> ubatches;

        while (true) {
            llmcomms_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_base()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLMCOMMS_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llmcomms_memory_hybrid_iswa_context>(LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache (iswa version returns both base and swa slot infos)
        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            LLMCOMMS_LOG_ERROR("%s: failed to prepare attention base ubatches\n", __func__);
            return std::make_unique<llmcomms_memory_hybrid_iswa_context>(LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_swa = mem_attn->get_swa()->prepare(ubatches);
        if (sinfos_swa.empty()) {
            LLMCOMMS_LOG_ERROR("%s: failed to prepare attention swa ubatches\n", __func__);
            return std::make_unique<llmcomms_memory_hybrid_iswa_context>(LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llmcomms_memory_hybrid_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while(false);

    return std::make_unique<llmcomms_memory_hybrid_iswa_context>(LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE);
}

llmcomms_memory_context_ptr llmcomms_memory_hybrid_iswa::init_full() {
    return std::make_unique<llmcomms_memory_hybrid_iswa_context>(this);
}

llmcomms_memory_context_ptr llmcomms_memory_hybrid_iswa::init_update(llmcomms_context * lctx, bool optimize) {
    return std::make_unique<llmcomms_memory_hybrid_iswa_context>(this, lctx, optimize);
}

bool llmcomms_memory_hybrid_iswa::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llmcomms_memory_hybrid_iswa::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llmcomms_memory_hybrid_iswa::seq_rm(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llmcomms_memory_hybrid_iswa::seq_cp(llmcomms_seq_id seq_id_src, llmcomms_seq_id seq_id_dst, llmcomms_pos p0, llmcomms_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llmcomms_memory_hybrid_iswa::seq_keep(llmcomms_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llmcomms_memory_hybrid_iswa::seq_add(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1, llmcomms_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llmcomms_memory_hybrid_iswa::seq_div(llmcomms_seq_id seq_id, llmcomms_pos p0, llmcomms_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llmcomms_pos llmcomms_memory_hybrid_iswa::seq_pos_min(llmcomms_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llmcomms_pos llmcomms_memory_hybrid_iswa::seq_pos_max(llmcomms_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llmcomms_memory_hybrid_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llmcomms_memory_hybrid_iswa::state_write(llmcomms_io_write_i & io, llmcomms_seq_id seq_id, llmcomms_state_seq_flags flags) const {
    mem_attn->state_write(io, seq_id, flags);
    mem_recr->state_write(io, seq_id, flags);
}

void llmcomms_memory_hybrid_iswa::state_read(llmcomms_io_read_i & io, llmcomms_seq_id seq_id, llmcomms_state_seq_flags flags) {
    mem_attn->state_read(io, seq_id, flags);
    mem_recr->state_read(io, seq_id, flags);
}

llmcomms_kv_cache_iswa * llmcomms_memory_hybrid_iswa::get_mem_attn() const {
    return mem_attn.get();
}

llmcomms_memory_recurrent * llmcomms_memory_hybrid_iswa::get_mem_recr() const {
    return mem_recr.get();
}

//
// llmcomms_memory_hybrid_iswa_context
//

llmcomms_memory_hybrid_iswa_context::llmcomms_memory_hybrid_iswa_context(llmcomms_memory_status status) : status(status) {}

llmcomms_memory_hybrid_iswa_context::llmcomms_memory_hybrid_iswa_context(llmcomms_memory_hybrid_iswa * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llmcomms_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llmcomms_memory_hybrid_iswa_context::llmcomms_memory_hybrid_iswa_context(
        llmcomms_memory_hybrid_iswa * mem,
                   llmcomms_context * lctx,
                            bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llmcomms_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llmcomms_memory_hybrid_iswa_context::llmcomms_memory_hybrid_iswa_context(
           llmcomms_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<llmcomms_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llmcomms_kv_cache_iswa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_swa), this->ubatches)),
    ctx_recr(new llmcomms_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llmcomms_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llmcomms_memory_hybrid_iswa_context::next() {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llmcomms_memory_hybrid_iswa_context::apply() {
    assert(!llmcomms_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llmcomms_memory_status llmcomms_memory_hybrid_iswa_context::get_status() const {
    return status;
}

const llmcomms_ubatch & llmcomms_memory_hybrid_iswa_context::get_ubatch() const {
    assert(status == LLMCOMMS_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llmcomms_kv_cache_iswa_context * llmcomms_memory_hybrid_iswa_context::get_attn() const {
    return static_cast<const llmcomms_kv_cache_iswa_context *>(ctx_attn.get());
}

const llmcomms_memory_recurrent_context * llmcomms_memory_hybrid_iswa_context::get_recr() const {
    return static_cast<const llmcomms_memory_recurrent_context *>(ctx_recr.get());
}
