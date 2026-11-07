#include "llmcomms-memory.h"

#include "llmcomms-ext.h"
#include "llmcomms-io.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

llmcomms_memory_status llmcomms_memory_status_combine(llmcomms_memory_status s0, llmcomms_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLMCOMMS_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLMCOMMS_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE:
        case LLMCOMMS_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLMCOMMS_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLMCOMMS_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE:
        case LLMCOMMS_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLMCOMMS_MEMORY_STATUS_SUCCESS : LLMCOMMS_MEMORY_STATUS_NO_UPDATE;
}

bool llmcomms_memory_status_is_fail(llmcomms_memory_status status) {
    switch (status) {
        case LLMCOMMS_MEMORY_STATUS_SUCCESS:
        case LLMCOMMS_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLMCOMMS_MEMORY_STATUS_FAILED_PREPARE:
        case LLMCOMMS_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

//
// llmcomms_memory_bufs
//

ggml_backend_buffer_ptr llmcomms_memory_bufs::alloc_buf(ggml_context * ctx, ggml_backend_buffer_type_t buft, bool no_alloc, const char * name) {
    ggml_backend_buffer_ptr buf;
    if (no_alloc) {
        buf.reset(ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0)); // dummy buffer
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            t->buffer = buf.get(); // set dummy buffer so that the backend scheduler won't try to allocate the tensors
        }
    } else {
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)); // real buffer
    }
    if (!buf) {
        throw std::runtime_error(std::string("failed to allocate buffer for ") + name);
    }
    return buf;
}

ggml_backend_buffer_t llmcomms_memory_bufs::emplace(ggml_context_ptr ctx, ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_buffer_ptr buf = alloc_buf(ctx.get(), buft, no_alloc, name);

    // initialize the buffer to avoid NaNs in the padding
    ggml_backend_buffer_clear(buf.get(), 0);

    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

std::map<ggml_backend_buffer_type_t, size_t> llmcomms_memory_bufs::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }
    return ret;
}

// collects the buffers of the tensors that a state write visits, without keeping the data
struct llmcomms_memory_buffers_writer : public llmcomms_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<ggml_backend_buffer_t> llmcomms_memory_get_buffers(const llmcomms_memory_t mem) {
    if (mem == nullptr) {
        return {};
    }

    llmcomms_memory_buffers_writer writer;

    mem->state_write(writer);

    return std::vector<ggml_backend_buffer_t>(writer.buffers.begin(), writer.buffers.end());
}
