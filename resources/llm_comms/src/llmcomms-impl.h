#pragma once

#include "ggml.h" // for ggml_log_level

#include <string>
#include <type_traits>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLMCOMMS_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLMCOMMS_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLMCOMMS_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

LLMCOMMS_ATTRIBUTE_FORMAT(2, 3)
void llmcomms_log_internal        (ggml_log_level level, const char * format, ...);
void llmcomms_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define LLMCOMMS_LOG(...)       llmcomms_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define LLMCOMMS_LOG_INFO(...)  llmcomms_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define LLMCOMMS_LOG_WARN(...)  llmcomms_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define LLMCOMMS_LOG_ERROR(...) llmcomms_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LLMCOMMS_LOG_DEBUG(...) llmcomms_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LLMCOMMS_LOG_CONT(...)  llmcomms_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

template <typename dst_t, typename src_t>
static inline dst_t llmcomms_cast(src_t v) {
    if constexpr (std::is_same_v<src_t, dst_t>) {
        return v;
    } else if constexpr (std::is_same_v<src_t, ggml_fp16_t> && std::is_same_v<dst_t, float>) {
        return ggml_fp16_to_fp32(v);
    } else if constexpr (std::is_same_v<src_t, float> && std::is_same_v<dst_t, ggml_fp16_t>) {
        return ggml_fp32_to_fp16(v);
    } else {
        static_assert(std::is_same_v<dst_t, void>, "unsupported type combination");
    }
}

static inline ggml_tensor * llmcomms_mul_mat_hadamard(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat(ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

// Prefetch the host pages needed to gather these rows.
void llmcomms_prefetch_rows(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows);

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};

void llmcomms_clear_tensor_data(ggml_tensor * t, size_t offset, size_t size);

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to llmcomms_format ?
LLMCOMMS_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string llmcomms_format_tensor_shape(const std::vector<int64_t> & ne);
std::string llmcomms_format_tensor_shape(const struct ggml_tensor * t);

// arrays are rendered only until the result is longer than max_len
std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i, size_t max_len = SIZE_MAX);
