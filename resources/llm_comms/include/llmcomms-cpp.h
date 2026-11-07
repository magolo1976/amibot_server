#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "llmcomms-core.h"

struct llmcomms_model_deleter {
    void operator()(llmcomms_model * model) { llmcomms_model_free(model); }
};

struct llmcomms_context_deleter {
    void operator()(llmcomms_context * context) { llmcomms_free(context); }
};

struct llmcomms_sampler_deleter {
    void operator()(llmcomms_sampler * sampler) { llmcomms_sampler_free(sampler); }
};

struct llmcomms_adapter_lora_deleter {
    void operator()(llmcomms_adapter_lora * adapter) { llmcomms_adapter_lora_free(adapter); }
};

struct llmcomms_batch_ext_deleter {
    void operator()(llmcomms_batch_ext * batch) { llmcomms_batch_ext_free(batch); }
};

typedef std::unique_ptr<llmcomms_model, llmcomms_model_deleter> llmcomms_model_ptr;
typedef std::unique_ptr<llmcomms_context, llmcomms_context_deleter> llmcomms_context_ptr;
typedef std::unique_ptr<llmcomms_sampler, llmcomms_sampler_deleter> llmcomms_sampler_ptr;
typedef std::unique_ptr<llmcomms_adapter_lora, llmcomms_adapter_lora_deleter> llmcomms_adapter_lora_ptr;
typedef std::unique_ptr<llmcomms_batch_ext, llmcomms_batch_ext_deleter> llmcomms_batch_ext_ptr;
