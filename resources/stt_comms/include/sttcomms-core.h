#ifndef STTCOMMS_H
#define STTCOMMS_H

#include "ggml.h"
#include "ggml-cpu.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __GNUC__
#    define STTCOMMS_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define STTCOMMS_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define STTCOMMS_DEPRECATED(func, hint) func
#endif

#ifdef STTCOMMS_SHARED
#    ifdef _WIN32
#        ifdef STTCOMMS_BUILD
#            define STTCOMMS_API __declspec(dllexport)
#        else
#            define STTCOMMS_API __declspec(dllimport)
#        endif
#    else
#        define STTCOMMS_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define STTCOMMS_API
#endif

#define STTCOMMS_SAMPLE_RATE 16000
#define STTCOMMS_N_FFT       400
#define STTCOMMS_HOP_LENGTH  160
#define STTCOMMS_CHUNK_SIZE  30

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // The following interface is thread-safe as long as the sample sttcomms_context is not used by multiple threads
    // concurrently.
    //
    // Basic usage:
    //
    //     #include "sttcomms.h"
    //
    //     ...
    //
    //     sttcomms_context_params cparams = sttcomms_context_default_params();
    //
    //     struct sttcomms_context * ctx = sttcomms_init_from_file_with_params("/path/to/ggml-base.en.bin", cparams);
    //
    //     if (sttcomms_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
    //         fprintf(stderr, "failed to process audio\n");
    //         return 7;
    //     }
    //
    //     const int n_segments = sttcomms_full_n_segments(ctx);
    //     for (int i = 0; i < n_segments; ++i) {
    //         const char * text = sttcomms_full_get_segment_text(ctx, i);
    //         printf("%s", text);
    //     }
    //
    //     sttcomms_free(ctx);
    //
    //     ...
    //
    // This is a demonstration of the most straightforward usage of the library.
    // "pcmf32" contains the RAW audio data in 32-bit floating point format.
    //
    // The interface also allows for more fine-grained control over the computation, but it requires a deeper
    // understanding of how the model works.
    //

    struct sttcomms_context;
    struct sttcomms_state;
    struct sttcomms_full_params;

    typedef int32_t sttcomms_pos;
    typedef int32_t sttcomms_token;
    typedef int32_t sttcomms_seq_id;

    enum sttcomms_alignment_heads_preset {
        STTCOMMS_AHEADS_NONE,
        STTCOMMS_AHEADS_N_TOP_MOST,  // All heads from the N-top-most text-layers
        STTCOMMS_AHEADS_CUSTOM,
        STTCOMMS_AHEADS_TINY_EN,
        STTCOMMS_AHEADS_TINY,
        STTCOMMS_AHEADS_BASE_EN,
        STTCOMMS_AHEADS_BASE,
        STTCOMMS_AHEADS_SMALL_EN,
        STTCOMMS_AHEADS_SMALL,
        STTCOMMS_AHEADS_MEDIUM_EN,
        STTCOMMS_AHEADS_MEDIUM,
        STTCOMMS_AHEADS_LARGE_V1,
        STTCOMMS_AHEADS_LARGE_V2,
        STTCOMMS_AHEADS_LARGE_V3,
        STTCOMMS_AHEADS_LARGE_V3_TURBO,
    };

    typedef struct sttcomms_ahead {
        int n_text_layer;
        int n_head;
    } sttcomms_ahead;

    typedef struct sttcomms_aheads {
        size_t n_heads;
        const sttcomms_ahead * heads;
    } sttcomms_aheads;

    struct sttcomms_context_params {
        bool  use_gpu;
        bool  flash_attn;
        int   gpu_device;  // CUDA device

        // [EXPERIMENTAL] Token-level timestamps with DTW
        bool dtw_token_timestamps;
        enum sttcomms_alignment_heads_preset dtw_aheads_preset;

        int dtw_n_top;
        struct sttcomms_aheads dtw_aheads;

        size_t dtw_mem_size; // TODO: remove
    };

    typedef struct sttcomms_token_data {
        sttcomms_token id;  // token id
        sttcomms_token tid; // forced timestamp token id

        float p;           // probability of the token
        float plog;        // log probability of the token
        float pt;          // probability of the timestamp token
        float ptsum;       // sum of probabilities of all timestamp tokens

        // token-level timestamp data
        // do not use if you haven't computed token-level timestamps
        int64_t t0;        // start time of the token
        int64_t t1;        //   end time of the token

        // [EXPERIMENTAL] Token-level timestamps with DTW
        // do not use if you haven't computed token-level timestamps with dtw
        // Roughly corresponds to the moment in audio in which the token was output
        int64_t t_dtw;

        float vlen;        // voice length of the token
    } sttcomms_token_data;

    typedef struct sttcomms_model_loader {
        void * context;

        size_t (*read)(void * ctx, void * output, size_t read_size);
        bool    (*eof)(void * ctx);
        void  (*close)(void * ctx);
    } sttcomms_model_loader;

    // grammar element type
    enum sttcomms_gretype {
        // end of rule definition
        STTCOMMS_GRETYPE_END            = 0,

        // start of alternate definition for rule
        STTCOMMS_GRETYPE_ALT            = 1,

        // non-terminal element: reference to rule
        STTCOMMS_GRETYPE_RULE_REF       = 2,

        // terminal element: character (code point)
        STTCOMMS_GRETYPE_CHAR           = 3,

        // inverse char(s) ([^a], [^a-b] [^abc])
        STTCOMMS_GRETYPE_CHAR_NOT       = 4,

        // modifies a preceding STTCOMMS_GRETYPE_CHAR or STTCOMMS_GRETYPE_CHAR_ALT to
        // be an inclusive range ([a-z])
        STTCOMMS_GRETYPE_CHAR_RNG_UPPER = 5,

        // modifies a preceding STTCOMMS_GRETYPE_CHAR or
        // STTCOMMS_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
        STTCOMMS_GRETYPE_CHAR_ALT       = 6,
    };

    typedef struct sttcomms_grammar_element {
        enum sttcomms_gretype type;
        uint32_t             value; // Unicode code point or rule ID
    } sttcomms_grammar_element;

    typedef struct sttcomms_vad_params {
        float threshold;               // Probability threshold to consider as speech.
        int   min_speech_duration_ms;  // Min duration for a valid speech segment.
        int   min_silence_duration_ms; // Min silence duration to consider speech as ended.
        float max_speech_duration_s;   // Max duration of a speech segment before forcing a new segment.
        int   speech_pad_ms;           // Padding added before and after speech segments.
        float samples_overlap;         // Overlap in seconds when copying audio samples from speech segment.
    } sttcomms_vad_params;

    STTCOMMS_API const char * sttcomms_version(void);

    // Various functions for loading a ggml sttcomms model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    STTCOMMS_API struct sttcomms_context * sttcomms_init_from_file_with_params  (const char * path_model,              struct sttcomms_context_params params);
    STTCOMMS_API struct sttcomms_context * sttcomms_init_from_buffer_with_params(void * buffer, size_t buffer_size,    struct sttcomms_context_params params);
    STTCOMMS_API struct sttcomms_context * sttcomms_init_with_params            (struct sttcomms_model_loader * loader, struct sttcomms_context_params params);

    // These are the same as the above, but the internal state of the context is not allocated automatically
    // It is the responsibility of the caller to allocate the state using sttcomms_init_state() (#523)
    STTCOMMS_API struct sttcomms_context * sttcomms_init_from_file_with_params_no_state  (const char * path_model,              struct sttcomms_context_params params);
    STTCOMMS_API struct sttcomms_context * sttcomms_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size,    struct sttcomms_context_params params);
    STTCOMMS_API struct sttcomms_context * sttcomms_init_with_params_no_state            (struct sttcomms_model_loader * loader, struct sttcomms_context_params params);

    STTCOMMS_DEPRECATED(
        STTCOMMS_API struct sttcomms_context * sttcomms_init_from_file(const char * path_model),
        "use sttcomms_init_from_file_with_params instead"
    );
    STTCOMMS_DEPRECATED(
        STTCOMMS_API struct sttcomms_context * sttcomms_init_from_buffer(void * buffer, size_t buffer_size),
        "use sttcomms_init_from_buffer_with_params instead"
    );
    STTCOMMS_DEPRECATED(
        STTCOMMS_API struct sttcomms_context * sttcomms_init(struct sttcomms_model_loader * loader),
        "use sttcomms_init_with_params instead"
    );
    STTCOMMS_DEPRECATED(
        STTCOMMS_API struct sttcomms_context * sttcomms_init_from_file_no_state(const char * path_model),
        "use sttcomms_init_from_file_with_params_no_state instead"
    );
    STTCOMMS_DEPRECATED(
        STTCOMMS_API struct sttcomms_context * sttcomms_init_from_buffer_no_state(void * buffer, size_t buffer_size),
        "use sttcomms_init_from_buffer_with_params_no_state instead"
    );
    STTCOMMS_DEPRECATED(
        STTCOMMS_API struct sttcomms_context * sttcomms_init_no_state(struct sttcomms_model_loader * loader),
        "use sttcomms_init_with_params_no_state instead"
    );

    STTCOMMS_API struct sttcomms_state * sttcomms_init_state(struct sttcomms_context * ctx);

    // Given a context, enable use of OpenVINO for encode inference.
    // model_path: Optional path to OpenVINO encoder IR model. If set to nullptr,
    //                      the path will be generated from the ggml model path that was passed
    //                      in to sttcomms_init_from_file. For example, if 'path_model' was
    //                      "/path/to/ggml-base.en.bin", then OpenVINO IR model path will be
    //                      assumed to be "/path/to/ggml-base.en-encoder-openvino.xml".
    // device: OpenVINO device to run inference on ("CPU", "GPU", etc.)
    // cache_dir: Optional cache directory that can speed up init time, especially for
    //                     GPU, by caching compiled 'blobs' there.
    //                     Set to nullptr if not used.
    // Returns 0 on success. If OpenVINO is not enabled in build, this simply returns 1.
    STTCOMMS_API int sttcomms_ctx_init_openvino_encoder_with_state(
        struct sttcomms_context * ctx,
          struct sttcomms_state * state,
                    const char * model_path,
                    const char * device,
                    const char * cache_dir);

    STTCOMMS_API int sttcomms_ctx_init_openvino_encoder(
        struct sttcomms_context * ctx,
                    const char * model_path,
                    const char * device,
                    const char * cache_dir);

    // Frees all allocated memory
    STTCOMMS_API void sttcomms_free      (struct sttcomms_context * ctx);
    STTCOMMS_API void sttcomms_free_state(struct sttcomms_state * state);
    STTCOMMS_API void sttcomms_free_params(struct sttcomms_full_params * params);
    STTCOMMS_API void sttcomms_free_context_params(struct sttcomms_context_params * params);

    // Convert RAW PCM audio to log mel spectrogram.
    // The resulting spectrogram is stored inside the default state of the provided sttcomms context.
    // Returns 0 on success
    STTCOMMS_API int sttcomms_pcm_to_mel(
            struct sttcomms_context * ctx,
                       const float * samples,
                               int   n_samples,
                               int   n_threads);

    STTCOMMS_API int sttcomms_pcm_to_mel_with_state(
            struct sttcomms_context * ctx,
              struct sttcomms_state * state,
                       const float * samples,
                               int   n_samples,
                               int   n_threads);

    // This can be used to set a custom log mel spectrogram inside the default state of the provided sttcomms context.
    // Use this instead of sttcomms_pcm_to_mel() if you want to provide your own log mel spectrogram.
    // n_mel must be 80
    // Returns 0 on success
    STTCOMMS_API int sttcomms_set_mel(
            struct sttcomms_context * ctx,
                       const float * data,
                               int   n_len,
                               int   n_mel);

    STTCOMMS_API int sttcomms_set_mel_with_state(
            struct sttcomms_context * ctx,
              struct sttcomms_state * state,
                       const float * data,
                               int   n_len,
                               int   n_mel);

    // Run the Sttcomms encoder on the log mel spectrogram stored inside the default state in the provided sttcomms context.
    // Make sure to call sttcomms_pcm_to_mel() or sttcomms_set_mel() first.
    // offset can be used to specify the offset of the first frame in the spectrogram.
    // Returns 0 on success
    STTCOMMS_API int sttcomms_encode(
            struct sttcomms_context * ctx,
                               int   offset,
                               int   n_threads);

    STTCOMMS_API int sttcomms_encode_with_state(
            struct sttcomms_context * ctx,
              struct sttcomms_state * state,
                               int   offset,
                               int   n_threads);

    // Run the Sttcomms decoder to obtain the logits and probabilities for the next token.
    // Make sure to call sttcomms_encode() first.
    // tokens + n_tokens is the provided context for the decoder.
    // n_past is the number of tokens to use from previous decoder calls.
    // Returns 0 on success
    // TODO: add support for multiple decoders
    STTCOMMS_API int sttcomms_decode(
            struct sttcomms_context * ctx,
               const sttcomms_token * tokens,
                               int   n_tokens,
                               int   n_past,
                               int   n_threads);

    STTCOMMS_API int sttcomms_decode_with_state(
            struct sttcomms_context * ctx,
              struct sttcomms_state * state,
               const sttcomms_token * tokens,
                               int   n_tokens,
                               int   n_past,
                               int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    STTCOMMS_API int sttcomms_tokenize(
            struct sttcomms_context * ctx,
                        const char * text,
                     sttcomms_token * tokens,
                               int   n_max_tokens);

    // Return the number of tokens in the provided text
    // Equivalent to: -sttcomms_tokenize(ctx, text, NULL, 0)
    int sttcomms_token_count(struct sttcomms_context * ctx, const char * text);

    // Largest language id (i.e. number of available languages - 1)
    STTCOMMS_API int sttcomms_lang_max_id(void);

    // Return the id of the specified language, returns -1 if not found
    // Examples:
    //   "de" -> 2
    //   "german" -> 2
    STTCOMMS_API int sttcomms_lang_id(const char * lang);

    // Return the short string of the specified language id (e.g. 2 -> "de"), returns nullptr if not found
    STTCOMMS_API const char * sttcomms_lang_str(int id);

    // Return the short string of the specified language name (e.g. 2 -> "german"), returns nullptr if not found
    STTCOMMS_API const char * sttcomms_lang_str_full(int id);

    // Use mel data at offset_ms to try and auto-detect the spoken language
    // Make sure to call sttcomms_pcm_to_mel() or sttcomms_set_mel() first
    // Returns the top language id or negative on failure
    // If not null, fills the lang_probs array with the probabilities of all languages
    // The array must be sttcomms_lang_max_id() + 1 in size
    // ref: https://github.com/openai/sttcomms/blob/main/sttcomms/decoding.py#L18-L69
    STTCOMMS_API int sttcomms_lang_auto_detect(
            struct sttcomms_context * ctx,
                               int   offset_ms,
                               int   n_threads,
                             float * lang_probs);

    STTCOMMS_API int sttcomms_lang_auto_detect_with_state(
            struct sttcomms_context * ctx,
              struct sttcomms_state * state,
                               int   offset_ms,
                               int   n_threads,
                             float * lang_probs);

    STTCOMMS_API int sttcomms_n_len           (struct sttcomms_context * ctx); // mel length
    STTCOMMS_API int sttcomms_n_len_from_state(struct sttcomms_state * state); // mel length
    STTCOMMS_API int sttcomms_n_vocab         (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_n_text_ctx      (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_n_audio_ctx     (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_is_multilingual (struct sttcomms_context * ctx);

    STTCOMMS_API int sttcomms_model_n_vocab      (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_audio_ctx  (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_audio_state(struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_audio_head (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_audio_layer(struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_text_ctx   (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_text_state (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_text_head  (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_text_layer (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_n_mels       (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_ftype        (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_model_type         (struct sttcomms_context * ctx);

    // Token logits obtained from the last call to sttcomms_decode()
    // The logits for the last token are stored in the last row
    // Rows: n_tokens
    // Cols: n_vocab
    STTCOMMS_API float * sttcomms_get_logits           (struct sttcomms_context * ctx);
    STTCOMMS_API float * sttcomms_get_logits_from_state(struct sttcomms_state * state);

    // Token Id -> String. Uses the vocabulary in the provided context
    STTCOMMS_API const char * sttcomms_token_to_str(struct sttcomms_context * ctx, sttcomms_token token);
    STTCOMMS_API const char * sttcomms_model_type_readable(struct sttcomms_context * ctx);


    // Special tokens
    STTCOMMS_API sttcomms_token sttcomms_token_eot (struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_sot (struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_solm(struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_prev(struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_nosp(struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_not (struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_beg (struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_lang(struct sttcomms_context * ctx, int lang_id);

    // Task tokens
    STTCOMMS_API sttcomms_token sttcomms_token_translate (struct sttcomms_context * ctx);
    STTCOMMS_API sttcomms_token sttcomms_token_transcribe(struct sttcomms_context * ctx);

    // Performance information from the default state.
    struct sttcomms_timings {
        float sample_ms;
        float encode_ms;
        float decode_ms;
        float batchd_ms;
        float prompt_ms;
    };
    STTCOMMS_API struct sttcomms_timings * sttcomms_get_timings(struct sttcomms_context * ctx);
    STTCOMMS_API void sttcomms_print_timings(struct sttcomms_context * ctx);
    STTCOMMS_API void sttcomms_reset_timings(struct sttcomms_context * ctx);

    // Print system information
    STTCOMMS_API const char * sttcomms_print_system_info(void);

    ////////////////////////////////////////////////////////////////////////////

    // Available sampling strategies
    enum sttcomms_sampling_strategy {
        STTCOMMS_SAMPLING_GREEDY,      // similar to OpenAI's GreedyDecoder
        STTCOMMS_SAMPLING_BEAM_SEARCH, // similar to OpenAI's BeamSearchDecoder
    };

    // Text segment callback
    // Called on every newly generated text segment
    // Use the sttcomms_full_...() functions to obtain the text segments
    typedef void (*sttcomms_new_segment_callback)(struct sttcomms_context * ctx, struct sttcomms_state * state, int n_new, void * user_data);

    // Progress callback
    typedef void (*sttcomms_progress_callback)(struct sttcomms_context * ctx, struct sttcomms_state * state, int progress, void * user_data);

    // Encoder begin callback
    // If not NULL, called before the encoder starts
    // If it returns false, the computation is aborted
    typedef bool (*sttcomms_encoder_begin_callback)(struct sttcomms_context * ctx, struct sttcomms_state * state, void * user_data);

    // Logits filter callback
    // Can be used to modify the logits before sampling
    // If not NULL, called after applying temperature to logits
    typedef void (*sttcomms_logits_filter_callback)(
            struct sttcomms_context * ctx,
              struct sttcomms_state * state,
          const sttcomms_token_data * tokens,
                               int   n_tokens,
                             float * logits,
                              void * user_data);

    // Parameters for the sttcomms_full() function
    // If you change the order or add new parameters, make sure to update the default values in sttcomms.cpp:
    // sttcomms_full_default_params()
    struct sttcomms_full_params {
        enum sttcomms_sampling_strategy strategy;

        int n_threads;
        int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
        int offset_ms;          // start offset in ms
        int duration_ms;        // audio duration to process in ms

        bool translate;
        bool no_context;        // do not use past transcription (if any) as initial prompt for the decoder
        bool no_timestamps;     // do not generate timestamps
        bool single_segment;    // force single segment output (useful for streaming)
        bool print_special;     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
        bool print_progress;    // print progress information
        bool print_realtime;    // print results from within sttcomms.cpp (avoid it, use callback instead)
        bool print_timestamps;  // print timestamps for each text segment when printing realtime

        // [EXPERIMENTAL] token-level timestamps
        bool  token_timestamps; // enable token-level timestamps
        float thold_pt;         // timestamp token probability threshold (~0.01)
        float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
        int   max_len;          // max segment length in characters
        bool  split_on_word;    // split on word rather than on token (when used with max_len)
        int   max_tokens;       // max tokens per segment (0 = no limit)

        // [EXPERIMENTAL] speed-up techniques
        // note: these can significantly reduce the quality of the output
        bool debug_mode;        // enable debug_mode provides extra info (eg. Dump log_mel)
        int  audio_ctx;         // overwrite the audio context size (0 = use default)

        // [EXPERIMENTAL] [TDRZ] tinydiarize
        bool tdrz_enable;       // enable tinydiarize speaker turn detection

        // A regular expression that matches tokens to suppress
        const char * suppress_regex;

        // tokens to provide to the sttcomms decoder as initial prompt
        // these are prepended to any existing text context from a previous call
        // use sttcomms_tokenize() to convert text to tokens
        // maximum of sttcomms_n_text_ctx()/2 tokens are used (typically 224)
        const char * initial_prompt;
        bool carry_initial_prompt; // if true, always prepend initial_prompt to every decode window (may reduce conditioning on previous text)
        const sttcomms_token * prompt_tokens;
        int prompt_n_tokens;

        // for auto-detection, set to nullptr, "" or "auto"
        const char * language;
        bool detect_language;

        // common decoding parameters:
        bool suppress_blank; // ref: https://github.com/openai/sttcomms/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/sttcomms/decoding.py#L89
        bool suppress_nst;   // non-speech tokens, ref: https://github.com/openai/sttcomms/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/sttcomms/tokenizer.py#L224-L253

        float temperature;      // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
        float max_initial_ts;   // ref: https://github.com/openai/sttcomms/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/sttcomms/decoding.py#L97
        float length_penalty;   // ref: https://github.com/openai/sttcomms/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/sttcomms/transcribe.py#L267

        // fallback parameters
        // ref: https://github.com/openai/sttcomms/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/sttcomms/transcribe.py#L274-L278
        float temperature_inc;
        float entropy_thold;    // similar to OpenAI's "compression_ratio_threshold"
        float logprob_thold;
        float no_speech_thold;

        struct {
            int best_of;    // ref: https://github.com/openai/sttcomms/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/sttcomms/transcribe.py#L264
        } greedy;

        struct {
            int beam_size;  // ref: https://github.com/openai/sttcomms/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/sttcomms/transcribe.py#L265

            float patience; // TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf
        } beam_search;

        // called for every newly generated text segment
        sttcomms_new_segment_callback new_segment_callback;
        void * new_segment_callback_user_data;

        // called on each progress update
        sttcomms_progress_callback progress_callback;
        void * progress_callback_user_data;

        // called each time before the encoder starts
        sttcomms_encoder_begin_callback encoder_begin_callback;
        void * encoder_begin_callback_user_data;

        // called each time before ggml computation starts
        ggml_abort_callback abort_callback;
        void * abort_callback_user_data;

        // called by each decoder to filter obtained logits
        sttcomms_logits_filter_callback logits_filter_callback;
        void * logits_filter_callback_user_data;

        const sttcomms_grammar_element ** grammar_rules;
        size_t                           n_grammar_rules;
        size_t                           i_start_rule;
        float                            grammar_penalty;

        // Voice Activity Detection (VAD) params
        bool         vad;                         // Enable VAD
        const char * vad_model_path;              // Path to VAD model

        sttcomms_vad_params vad_params;
    };

    // NOTE: this function allocates memory, and it is the responsibility of the caller to free the pointer - see sttcomms_free_context_params & sttcomms_free_params()
    STTCOMMS_API struct sttcomms_context_params * sttcomms_context_default_params_by_ref(void);
    STTCOMMS_API struct sttcomms_context_params   sttcomms_context_default_params       (void);

    STTCOMMS_API struct sttcomms_full_params * sttcomms_full_default_params_by_ref(enum sttcomms_sampling_strategy strategy);
    STTCOMMS_API struct sttcomms_full_params   sttcomms_full_default_params       (enum sttcomms_sampling_strategy strategy);

    // Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
    // Not thread safe for same context
    // Uses the specified decoding strategy to obtain the text.
    STTCOMMS_API int sttcomms_full(
                struct sttcomms_context * ctx,
            struct sttcomms_full_params   params,
                           const float * samples,
                                   int   n_samples);

    STTCOMMS_API int sttcomms_full_with_state(
                struct sttcomms_context * ctx,
                  struct sttcomms_state * state,
            struct sttcomms_full_params   params,
                           const float * samples,
                                   int   n_samples);

    // Split the input audio in chunks and process each chunk separately using sttcomms_full_with_state()
    // Result is stored in the default state of the context
    // Not thread safe if executed in parallel on the same context.
    // It seems this approach can offer some speedup in some cases.
    // However, the transcription accuracy can be worse at the beginning and end of each chunk.
    STTCOMMS_API int sttcomms_full_parallel(
                struct sttcomms_context * ctx,
            struct sttcomms_full_params   params,
                           const float * samples,
                                   int   n_samples,
                                   int   n_processors);

    // Number of generated text segments
    // A segment can be a few words, a sentence, or even a paragraph.
    STTCOMMS_API int sttcomms_full_n_segments           (struct sttcomms_context * ctx);
    STTCOMMS_API int sttcomms_full_n_segments_from_state(struct sttcomms_state * state);

    // Language id associated with the context's default state
    STTCOMMS_API int sttcomms_full_lang_id(struct sttcomms_context * ctx);

    // Language id associated with the provided state
    STTCOMMS_API int sttcomms_full_lang_id_from_state(struct sttcomms_state * state);

    // Get the start and end time of the specified segment
    STTCOMMS_API int64_t sttcomms_full_get_segment_t0           (struct sttcomms_context * ctx, int i_segment);
    STTCOMMS_API int64_t sttcomms_full_get_segment_t0_from_state(struct sttcomms_state * state, int i_segment);

    STTCOMMS_API int64_t sttcomms_full_get_segment_t1           (struct sttcomms_context * ctx, int i_segment);
    STTCOMMS_API int64_t sttcomms_full_get_segment_t1_from_state(struct sttcomms_state * state, int i_segment);

    // Get whether the next segment is predicted as a speaker turn
    STTCOMMS_API bool sttcomms_full_get_segment_speaker_turn_next(struct sttcomms_context * ctx, int i_segment);
    STTCOMMS_API bool sttcomms_full_get_segment_speaker_turn_next_from_state(struct sttcomms_state * state, int i_segment);

    // Get the text of the specified segment
    STTCOMMS_API const char * sttcomms_full_get_segment_text           (struct sttcomms_context * ctx, int i_segment);
    STTCOMMS_API const char * sttcomms_full_get_segment_text_from_state(struct sttcomms_state * state, int i_segment);

    // Get number of tokens in the specified segment
    STTCOMMS_API int sttcomms_full_n_tokens           (struct sttcomms_context * ctx, int i_segment);
    STTCOMMS_API int sttcomms_full_n_tokens_from_state(struct sttcomms_state * state, int i_segment);

    // Get the token text of the specified token in the specified segment
    STTCOMMS_API const char * sttcomms_full_get_token_text           (struct sttcomms_context * ctx, int i_segment, int i_token);
    STTCOMMS_API const char * sttcomms_full_get_token_text_from_state(struct sttcomms_context * ctx, struct sttcomms_state * state, int i_segment, int i_token);

    STTCOMMS_API sttcomms_token sttcomms_full_get_token_id           (struct sttcomms_context * ctx, int i_segment, int i_token);
    STTCOMMS_API sttcomms_token sttcomms_full_get_token_id_from_state(struct sttcomms_state * state, int i_segment, int i_token);

    // Get token data for the specified token in the specified segment
    // This contains probabilities, timestamps, etc.
    STTCOMMS_API sttcomms_token_data sttcomms_full_get_token_data           (struct sttcomms_context * ctx, int i_segment, int i_token);
    STTCOMMS_API sttcomms_token_data sttcomms_full_get_token_data_from_state(struct sttcomms_state * state, int i_segment, int i_token);

    // Get the start/end time of the specified token, in centiseconds. When VAD is enabled
    // these are mapped back to the original audio timeline (a token landing in a removed
    // inter-segment silence snaps to the nearest speech boundary), unlike
    // sttcomms_full_get_token_data().t0/t1 which stay in VAD-processed time. Without VAD the
    // raw token times are returned unchanged.
    STTCOMMS_API int64_t sttcomms_full_get_token_t0           (struct sttcomms_context * ctx, int i_segment, int i_token);
    STTCOMMS_API int64_t sttcomms_full_get_token_t0_from_state(struct sttcomms_state * state, int i_segment, int i_token);
    STTCOMMS_API int64_t sttcomms_full_get_token_t1           (struct sttcomms_context * ctx, int i_segment, int i_token);
    STTCOMMS_API int64_t sttcomms_full_get_token_t1_from_state(struct sttcomms_state * state, int i_segment, int i_token);

    // Get the probability of the specified token in the specified segment
    STTCOMMS_API float sttcomms_full_get_token_p           (struct sttcomms_context * ctx, int i_segment, int i_token);
    STTCOMMS_API float sttcomms_full_get_token_p_from_state(struct sttcomms_state * state, int i_segment, int i_token);

    // Access the speech segments detected by the internal VAD (only when params.vad = true).
    // Times are on the original audio timeline, in centiseconds. The count is 0 when VAD was
    // not used, so callers can reuse sttcomms's own speech boundaries instead of running a
    // separate VAD pass.
    STTCOMMS_API int     sttcomms_full_n_vad_segments               (struct sttcomms_context * ctx);
    STTCOMMS_API int     sttcomms_full_n_vad_segments_from_state    (struct sttcomms_state * state);
    STTCOMMS_API int64_t sttcomms_full_get_vad_segment_t0           (struct sttcomms_context * ctx, int i);
    STTCOMMS_API int64_t sttcomms_full_get_vad_segment_t0_from_state(struct sttcomms_state * state, int i);
    STTCOMMS_API int64_t sttcomms_full_get_vad_segment_t1           (struct sttcomms_context * ctx, int i);
    STTCOMMS_API int64_t sttcomms_full_get_vad_segment_t1_from_state(struct sttcomms_state * state, int i);

    //
    // Voice Activity Detection (VAD)
    //

    struct sttcomms_vad_context;

    STTCOMMS_API struct sttcomms_vad_params sttcomms_vad_default_params(void);

    struct sttcomms_vad_context_params {
        int   n_threads;  // The number of threads to use for processing.
        bool  use_gpu;
        int   gpu_device; // CUDA device
    };

    STTCOMMS_API struct sttcomms_vad_context_params sttcomms_vad_default_context_params(void);

    STTCOMMS_API struct sttcomms_vad_context * sttcomms_vad_init_from_file_with_params(const char * path_model,              struct sttcomms_vad_context_params params);
    STTCOMMS_API struct sttcomms_vad_context * sttcomms_vad_init_with_params          (struct sttcomms_model_loader * loader, struct sttcomms_vad_context_params params);

    STTCOMMS_API bool sttcomms_vad_detect_speech(
            struct sttcomms_vad_context * vctx,
                           const float * samples,
                                   int   n_samples);

    // Like sttcomms_vad_detect_speech, but does not reset LSTM state.
    // Use for streaming: call sttcomms_vad_reset_state() between utterances.
    STTCOMMS_API bool sttcomms_vad_detect_speech_no_reset(
            struct sttcomms_vad_context * vctx,
                           const float * samples,
                                   int   n_samples);

    // Reset LSTM hidden/cell states to zero.
    STTCOMMS_API void sttcomms_vad_reset_state(struct sttcomms_vad_context * vctx);

    STTCOMMS_API int     sttcomms_vad_n_probs(struct sttcomms_vad_context * vctx);
    STTCOMMS_API float * sttcomms_vad_probs  (struct sttcomms_vad_context * vctx);

    struct sttcomms_vad_segments;

    STTCOMMS_API struct sttcomms_vad_segments * sttcomms_vad_segments_from_probs(
            struct sttcomms_vad_context * vctx,
            struct sttcomms_vad_params    params);

    STTCOMMS_API struct sttcomms_vad_segments * sttcomms_vad_segments_from_samples(
            struct sttcomms_vad_context * vctx,
            struct sttcomms_vad_params    params,
                           const float * samples,
                                   int   n_samples);

    STTCOMMS_API int sttcomms_vad_segments_n_segments(struct sttcomms_vad_segments * segments);

    STTCOMMS_API float sttcomms_vad_segments_get_segment_t0(struct sttcomms_vad_segments * segments, int i_segment);
    STTCOMMS_API float sttcomms_vad_segments_get_segment_t1(struct sttcomms_vad_segments * segments, int i_segment);

    STTCOMMS_API void sttcomms_vad_free_segments(struct sttcomms_vad_segments * segments);
    STTCOMMS_API void sttcomms_vad_free         (struct sttcomms_vad_context  * ctx);

    ////////////////////////////////////////////////////////////////////////////

    // Temporary helpers needed for exposing ggml interface

    STTCOMMS_API int          sttcomms_bench_memcpy          (int n_threads);
    STTCOMMS_API const char * sttcomms_bench_memcpy_str      (int n_threads);
    STTCOMMS_API int          sttcomms_bench_ggml_mul_mat    (int n_threads);
    STTCOMMS_API const char * sttcomms_bench_ggml_mul_mat_str(int n_threads);

    // Control logging output; default behavior is to print to stderr

    STTCOMMS_API void sttcomms_log_set(ggml_log_callback log_callback, void * user_data);

    // Get the no_speech probability for the specified segment
    STTCOMMS_API float sttcomms_full_get_segment_no_speech_prob           (struct sttcomms_context * ctx, int i_segment);
    STTCOMMS_API float sttcomms_full_get_segment_no_speech_prob_from_state(struct sttcomms_state * state, int i_segment);
#ifdef __cplusplus
}
#endif

#endif
