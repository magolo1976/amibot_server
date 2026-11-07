#ifndef STT_COMMS_H
#define STT_COMMS_H

#include <string>
#include <vector>
#include <cstdint>

#if defined(_WIN32) && defined(STT_COMMS_BUILD_SHARED)
    #define STT_COMMS_API __declspec(dllexport)
#elif defined(_WIN32) && defined(STT_COMMS_USE_SHARED)
    #define STT_COMMS_API __declspec(dllimport)
#else
    #define STT_COMMS_API
#endif

namespace stt_comms {

// Limpia el texto transcrito eliminando etiquetas de ruido ([BLANK_AUDIO], etc.),
// emojis y caracteres especiales, dejando solo texto limpio de cualquier idioma.
STT_COMMS_API std::string clean_text(const std::string& input);

// Decodifica un buffer de bytes .wav en memoria (PCM 8/16-bit o Float 32-bit, mono o estereo, cualquier frecuencia)
// y lo convierte automaticamente a muestras float32 mono a 16000 Hz.
STT_COMMS_API bool read_wav_bytes_16k_mono(const uint8_t* wav_bytes, size_t n_bytes, std::vector<float>& out_samples, std::string& out_error);

// Lee un archivo .wav (PCM 16-bit o Float 32-bit, mono o estereo, cualquier frecuencia)
// y lo convierte automaticamente a muestras float32 mono a 16000 Hz.
STT_COMMS_API bool read_wav_16k_mono(const std::string& wav_path, std::vector<float>& out_samples, std::string& out_error);

struct Config {
    std::string model_path;
    bool use_gpu = true;
    int32_t n_threads = 0;       // 0 = auto-detectar
    bool verbose = false;
};

struct TranscribeOptions {
    std::string language = "es"; // "es", "en", "auto", etc.
    bool translate = false;      // Traducir a ingles si es true
    int32_t max_tokens = 0;      // 0 = sin limite
};

class STT_COMMS_API STTClient {
public:
    STTClient();
    ~STTClient();

    STTClient(const STTClient&) = delete;
    STTClient& operator=(const STTClient&) = delete;

    // Inicializa el motor y carga el modelo de voz a texto en memoria
    bool init(const Config& config);

    // Retorna true si el modelo esta cargado
    bool is_ready() const;

    // Libera recursos del modelo
    void close();

    // Transcribe un vector/buffer de muestras PCM float32 a 16000 Hz mono
    std::string transcribe_samples(const float* samples, size_t n_samples, const TranscribeOptions& options = {});

    // Transcribe un buffer de bytes .wav en memoria y retorna el texto transcrito limpio
    std::string transcribe_wav_bytes(const uint8_t* wav_bytes, size_t n_bytes, const TranscribeOptions& options = {});

    // Lee un archivo .wav en disco y retorna el texto transcrito limpio
    std::string transcribe_wav(const std::string& wav_path, const TranscribeOptions& options = {});

    // Retorna el ultimo mensaje de error
    std::string get_last_error() const;

private:
    class Impl;
    Impl* pimpl;
};

// ========================================================================
// Interfaz C pura para enlaces externos (Node.js FFI, Python ctypes, C++)
// ========================================================================
extern "C" {
    typedef void* stt_comms_handle;

    STT_COMMS_API stt_comms_handle stt_comms_create();
    STT_COMMS_API int stt_comms_init(stt_comms_handle handle, const char* model_path, int use_gpu, int n_threads);
    STT_COMMS_API int stt_comms_transcribe_wav(stt_comms_handle handle, const char* wav_path, const char* language, char* out_buf, int max_out_len);
    STT_COMMS_API int stt_comms_transcribe_pcm(stt_comms_handle handle, const float* samples, int n_samples, const char* language, char* out_buf, int max_out_len);
    STT_COMMS_API void stt_comms_destroy(stt_comms_handle handle);
}

} // namespace stt_comms

#endif // STT_COMMS_H
