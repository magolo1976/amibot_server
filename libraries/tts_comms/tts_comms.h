#ifndef TTS_COMMS_H
#define TTS_COMMS_H

#include <string>
#include <vector>
#include <cstdint>

#if defined(_WIN32) && defined(TTS_COMMS_BUILD_SHARED)
    #define TTS_COMMS_API __declspec(dllexport)
#elif defined(_WIN32) && defined(TTS_COMMS_USE_SHARED)
    #define TTS_COMMS_API __declspec(dllimport)
#else
    #define TTS_COMMS_API
#endif

namespace tts_comms {

constexpr uint32_t TTSCOMMS_SAMPLE_RATE = 24000;
constexpr size_t TTSCOMMS_STYLE_DIM = 256;
constexpr size_t TTSCOMMS_MAX_STYLE_FRAMES = 510;

// Limpia el texto de entrada eliminando emojis, markdown y caracteres especiales,
// dejando solo texto limpio de cualquier idioma apto para sintesis de voz.
TTS_COMMS_API std::string clean_text(const std::string& input);

// Divide un texto largo en frases completas respetando abreviaturas, numeros decimales y puntuacion.
TTS_COMMS_API std::vector<std::string> split_sentences(const std::string& text);

// Codifica muestras float32 [-1.0, 1.0] en un buffer de bytes con formato RIFF/WAVE (PCM 16-bit mono).
TTS_COMMS_API std::vector<uint8_t> encode_wav_bytes(const float* samples, size_t n_samples, uint32_t sample_rate = TTSCOMMS_SAMPLE_RATE);

// Guarda muestras float32 [-1.0, 1.0] en un archivo .wav (PCM 16-bit mono) en disco.
TTS_COMMS_API bool write_wav_file(const std::string& wav_path, const float* samples, size_t n_samples, uint32_t sample_rate = TTSCOMMS_SAMPLE_RATE);

struct Config {
    std::string voices_dir;            // Carpeta donde residen los archivos <voice>.bin
    std::string default_voice = "ef_dora";
    bool verbose = false;
};

struct SynthesizeOptions {
    std::string voice = "ef_dora";     // ej. "ef_dora", "em_alex", "af_heart", etc.
    float speed = 1.0f;                // Velocidad de locucion (0.5 - 2.0)
};

class TTS_COMMS_API TTSClient {
public:
    TTSClient();
    ~TTSClient();

    TTSClient(const TTSClient&) = delete;
    TTSClient& operator=(const TTSClient&) = delete;

    // Inicializa el motor TTS y verifica el directorio de voces (.bin)
    bool init(const Config& config);

    // Retorna true si el cliente esta inicializado y listo
    bool is_ready() const;

    // Cierra y libera la cache de voces en memoria
    void close();

    // Lista todos los identificadores de voz disponibles en voices_dir
    std::vector<std::string> list_voices() const;

    // Comprueba si una voz existe en el catalogo cargado
    bool has_voice(const std::string& voice) const;

    // Obtiene el vector de estilo de 256 dimensiones para una voz y longitud de tokens dada
    bool get_voice_style(const std::string& voice, size_t token_count, std::vector<float>& out_style);

    // Sintetiza texto limpio a muestras de audio PCM float32 mono a 24000 Hz
    bool synthesize_to_pcm(const std::string& text, std::vector<float>& out_samples, const SynthesizeOptions& options = {});

    // Sintetiza texto limpio directamente a un archivo .wav en disco
    bool synthesize_to_wav(const std::string& text, const std::string& wav_path, const SynthesizeOptions& options = {});

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
    typedef void* tts_comms_handle;

    TTS_COMMS_API tts_comms_handle tts_comms_create();
    TTS_COMMS_API int tts_comms_init(tts_comms_handle handle, const char* voices_dir, const char* default_voice);
    TTS_COMMS_API int tts_comms_synthesize_wav(tts_comms_handle handle, const char* text, const char* voice, float speed, const char* out_wav_path);
    TTS_COMMS_API int tts_comms_synthesize_pcm(tts_comms_handle handle, const char* text, const char* voice, float speed, float* out_samples, int max_samples);
    TTS_COMMS_API void tts_comms_destroy(tts_comms_handle handle);
}

} // namespace tts_comms

#endif // TTS_COMMS_H
