#ifndef LLM_API_CORE_H
#define LLM_API_CORE_H

#include "llm_comms.h"
#include "stt_comms.h"
#include "tts_comms.h"
#include <string>
#include <vector>
#include <mutex>

namespace llm_api {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    bool valid = false;
    std::string query;
    std::string api_key; // Credencial recibida (X-API-Key, Authorization: Bearer o ?api_key=)
};

struct HttpResponse {
    int status_code = 200;
    std::string status_text = "OK";
    std::string content_type = "application/json; charset=utf-8";
    std::string extra_headers;
    std::string body;
};

struct GenerateRequest {
    std::string prompt;
    std::string system_prompt;
    std::string model;
    int max_tokens = 128;
    float temperature = 0.7f;
};

struct GenerateResult {
    bool success = false;
    std::string model_used;
    std::string response_text;
    std::string error_message;
    long long elapsed_ms = 0;
};

struct VoiceRequest {
    std::string wav_bytes;         // Bytes crudos en RAM (.wav, PCM 16-bit de I2S, o tramas ESP32 de 4 bytes)
    int input_sample_rate = 16000; // Frecuencia de entrada si es PCM crudo sin cabecera WAV (ej. 16000 Hz en ESP32)
    int output_sample_rate = 24000;// Frecuencia de salida deseada (24000 Hz por defecto, o 16000 Hz para ESP32)
    bool raw_pcm_output = false;   // true = devolver PCM 16-bit crudo sin cabecera WAV (para i2s_write directo en ESP32)
    bool esp32_framed_output = false; // true = empaquetar salida en tramas ESP32 con cabecera de 4 bytes [0,0,len_BE]
    std::string language = "auto"; // Idioma STT ("es", "en", "auto")
    std::string model;             // Modelo LLM ("qwen", "gpt2", etc.)
    std::string voice = "ef_dora"; // Voz TTS ("ef_dora", "em_alex", "af_heart", etc.)
    float speed = 1.0f;            // Velocidad TTS
    int max_tokens = 64;
    float temperature = 0.7f;
};

struct VoiceResult {
    bool success = false;
    std::string transcribed_text;
    std::string llm_response;
    std::string emotion = "neutral";
    std::string model_used;
    std::string voice_used;
    std::vector<uint8_t> audio_wav;
    std::string error_message;
    long long elapsed_ms = 0;
};

// Empaqueta/desempaqueta tramas binarias del protocolo ESP32 ([tipo(1B), res(1B), len_BE(2B)] + payload)
std::string unwrap_esp32_audio_frames(const std::string& raw_bytes);
std::vector<uint8_t> pack_esp32_audio_frames(const uint8_t* data, size_t len, size_t chunk_size = 120);

// ============================================================================
// Funciones utilitarias JSON, Query y HTTP (100% libres de dependencias externas)
// ============================================================================
std::string json_escape(const std::string& input);
std::string json_extract_string(const std::string& json, const std::string& key);
int json_extract_int(const std::string& json, const std::string& key, int default_val);
float json_extract_float(const std::string& json, const std::string& key, float default_val);
std::string query_extract_param(const std::string& query, const std::string& key, const std::string& default_val = "");

HttpRequest parse_http_request(const std::string& raw_request);

// Autenticacion por API key. Si 'expected_key' esta vacio la autenticacion queda desactivada.
// La comparacion es en tiempo constante para evitar ataques de temporizacion.
bool is_request_authorized(const HttpRequest& req, const std::string& expected_key);
// Lee la clave configurada en la variable de entorno API_KEY (vacia = sin autenticacion).
std::string configured_api_key();
std::string build_http_response(const HttpResponse& resp);
std::string build_http_response(int status_code, const std::string& status_text, const std::string& json_body);

// ============================================================================
// Servicio principal de la API conectado con stt_comms -> llm_comms -> tts_comms
// ============================================================================
class ApiService {
public:
    explicit ApiService(const std::string& models_dir = "", int n_gpu_layers = 99, int n_ctx = 2048);
    ~ApiService();

    static std::string find_models_directory(const std::string& preferred_dir = "");
    static bool is_instruct_model(const std::string& model_name_or_path);
    static std::string detect_emotion(const std::string& text);

    std::vector<std::string> list_models() const;
    std::string resolve_model_path(const std::string& model_query) const;
    std::string resolve_stt_model_path() const;

    bool ensure_model_loaded(const std::string& model_query);
    bool ensure_stt_loaded();
    bool ensure_tts_loaded();

    std::string get_loaded_model_name() const;
    std::string get_models_directory() const;

    // Flujo Texto -> LLM -> Texto
    GenerateResult generate_text(const GenerateRequest& req);

    // Flujo completo Voz -> STT (stt_comms) -> LLM (llm_comms) -> TTS (tts_comms) -> Voz (.wav)
    VoiceResult process_voice(const VoiceRequest& req);

    // Procesa una peticion HTTP y retorna su respuesta HTTP (audio/wav o application/json)
    HttpResponse handle_http_request(const HttpRequest& req);

private:
    GenerateResult generate_text_unlocked(const GenerateRequest& req);

    std::string models_dir_;
    int n_gpu_layers_;
    int n_ctx_;
    std::string loaded_model_path_;
    std::string loaded_model_name_;
    std::string loaded_stt_path_;
    llm_comms::LLMClient client_;
    stt_comms::STTClient stt_client_;
    tts_comms::TTSClient tts_client_;
    mutable std::mutex mutex_;
};

} // namespace llm_api

#endif // LLM_API_CORE_H
