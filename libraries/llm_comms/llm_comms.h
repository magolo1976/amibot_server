#ifndef LLM_COMMS_H
#define LLM_COMMS_H

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

#if defined(_WIN32) && defined(LLM_COMMS_BUILD_SHARED)
    #define LLM_COMMS_API __declspec(dllexport)
#elif defined(_WIN32) && defined(LLM_COMMS_USE_SHARED)
    #define LLM_COMMS_API __declspec(dllimport)
#else
    #define LLM_COMMS_API
#endif

namespace llm_comms {

// Limpia el texto eliminando emojis, simbolos especiales, markdown y caracteres de control,
// conservando unicamente letras de cualquier idioma, numeros y puntuacion basica limpia.
LLM_COMMS_API std::string clean_text(const std::string& input);

struct Config {
    std::string model_path;
    int32_t n_ctx = 2048;            // Tamaño de contexto en tokens
    int32_t n_gpu_layers = 99;        // Capas a descargar a GPU (0 = solo CPU, 99 = todas las posibles)
    int32_t n_threads = 0;            // Hilos de CPU (0 = auto-detectar)
    bool verbose = false;             // Mostrar logs detallados del motor
};

struct GenerateOptions {
    int32_t max_tokens = 256;         // Máximo de tokens a generar en la respuesta
    float temperature = 0.7f;         // Creatividad (0.0 = determinista/greedy)
    float top_p = 0.9f;               // Muestreo Top-P
    // Callback opcional de streaming token por token (ya limpio)
    // Retornar 'false' en el callback aborta la generación tempranamente
    std::function<bool(const std::string& token)> stream_cb = nullptr;
};

struct ChatMessage {
    std::string role;    // "system", "user", "assistant"
    std::string content;
};

class LLM_COMMS_API LLMClient {
public:
    LLMClient();
    ~LLMClient();

    LLMClient(const LLMClient&) = delete;
    LLMClient& operator=(const LLMClient&) = delete;

    // Inicializa el backend y carga el modelo GGUF en memoria
    bool init(const Config& config);

    // Retorna true si el modelo está cargado y listo
    bool is_ready() const;

    // Cierra la sesión y libera modelo y contexto
    void close();

    // Envía un texto / prompt plano y retorna la respuesta limpia generada
    std::string send_text(const std::string& prompt, const GenerateOptions& options = {});

    // Envía un historial de chat aplicando la plantilla nativa del modelo y retorna texto limpio
    std::string send_chat(const std::vector<ChatMessage>& messages, const GenerateOptions& options = {});

    // Formatea mensajes de chat según la plantilla del modelo a un prompt plano
    std::string format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_assistant = true) const;

    // Retorna el último mensaje de error si ocurrió alguno
    std::string get_last_error() const;

private:
    class Impl;
    Impl* pimpl;
};

// ========================================================================
// Interfaz C pura para enlaces externos (Node.js FFI, Python ctypes, etc.)
// ========================================================================
extern "C" {
    typedef void* llm_comms_handle;

    LLM_COMMS_API llm_comms_handle llm_comms_create();
    LLM_COMMS_API int llm_comms_init(llm_comms_handle handle, const char* model_path, int n_ctx, int n_gpu_layers, int n_threads);
    LLM_COMMS_API int llm_comms_send_text(llm_comms_handle handle, const char* prompt, char* out_buf, int max_out_len, int max_tokens, float temperature);
    LLM_COMMS_API int llm_comms_send_chat_simple(llm_comms_handle handle, const char* system_prompt, const char* user_prompt, char* out_buf, int max_out_len, int max_tokens, float temperature);
    LLM_COMMS_API void llm_comms_destroy(llm_comms_handle handle);
}

} // namespace llm_comms

#endif // LLM_COMMS_H
