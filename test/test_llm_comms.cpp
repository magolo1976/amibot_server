#include "llm_comms.h"
#include <iostream>
#include <string>
#include <filesystem>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "   llm_comms - Prueba de Comunicacion   " << std::endl;
    std::cout << "========================================" << std::endl;

    std::string default_model = "../../model/qwen2.5-3b-instruct-q4_k_m.gguf";
    if (!std::filesystem::exists(default_model)) {
        default_model = "C:/_PROYECTOS_2030/amibot/server/model/qwen2.5-3b-instruct-q4_k_m.gguf";
    }

    std::string model_path = (argc > 1) ? argv[1] : default_model;
    std::string user_prompt = (argc > 2) ? argv[2] : "Hola! Presentate brevemente en una sola frase.";

    std::cout << "[1] Modelo objetivo: " << model_path << std::endl;
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[-] Error: El archivo del modelo no existe en la ruta indicada." << std::endl;
        return 1;
    }

    llm_comms::Config cfg;
    cfg.model_path = model_path;
    cfg.n_ctx = 2048;
    cfg.n_gpu_layers = 99;
    cfg.n_threads = 4;
    cfg.verbose = false;

    std::cout << "[2] Cargando modelo en memoria y GPU..." << std::endl;
    llm_comms::LLMClient client;
    if (!client.init(cfg)) {
        std::cerr << "[-] Error al inicializar llm_comms: " << client.get_last_error() << std::endl;
        return 1;
    }
    std::cout << "[+] Modelo cargado con exito!" << std::endl;

    llm_comms::GenerateOptions opts;
    opts.max_tokens = 128;
    opts.temperature = 0.7f;
    opts.top_p = 0.9f;

    std::cout << "\n[3] Enviando texto al LLM:" << std::endl;
    std::cout << "Prompt: \"" << user_prompt << "\"\n" << std::endl;
    std::cout << "--- Respuesta del LLM (Streaming) ---" << std::endl;

    opts.stream_cb = [](const std::string& token) -> bool {
        std::cout << token << std::flush;
        return true;
    };

    if (model_path.find("Instruct") != std::string::npos || model_path.find("instruct") != std::string::npos) {
        std::vector<llm_comms::ChatMessage> messages = {
            { "system", "Eres un asistente directo y conciso. Responde unicamente con texto limpio en el idioma del usuario, sin emojis, sin markdown y sin caracteres especiales." },
            { "user", user_prompt }
        };
        client.send_chat(messages, opts);
    } else {
        client.send_text(user_prompt, opts);
    }

    std::cout << "\n-------------------------------------" << std::endl;
    std::cout << "[+] Comunicacion completada con exito." << std::endl;

    client.close();
    return 0;
}
