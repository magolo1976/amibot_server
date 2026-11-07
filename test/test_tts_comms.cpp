#include "tts_comms.h"
#include <iostream>
#include <chrono>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::string text = (argc >= 2) ? argv[1] : "Hola, este es un mensaje de prueba generado con tts comms.";
    std::string out_wav = (argc >= 3) ? argv[2] : "tts_test_out.wav";
    std::string voice = (argc >= 4) ? argv[3] : "ef_dora";

    tts_comms::TTSClient client;
    tts_comms::Config cfg;
    cfg.default_voice = voice;

    if (!client.init(cfg)) {
        std::cerr << "[Error] " << client.get_last_error() << "\n";
        return 1;
    }

    auto voices = client.list_voices();
    std::cout << "[tts_comms] Voces cargadas: " << voices.size() << " (usando '" << voice << "')\n";

    tts_comms::SynthesizeOptions opts;
    opts.voice = voice;
    opts.speed = 1.0f;

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!client.synthesize_to_wav(text, out_wav, opts)) {
        std::cerr << "[Error] " << client.get_last_error() << "\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[tts_comms] Audio sintetizado en " << ms << " ms -> " << out_wav << "\n";
    return 0;
}
