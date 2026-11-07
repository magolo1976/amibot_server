# stt_comms — Librería Minimalista de Comunicación Voz a Texto (STT)

Proyecto base minimalista y autocontenido para transcripción de audio a texto (`stt_comms`), depurado para contener **únicamente** el motor de inferencia acústica y el lector/remuestreador WAV a 16 kHz.

## Estructura
- `include/stt_comms.h`: Interfaz pública C++ (`stt_comms::STTClient`) y C ABI (`stt_comms_*`).
- `include/sttcomms-core.h`: Motor interno de inferencia STT.
- `src/`: Implementación del motor acústico, decodificador WAV + remuestreador 16 kHz y filtro de texto limpio (`stt_comms::clean_text`).
- `ggml/`: Motor de cálculo tensorial en CPU (x86 AVX2/FMA/F16C + OpenMP) integrado de forma estática.

## Compilación y Exportación Automática
Al compilar este proyecto con CMake, los artefactos finales (`stt_comms.dll`, `stt_comms.lib`, `stt_comms.h` en Windows o `libstt_comms.so`, `stt_comms.h` en Linux) se exportan automáticamente a `server/libraries/stt_comms`.
