# tts_comms — Librería Minimalista de Comunicación Texto a Voz (TTS)

Proyecto base minimalista y autocontenido para síntesis de texto a voz (`tts_comms`), depurado para contener únicamente el motor de comunicación de síntesis acústica, segmentación de frases, limpieza de texto y catálogo de 54 voces multilingües (`voices/*.bin`).

## Estructura
- `include/tts_comms.h` & `src/tts_comms.cpp`: Librería nativa C++17 (`tts_comms::TTSClient`) y C ABI (`tts_comms_*`) para síntesis PCM a 24 kHz, codificación RIFF/WAVE, extracción de vectores de estilo de 256 dimensiones y limpieza de texto.
- `js/`: Módulos ES (`tts_comms.js`, `phonemize.js`, `splitter.js`, `voices.js`) para inferencia ONNX/WebGPU/CPU.
- `python/ttscomms/`: Módulos PyTorch (`TTSCommsModel`, `TTSCommsPipeline`, `istftnet.py`, `modules.py`).
- `voices/`: 54 vectores de estilo de voz (`.bin`), incluyendo español (`ef_dora`, `em_alex`, `em_santa`) e inglés (`af_heart`, `af_bella`, `bf_emma`, etc.).

## Compilación y Exportación Automática
Al compilar este proyecto con CMake, los artefactos finales (`tts_comms.dll`, `tts_comms.lib`, `tts_comms.h`, `tts_comms.js` y `voices/`) se exportan automáticamente a `server/libraries/tts_comms`.
