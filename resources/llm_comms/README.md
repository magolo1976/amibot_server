# Proyecto Fuente: `llm_comms`

Este directorio (`server/resources/llm_comms`) contiene el **proyecto fuente completo y depurado** de `llm_comms`, basado únicamente en los módulos esenciales habilitados del motor de inferencia (`ggml-cpu` x86/AVX2 + motor `llm_comms`).

Al compilarse este proyecto, empaqueta todo el motor de forma estática dentro de una única librería dinámica autocontenida (**`llm_comms.dll`** + **`llm_comms.lib`** + **`llm_comms.h`**) y la exporta automáticamente hacia **`server/libraries/llm_comms`**.

---

## 📁 Estructura de `resources/llm_comms`

```
server/resources/llm_comms/
├── CMakeLists.txt          # Configuración de compilación y exportación automática
├── cmake/
│   └── common.cmake        # Flags de compilación necesarios
├── ggml/                   # Motor de tensores (únicamente backend CPU x86 habilitado)
├── include/
│   ├── llm_comms.h         # API pública C++ y C de llm_comms
│   ├── llama.h             # Cabecera interna del motor
│   └── llmcomms-cpp.h
├── src/
│   ├── llm_comms.cpp       # Implementación de comunicación LLMClient y ABI C
│   ├── llm_comms_engine.cpp# Núcleo del motor renombrado y unificado
│   └── ...                 # Vocabulario, muestreador, caché KV y arquitecturas
└── examples/
    └── main.cpp            # Ejecutable de prueba (test_comms)
```

---

## 🔨 Compilación

Desde `server/resources/llm_comms`:

```powershell
cmake -B build -A x64
cmake --build build --config Release -j 8
```

Al finalizar la compilación, CMake copia automáticamente los artefactos resultantes a **`server/libraries/llm_comms/`**:
* `llm_comms.dll`
* `llm_comms.lib`
* `llm_comms.h`
