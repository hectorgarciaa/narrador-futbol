# Informe tecnico: evaluacion de Qwen TTS en `narrador-futbol`

## 1. Objetivo del spike

El objetivo de este trabajo fue intentar sustituir o complementar el backend de voz estable basado en XTTS con una ruta Qwen TTS que ofreciese:

- mejor calidad de voz;
- prosodia mas creible para narracion deportiva;
- una voz de locutor consistente reutilizable en muchas frases;
- latencia suficiente para un flujo `live` y un flujo `deferred`.

El problema practico observado desde el inicio fue muy claro:

- `XTTS` daba la mejor latencia en caliente, pero el resultado sonoro tenia menos expresividad.
- `Qwen` daba una voz mas convincente, pero con latencias sensiblemente peores.

Este informe deja por escrito todos los intentos tecnicos realizados para cerrar ese gap.

## 2. Entorno real de pruebas

Todas las mediciones de esta sesion se validaron con:

- interprete: `.venv/bin/python`
- version real usada en pruebas: `Python 3.12.3`
- version objetivo del proyecto: `Python 3.13.7`
- GPU de desarrollo: `RTX 3080 10 GB`
- LLM local para comentarios: `gemma4:e2b` via Ollama

Importante:

- los tiempos aqui documentados son tiempos reales medidos en esta maquina de desarrollo;
- no equivalen a los benchmarks oficiales de Qwen;
- cuando se indica `finish_listen_sec`, significa tiempo total hasta terminar de escuchar el audio, no solo hasta terminar de sintetizar el WAV.

## 3. Hipotesis inicial

La hipotesis de partida fue que `Qwen3-TTS` podia mejorar claramente la calidad respecto a XTTS, pero el runtime actual no estaba usando el stack optimizado con el que el paper publica sus cifras.

Por eso se intentaron, en distintas fases:

- integracion directa Python `VoiceDesign -> Base`;
- carga completa en GPU sin offload;
- instalacion de `flash-attn`;
- comparativas `1.7B` frente a `0.6B`;
- proceso persistente para no recargar el modelo en cada comentario;
- ruta `qwen3-tts.cpp` con GGUF;
- build CUDA de `qwen3-tts.cpp`;
- SSE para adelantar el texto antes de que terminase el TTS;
- ajuste de carga a la ruta oficial de la demo (`device_map="cuda:0"` + `flash_attention_2`);
- evaluacion conceptual de `vLLM-Omni` como salto siguiente.

## 4. Cronologia de intentos

### 4.1. Integracion inicial de Qwen TTS por la ruta Python

Se integro una primera version de backend `qwen` en Python con este esquema:

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` para disenar la voz;
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` o `1.7B-Base` para reutilizar esa identidad;
- cache de `voice_design_reference.wav`;
- cache de `voice_clone_prompt.pt`.

Resultado inicial relevante:

- primer pipeline completo en CPU: `TOTAL_SEC=168.347`
- de ese total:
  - `LLM_SEC=76.186`
  - `TTS_SEC=92.161`

Conclusiones:

- el flujo funcionaba;
- la calidad potencial de voz era buena;
- la latencia inicial era inviable para directo;
- quedo claro que habia que atacar el runtime y no solo el prompt.

### 4.2. Carga del modelo entero en GPU, sin offload

Se forzo despues la carga completa de Qwen en GPU para evitar errores de `meta tensors`, offload parcial y degradacion de rendimiento.

Mediciones relevantes en esa fase:

- `Qwen 1.7B Base`, sin flash:
  - `TTS_SEC=47.326`
- `Qwen 1.7B Base`, con flash:
  - `TTS_SEC=57.273`

Conclusiones:

- cargar el modelo entero en VRAM era posible;
- desaparecio la ruta defectuosa de offload;
- `flash-attn` no mejoro en esta configuracion puntual y de hecho empeoro.

### 4.3. Instalacion y prueba de FlashAttention

Se hizo la instalacion completa de `flash-attn` en la `.venv`, incluyendo:

- compilacion larga de kernels CUDA;
- verificacion de que el paquete quedaba importable desde la venv;
- reintento de benchmarks con y sin `flash_attention_2`.

Observaciones tecnicas:

- la build fue costosa pero se completo;
- el paquete quedo utilizable de forma permanente dentro de esta `.venv`;
- el runtime `qwen` paso a poder usar `flash_attention_2` cuando estaba disponible.

Resultado practico:

- en esta maquina y con esta integracion, `flash-attn` no dio la mejor latencia;
- la mejor configuracion de latencia siguio saliendo con `FlashAttention` desactivado.

### 4.4. Comparativa `1.7B` frente a `0.6B`

Se midio el coste total del pipeline de voz separando:

- carga de voz ya disenada;
- carga del modelo Base;
- sintesis del audio;
- total.

Mediciones representativas:

`1.7B`, sin flash:

- `VOICE_CACHE_LOAD_SEC=2.920`
- `BASE_MODEL_LOAD_SEC=4.316`
- `SYNTHESIS_SEC=18.553`
- `TOTAL_SEC=25.788`

`0.6B`, con flash:

- `VOICE_CACHE_LOAD_SEC=3.137`
- `BASE_MODEL_LOAD_SEC=28.071`
- `SYNTHESIS_SEC=19.766`
- `TOTAL_SEC=50.974`

`0.6B`, sin flash:

- `VOICE_CACHE_LOAD_SEC=2.972`
- `BASE_MODEL_LOAD_SEC=3.977`
- `SYNTHESIS_SEC=19.218`
- `TOTAL_SEC=26.166`

Conclusion intermedia:

- el `0.6B` reducia footprint y carga base;
- la mejora en sintesis pura no era tan grande como se esperaba;
- la configuracion `0.6B + no flash` salio como la mas razonable dentro del backend Python.

### 4.5. Benchmark limpio en caliente del backend Python

Se hizo una comparativa controlada con la misma voz ya disenada y varias frases por configuracion.

Resultados:

`1.7B no flash`

- `voice_cache_load_sec = 3.442`
- `base_model_load_sec = 4.565`
- `first_request_total_sec = 15.998`
- `warm_request_avg_sec = 18.370`

`1.7B flash`

- `voice_cache_load_sec = 3.368`
- `base_model_load_sec = 4.499`
- `first_request_total_sec = 15.860`
- `warm_request_avg_sec = 20.589`

`0.6B no flash`

- `voice_cache_load_sec = 3.447`
- `base_model_load_sec = 4.366`
- `first_request_total_sec = 13.235`
- `warm_request_avg_sec = 16.938`

`0.6B flash`

- `voice_cache_load_sec = 3.372`
- `base_model_load_sec = 4.383`
- `first_request_total_sec = 14.274`
- `warm_request_avg_sec = 19.194`

Conclusion:

- `0.6B no flash` fue el mejor punto de equilibrio dentro de la ruta Python;
- `flash-attn` volvio a salir peor;
- el cuello dominante seguia estando en la sintesis total, no solo en la carga del modelo.

### 4.6. Proceso persistente y servidor caliente con backend Python

Se implemento modo persistente para no recargar modelo y voz en cada comentario:

- `--jsonl-stdin`
- `--http-server`
- warmup real de LLM y TTS

Mediciones importantes:

- ejecucion puntual de `pase largo` con `0.6B`:
  - `LLM_SEC=0.385`
  - `TTS_SEC=15.445`
  - `TOTAL_SEC=15.830`
- proceso persistente ya caliente:
  - peticion 1: `7.610 s`
  - peticion 2: `9.045 s`

Conclusion:

- mantener el proceso vivo mejoro mucho la latencia;
- aun asi, el backend Python de Qwen seguia lejos de XTTS en caliente.

### 4.7. Ruta `qwen3-tts.cpp` inicial

Se integro un backend experimental `qwen_cpp` con:

- modelos GGUF;
- `speaker embedding` cacheado;
- runtime persistente embebido en el proceso Python;
- endpoint SSE para adelantar el texto y el estado del trabajo.

Primeros resultados relevantes:

- tres sintesis seguidas:
  - `15.282 s`
  - `21.941 s`
  - `22.536 s`
  - media: `19.919 s`
- pipeline completo `JSON -> gemma4:e2b -> audio`:
  - `LLM_SEC=4.672`
  - `TTS_SEC=24.461`
  - `TOTAL_SEC=29.133`
- stream SSE:
  - texto disponible a ~`0.4 s`
  - `tts_seconds=21.376`
  - `total_seconds=21.776`

Conclusion:

- la experiencia SSE mejoraba la percepcion porque adelantaba el texto;
- el audio seguia llegando tarde;
- el runtime estaba yendo por CPU, asi que el backend no estaba aprovechando la GPU.

### 4.8. Rebuild de `qwen3-tts.cpp` con CUDA

Se rehizo la build de `qwen3-tts.cpp` paso a paso para activar CUDA real.

Evidencia de que la ruta nueva si usaba GPU:

- `TTSTransformer backend: CUDA0`
- `AudioTokenizerDecoder backend: CUDA0`

Mediciones tras la build CUDA:

- sintesis directa:
  - `TOTAL_SEC=13.589`
- pipeline completo:
  - `LLM_SEC=4.378`
  - `TTS_SEC=12.582`
  - `TOTAL_SEC=16.959`
- dos sintesis seguidas en caliente:
  - `RUN_1_SEC=11.121`
  - `RUN_2_SEC=10.124`

Conclusion:

- el salto respecto a la ruta CPU fue real;
- aun asi, la latencia seguia siendo peor que XTTS caliente.

### 4.9. Ajuste fino de hilos en `qwen_cpp`

Se midio el efecto del numero de hilos una vez activada la ruta CUDA.

Resultados:

- `4` hilos: `AVG=9.744 s`
- `6` hilos: `AVG=9.333 s`
- `8` hilos: `AVG=10.709 s`
- `10` hilos: `AVG=9.726 s`
- `12` hilos: `AVG=10.573 s`

Resultado operativo:

- `6` hilos fue la configuracion mas rapida y estable en esta RTX 3080;
- se adopto `6` como valor por defecto para el runtime experimental.

### 4.10. Servidor persistente `qwen_cpp`

Tambien se midio la ruta `HTTP` persistente sobre `qwen_cpp`.

Resultados:

- arranque con build ya hecha:
  - `STARTUP_TO_LISTEN_SEC=13.601`
- peticiones sucesivas:
  - `11.608 s`
  - `12.033 s`
  - `12.727 s`

Observaciones:

- no recompilaba si la libreria ya existia;
- no recargaba modelos en cada request;
- el `speaker embedding` ya se reutilizaba;
- el cuello seguia siendo la generacion de codes y el decode del vocoder.

### 4.11. Comparativa directa `XTTS` frente a `Qwen`

Se compararon los backends sobre el mismo comentario corto.

Comparativa caliente `XTTS` frente a `qwen_cpp`:

`XTTS`

- `tts_hot_sec = 1.581`
- `audio_sec = 3.243`
- `finish_listen_sec = 4.824`

`qwen_cpp`

- `tts_hot_sec = 14.188`
- `audio_sec = 3.657`
- `finish_listen_sec = 17.844`

Conclusion:

- para el flujo en directo, XTTS siguio ganando con mucha claridad;
- `qwen_cpp` mejoro mucho respecto a sus primeras iteraciones, pero no cerro el gap suficiente.

### 4.12. VoiceDesign de Qwen como referencia para XTTS

Este fue el punto de inflexion mas util del trabajo.

Idea:

- usar `Qwen VoiceDesign` solo para fabricar una voz de locutor deportivo;
- pasar ese `voice_design_reference.wav` a XTTS como `speaker_wav`;
- aprovechar la latencia de XTTS sin renunciar al timbre mas radiofonico de Qwen.

Resultados:

`XTTS` clonando la voz disenada por Qwen:

- primera pasada: `3.790 s`
- en caliente:
  - `1.276 s`
  - `0.939 s`

`XTTS` con las muestras habituales:

- `0.880 s`

Conclusion:

- la mezcla `Qwen VoiceDesign -> XTTS` fue totalmente viable;
- la penalizacion de latencia en caliente fue minima;
- la ganancia cualitativa de voz fue real;
- esta ruta se convirtio en la mejor solucion practica para produccion.

### 4.13. Pruebas de lote Gemma + XTTS con voz Qwen

Se genero una tanda completa de acciones con `gemma4:e2b` y `XTTS` usando la voz de locutor disenada por Qwen.

Artefacto resumen:

- `output/commentaries/audio/gemma4_xtts_qwenvoice_batch/summary.json`

Resultados representativos:

`gol`

- `llm_sec = 0.667`
- `tts_hot_sec = 4.651`
- `audio_sec = 13.901`
- `finish_listen_sec = 19.219`

`pase largo`

- `finish_listen_sec = 5.014`

`tiro`

- `finish_listen_sec = 3.595`

`corner`

- `finish_listen_sec = 4.660`

`robo`

- `finish_listen_sec = 4.373`

Conclusion:

- la combinacion `gemma4 + XTTS + voz Qwen` era funcional y util en varios tipos de accion;
- la diferencia principal de latencia la seguia marcando la longitud del comentario, no el timbre de referencia.

### 4.14. Limitacion de comentarios de gol tipo `GOOOOOOOL`

Durante las pruebas se detecto que `gemma4:e2b` tendia a producir goles con letras excesivamente estiradas, algo que Qwen y XTTS no narraban bien.

Se hizo una mitigacion especifica:

- prompt prohibiendo interjecciones estiradas;
- limpieza final que colapsa patrones tipo `goooool`.

Esto no fue una optimizacion de Qwen en si, pero si una mejora necesaria para que cualquier backend de voz recibiera texto viable.

### 4.15. Ultimo intento: acercar la carga de Qwen a la ruta oficial

Se hizo un ultimo intento para acercar el backend Python a la demo oficial de Qwen:

- `device_map="cuda:0"`
- `flash_attention_2` cuando estaba disponible
- carga directa en VRAM

Tambien se probó `torch.compile` de forma separada sobre la parte del talker.

Resultado con el mismo gol largo de referencia:

Antes del ajuste:

- `prepare_sec = 27.770`
- `tts_sec = 21.985`
- `finish_listen_sec = 33.745`

Despues del ajuste:

- `prepare_sec = 22.777`
- `tts_sec = 18.559`
- `finish_listen_sec = 31.199`

Conclusion:

- hubo mejora real;
- la mejora no fue suficiente para poner a Qwen por delante de XTTS en directo;
- `torch.compile` no compenso en esta integracion concreta;
- mas alla de este punto, el siguiente salto realista ya apuntaba a otro serving stack.

## 5. Papel de `vLLM-Omni`

`vLLM-Omni` se estudio como la siguiente via razonable para acercarse mas a las cifras del paper de Qwen.

Razon tecnica:

- el paper de Qwen publica resultados con un entorno optimizado basado en `vLLM` interno, `torch.compile` y `CUDA Graphs`;
- nuestra integracion no estaba usando ese stack completo;
- por tanto, las diferencias de latencia frente a cifras publicadas eran esperables.

Decisiones tomadas en este spike:

- no se sustituyo el backend estable del proyecto por `vLLM-Omni`;
- no se abrio un servidor independiente de `vLLM-Omni` dentro de la ruta principal;
- si se dejo por escrito que esa seria la siguiente prueba seria si en el futuro se reabre la linea de Qwen puro para tiempo real.

Motivo para no integrarlo ya:

- implicaba introducir otro serving stack completo;
- aumentaba la complejidad operativa del proyecto;
- no tenia sentido hacerlo mientras ya existia una solucion hibrida mucho mas practica con `VoiceDesign + XTTS`.

## 6. Decision final adoptada

La decision final fue:

- mantener `XTTS` como backend estable y de produccion para el flujo `live`;
- mantener Qwen como linea experimental y de comparacion;
- reutilizar `Qwen VoiceDesign` para fabricar una voz de locutor y usar ese WAV como referencia de `XTTS`.

Justificacion tecnica:

- `XTTS` caliente gano claramente en latencia;
- `Qwen` gano en timbre y sensacion de locutor;
- el hibrido `VoiceDesign + XTTS` capturo lo mejor de ambos:
  - la voz mejorada de Qwen;
  - la rapidez operativa de XTTS.

Decision operacional recomendada:

- `live`: `XTTS` con referencia generada por `Qwen VoiceDesign`
- `deferred` o comparativas: mantener `qwen` y `qwen_cpp` como rutas opcionales

## 7. Cambios de organizacion del codigo derivados de esta decision

Para no seguir sobrecargando `football_ai/commentaries/voice.py`, la parte experimental de Qwen se ha movido a:

- `football_ai/commentaries/experimental/qwen_voice.py`

La ruta estable se mantiene en:

- `football_ai/commentaries/voice.py`

Esto deja:

- una fachada mas limpia para la ruta principal de voz;
- los backends experimentales aislados;
- una separacion mas clara entre produccion y benchmarking.

## 8. Artefactos de referencia utiles

Voz Qwen usada como base:

- `output/commentaries/qwen_voices/qwen-voice-3b3278ad3b14b9e8/voice_design_reference.wav`

Comparativas directas de Qwen:

- `output/commentaries/audio/compare_qwen/qwen_1_7b.wav`
- `output/commentaries/audio/compare_qwen/qwen_0_6b.wav`
- `output/commentaries/audio/compare_qwen_alt/qwen_1_7b_alt.wav`
- `output/commentaries/audio/compare_qwen_alt/qwen_0_6b_alt.wav`

Comparativas XTTS vs Qwen:

- `output/commentaries/audio/compare_qwen_cpp.wav`
- `output/commentaries/audio/compare_xtts.wav`
- `output/commentaries/audio/hot_compare_xtts.wav`
- `output/commentaries/audio/hot_compare_qwen_cpp_gpu_free.wav`
- `output/commentaries/audio/xtts_from_qwen_voice_design.wav`
- `output/commentaries/audio/xtts_from_qwen_voice_design_hot_1.wav`
- `output/commentaries/audio/xtts_from_qwen_voice_design_hot_2.wav`

Batch Gemma + XTTS con voz Qwen:

- `output/commentaries/audio/gemma4_xtts_qwenvoice_batch/summary.json`
- `output/commentaries/audio/gemma4_xtts_qwenvoice_batch/goal_best.json`

## 9. Conclusiones finales

Conclusion tecnica breve:

- `Qwen` fue mejor en calidad percibida de voz;
- `XTTS` siguio siendo mucho mejor en latencia;
- `flash-attn` no dio el beneficio esperado en esta maquina;
- `qwen3-tts.cpp` mejoro mucho al pasar a CUDA, pero no suficiente para ganar a XTTS caliente;
- `vLLM-Omni` quedo identificado como la unica via clara para perseguir cifras mas cercanas a las oficiales de Qwen;
- la solucion ganadora para este proyecto fue `Qwen VoiceDesign + XTTS`.

Conclusion de producto:

- no se descarta Qwen;
- pero no se usa Qwen puro como backend por defecto en produccion;
- si se reapertura esta linea en el futuro, el siguiente paso logico no es seguir afinando detalles pequeños, sino abrir un spike independiente con `vLLM-Omni`.
