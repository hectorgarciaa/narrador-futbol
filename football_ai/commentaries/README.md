# commentaries

Generacion de comentarios sinteticos de futbol a partir de eventos ya detectados o simulados.

## Objetivo

Tomar un evento estructurado en JSON y convertirlo en un comentario corto de narrador usando un backend LLM local y convertirlo despues a audio. El backend estable sigue siendo XTTS, pero ahora tambien puedes probar ElevenLabs y dos rutas de Qwen TTS:

- `elevenlabs`: usa una voz de tu libreria de ElevenLabs, por ejemplo una voz creada con Voice Design, mediante la API streaming de Text to Speech.
- `qwen`: flujo Python `VoiceDesign -> Base`, donde primero diseñas una voz de narrador y luego reutilizas esa identidad vocal.
- `qwen_cpp`: runtime experimental con `qwen3-tts.cpp`, speaker embedding cacheado y modelos GGUF (`q8_0` para el TTS principal + `f16` para tokenizer/vocoder).

El informe tecnico completo de todas las pruebas de Qwen, con tiempos reales, intentos de optimizacion y decision final, esta en [`football_ai/report/qwen_tts_evaluation_report.md`](../report/qwen_tts_evaluation_report.md).

## Campos minimos del evento

```json
{
  "action": "tiro",
  "player_name": "Bellingham",
  "player_position": "MC",
  "event_time_s": 132.4
}
```

`event_time_s` se interpreta desde el inicio del video, suponiendo que el clip empieza en el minuto `0`.

Para la accion especial `intro`, no hace falta pasar `player_name` ni `player_position`:

```json
{
  "action": "intro",
  "event_time_s": 0.0,
  "team_name": "Real Madrid",
  "opponent_team_name": "Wolfsburgo"
}
```

Si el `intro` se genera desde la interfaz y la respuesta del modelo parece una plantilla legacy o menciona placeholders, la interfaz lo regenera automaticamente para dejar una bienvenida util.

## Campos opcionales utiles

- `team_name`: equipo del jugador. Obligatorio en `gol`.
- `opponent_team_name`: equipo que encaja el gol. Obligatorio en `gol`.
- `team_in_favor`: obligatorio en `corner`, `fuera de banda` y `saque de puerta`.
- `field_zone`: zona del campo donde ocurre la accion.
- `action_target`: destinatario o objetivo del gesto tecnico.
- `play_context`: contexto corto de la jugada.
- `match_score`: marcador si algun dia quieres que el LLM lo tenga en cuenta.
- `intensity`: pista de tono para una narracion mas agresiva o calmada.
- `action_index`: indice de la accion dentro de la secuencia. Se conserva por compatibilidad, pero el minuto ya solo se menciona en `gol`.

## Uso rapido

Servidor Ollama:

```bash
ollama serve
```

Modelo:

```bash
ollama pull gemma4:e2b
```

Sintesis de voz:

```bash
pip install qwen-tts elevenlabs
```

Generar comentario con demo integrada:

```bash
python -m football_ai.commentaries
```

La temperatura por defecto del backend de comentarios es `0.7`. Cuando se usa el servidor HTTP de comentarios, el servicio recuerda el último texto por tipo de acción y añade una instrucción al prompt para no repetir la misma frase, verbo principal ni estructura. Además de acciones de partido e `intro`, el generador acepta `contexto` para comentarios de apoyo sin jugador obligatorio.

Generar comentario pasando un evento inline:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}'
```

Generar un comentario de apertura antes de empezar el partido:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"intro","event_time_s":0.0,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo"}' \
  --text-only
```

Evaluar solo el LLM, sin TTS, viendo prompts y salida cruda:

```bash
python -m football_ai.commentaries.eval_llm \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --show-prompts \
  --show-raw-response
```

Evaluar el Gemma cuantizado servido por `llama.cpp`, sin pasar por Ollama:

```bash
python -m football_ai.commentaries.eval_llm \
  --backend llama_cpp \
  --base-url http://127.0.0.1:8001 \
  --model gemma4-q4ks-text \
  --event-json '{"action":"pase largo","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","field_zone":"medio campo","action_index":30}' \
  --show-prompts \
  --show-raw-response
```

En este caso `llama-server` debe estar levantado de antemano en la URL indicada por `--base-url`.

Lanzar varias generaciones seguidas para comparar comportamiento:

```bash
python -m football_ai.commentaries.eval_llm \
  --event-json '{"action":"tiro","player_name":"Modric","player_position":"MC","event_time_s":245.0,"field_zone":"frontal del area","action_index":31}' \
  --repeat 5
```

En este modo `FINAL_COMMENTARY` devuelve directamente el texto del modelo ya limpiado y `TOTAL_DURATION_SEC` muestra el tiempo total del backend en segundos.

La limpieza final del comentario tambien recorta interjecciones exageradas del modelo, para evitar salidas tipo `GOOOOOOOOL` o palabras con letras estiradas de forma poco natural.

Probar Hymba-1.5B-Instruct directamente desde Hugging Face con `transformers`, sin tocar el flujo de Ollama:

```bash
.venv-hymba/bin/python -m football_ai.commentaries.eval_llm \
  --backend transformers \
  --model nvidia/Hymba-1.5B-Instruct \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --show-prompts \
  --show-raw-response
```

En ese backend:

- se usa `trust_remote_code=True`;
- se intenta cargar el modelo en `cuda` con `bfloat16`;
- por defecto se desactiva Xet y se usa una cache separada en `~/.cache/huggingface-no-xet/` para evitar errores `416 Range Not Satisfiable`.

Si quieres cambiar cache o dispositivo:

```bash
.venv-hymba/bin/python -m football_ai.commentaries.eval_llm \
  --backend transformers \
  --model nvidia/Hymba-1.5B-Instruct \
  --device cuda \
  --torch-dtype bfloat16 \
  --hf-home ~/.cache/huggingface-no-xet \
  --max-new-tokens 80
```

Generar comentario y audio con la voz disenada por Qwen3-TTS y reutilizada con el modelo Base:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --tts-backend qwen \
  --audio-out output/commentaries/audio/demo.wav
```

En la primera ejecucion, el backend Qwen:

- genera un clip de referencia con `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`;
- construye un `voice_clone_prompt` reutilizable con `Qwen/Qwen3-TTS-12Hz-0.6B-Base`;
- guarda ambos artefactos en `output/commentaries/qwen_voices/<voice_id>/`.
- en GPU intenta cargar el modelo con una ruta alineada con la demo oficial: `device_map="cuda:0"` y `flash_attention_2` cuando esta disponible.

Puedes cambiar el prompt de diseno y el texto base de referencia:

```bash
python -m football_ai.commentaries \
  --tts-backend qwen \
  --qwen-voice-design-prompt "Design a voice for a professional football commentator speaking in Spanish from Spain..." \
  --qwen-reference-text "Buenas tardes, bienvenidos a una gran noche de futbol." \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo"}'
```

Para preparar una segunda voz femenina con el mismo diseno de locutor deportivo y alternarla con la voz masculina:

```bash
python -m football_ai.commentaries \
  --tts-backend xtts \
  --alternate-voices \
  --generate-female-qwen-reference \
  --prepare-voice-only
```

Ese comando genera/cachea la referencia femenina de Qwen VoiceDesign y la deja en `output/commentaries/qwen_voices/<voice_id>/voice_design_reference.wav`. A partir de ahi, la interfaz la detecta automaticamente y XTTS elige entre `male` y `female` con aleatoriedad controlada: una misma voz puede repetir, pero nunca mas de tres audios seguidos. Por defecto la voz masculina usa la referencia masculina cacheada de Qwen VoiceDesign y la femenina usa la referencia femenina cacheada. Tambien puedes pasar una muestra propia con `--speaker-wav` o `--female-speaker-wav`.

Si quieres volver al backend estable:

```bash
python -m football_ai.commentaries --tts-backend xtts
```

## Backend `elevenlabs`

ElevenLabs queda integrado como backend TTS opcional. No sustituye XTTS por defecto: se activa con `--tts-backend elevenlabs` o, desde la interfaz, con `--commentary-tts-backend elevenlabs`.

Configura `.env` en la raiz del repo:

```env
ELEVENLABS_API_KEY=tu_api_key_real
ELEVENLABS_VOICE_ID=id_de_tu_voz_masculina
ELEVENLABS_FEMALE_VOICE_ID=id_de_tu_voz_femenina
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128
ELEVENLABS_LANGUAGE_CODE=es
```

Ejemplo puntual:

```bash
python -m football_ai.commentaries \
  --tts-backend elevenlabs \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --audio-out output/commentaries/audio/elevenlabs_demo.mp3 \
  --print-timings
```

Ejemplo desde la interfaz:

```bash
python interfaz/app.py --commentary-tts-backend elevenlabs
```

En este proyecto el texto a sintetizar ya llega completo desde el LLM. El streaming de ElevenLabs no significa que recibamos palabra a palabra desde el modelo: significa que ElevenLabs empieza a devolver bytes de audio mientras genera la locucion. El backend los va escribiendo en el MP3 de salida por chunks y el manifiesto sigue apuntando al audio final para mantener compatible el flujo actual de `live` y `deferred`.

Si `--alternate-voices` o la interfaz activan la alternancia y existe `ELEVENLABS_FEMALE_VOICE_ID`, el mismo `AlternatingVoiceSynthesizer` que ya se usaba con XTTS alterna entre la voz principal (`male`) y la voz femenina (`female`), conservando el limite de tres comentarios seguidos con la misma voz.

Si lo que buscas es la mejor solucion practica ahora mismo, la combinacion recomendada es:

- `XTTS` para el backend estable y de baja latencia;
- `Qwen VoiceDesign` solo para generar una voz de locutor de referencia y reutilizarla como `speaker_wav` de XTTS.

## Backend `qwen_cpp`

El backend `qwen_cpp` usa la libreria compartida de `qwen3-tts.cpp` y mantiene el runtime cargado en memoria dentro del proceso Python. Ademas:

- cachea el `speaker embedding` extraido de la voz de referencia;
- reutiliza por defecto el `voice_design_reference.wav` ya generado por el backend `qwen`;
- busca el repo de `qwen3-tts.cpp` en `QWEN_CPP_REPO_DIR` o en `/tmp/qwen3-tts.cpp`;
- usa por defecto los GGUF en `output/commentaries/qwen_cpp_runtime/models/`.

Ejemplo de una ejecucion puntual:

```bash
python -m football_ai.commentaries \
  --model gemma4:e2b \
  --base-url http://127.0.0.1:11435 \
  --tts-backend qwen_cpp \
  --qwen-cpp-repo-dir /tmp/qwen3-tts.cpp \
  --qwen-cpp-model-dir output/commentaries/qwen_cpp_runtime/models \
  --qwen-cpp-threads 6 \
  --event-json '{"action":"pase largo","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","field_zone":"medio campo","action_index":30}' \
  --audio-out output/commentaries/audio/qwen_cpp_pipeline.wav \
  --print-timings
```

Comportamiento actual importante de `qwen_cpp` en esta maquina:

- el backend intenta compilar `ggml` con CUDA y generar una build-wrapper propia de `qwen3-tts.cpp`;
- si esa ruta CUDA no entra, cae automaticamente a la build CPU anterior;
- la primera ejecucion puede tardar bastante porque compila `ggml-cuda`, pero despues reutiliza la libreria ya construida.
- con esta RTX 3080, `6` hilos ha dado la mejor latencia estable en las pruebas locales.

Importante sobre hardware:

- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` y `Qwen/Qwen3-TTS-12Hz-0.6B-Base` son bastante mas pesados que XTTS.
- En una GPU de `10 GB` compartida con tracking, lo normal es que necesites tener la GPU libre o lanzar Qwen con `--cpu`.
- Cuando Qwen se usa en GPU, el backend intenta cargar el modelo entero en VRAM, sin offload a CPU. Si no cabe, esa ejecucion fallara y tendras que liberar GPU o usar `--cpu`.
- `flash-attn` es opcional: una vez instalado en la misma `.venv`, Qwen lo detecta automaticamente y puede usar `flash_attention_2`, pero en esta maquina las mejores latencias han salido con FlashAttention desactivado.
- La integracion queda disponible para pruebas y comparativas, pero XTTS sigue siendo la opcion mas estable para el flujo en directo mientras el tracking tambien usa CUDA.
- El codigo experimental de Qwen ya no vive dentro de `voice.py`; ahora esta aislado en `football_ai/commentaries/experimental/qwen_voice.py`.

Medir tiempos de una ejecucion puntual:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --audio-out output/commentaries/audio/demo.wav \
  --print-timings
```

Modo caliente para procesar muchos eventos y no recargar el backend TTS en cada comentario:

```bash
python -m football_ai.commentaries --jsonl-stdin
```

Luego envias un JSON por linea y recibes una respuesta JSON por linea con `commentary`, `audio_path`, `llm_seconds`, `tts_seconds` y `total_seconds`.

En `jsonl-stdin` y `http-server`, el proceso hace warmup real antes de anunciarse como listo:

- precalienta Ollama con un evento corto para dejar el modelo cargado;
- precalienta Qwen TTS sintetizando una frase breve dentro del mismo proceso;
- usa `keep_alive=30m` en Ollama para evitar que el modelo se descargue enseguida por inactividad.

Servidor HTTP local para mandar eventos por `POST` sin recargar el backend TTS:

```bash
python -m football_ai.commentaries --http-server
```

El servidor escucha por defecto en `http://127.0.0.1:8788`.

Si quieres la configuracion de menor latencia que mejor ha salido aqui con Qwen:

```bash
python -m football_ai.commentaries \
  --http-server \
  --tts-backend qwen \
  --model gemma4:e2b \
  --base-url http://127.0.0.1:11435
```

Con esa ruta, el arranque tarda unos `14 s` por el warmup, pero despues un `pase largo` ha bajado de ~`15.8 s` en ejecucion puntual a una banda de ~`7.6-9.0 s` por peticion en servidor caliente.

Comprobar salud:

```bash
curl http://127.0.0.1:8788/health
```

Generar comentario y audio:

```bash
curl -X POST http://127.0.0.1:8788/api/commentaries \
  -H 'Content-Type: application/json' \
  -d '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}'
```

Tambien puedes pasar un objeto con `event`, `audio_out` y `text_only`:

```json
{
  "event": {
    "action": "tiro",
    "player_name": "Modric",
    "player_position": "MC",
    "event_time_s": 245.0
  },
  "audio_out": "output/commentaries/audio/mi_peticion.wav",
  "text_only": false
}
```

## Streaming HTTP

El servidor tambien expone un endpoint SSE para recibir el comentario tan pronto como lo genera el LLM y cerrar la peticion cuando el audio ya este listo:

```bash
curl -N -X POST http://127.0.0.1:8788/api/commentaries/stream \
  -H 'Content-Type: application/json' \
  -d '{"event":{"action":"pase largo","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","field_zone":"medio campo","action_index":30},"audio_out":"output/commentaries/audio/stream_demo.wav"}'
```

Los eventos SSE actuales son:

- `accepted`: la peticion ha sido aceptada;
- `commentary`: el LLM ya ha devuelto el texto;
- `tts_start`: empieza la sintesis del audio;
- `audio_chunk`: solo con backends de audio streaming como `elevenlabs`; contiene `data_base64`, `content_type` y `chunk_index` para clientes que quieran procesar audio incremental;
- `completed`: ya existe el WAV y se devuelve su ruta.

Importante: con `qwen_cpp` este stream adelanta el texto y el estado del trabajo, pero no hace streaming PCM real porque `qwen3-tts.cpp` genera primero los `speech codes` y solo despues decodifica el audio completo. Con `elevenlabs`, la peticion a la API de TTS si usa streaming de audio por chunks y el endpoint SSE reemite esos chunks en base64 mientras tambien escribe el MP3 final. La interfaz actual puede ignorar `audio_chunk` y seguir usando el fichero publicado en `completed`.

Si quieres que el servidor vaya dejando un manifiesto listo para `live` o `deferred`, puedes añadir `manifest_path`, `mode` y `metadata`:

```json
{
  "event": {
    "action": "gol",
    "player_name": "Bellingham",
    "player_position": "MC",
    "event_time_s": 132.4,
    "team_name": "Real Madrid",
    "opponent_team_name": "Wolfsburgo"
  },
  "audio_out": "output/interfaz/runs/demo/commentaries/audio/gol_0001.wav",
  "manifest_path": "output/interfaz/runs/demo/commentaries/events_manifest.jsonl",
  "mode": "live",
  "metadata": {
    "run_id": "demo",
    "source": "action-listener"
  }
}
```

Ese manifiesto se puede convertir después en una pista completa y muxear dentro del MP4 final del tracking. La interfaz usa ese flujo automáticamente al cerrar un run en modo `deferred`.
Durante ese flujo, el servidor de comentarios ignora duplicados consecutivos del mismo `(action, player_name, team_name/team_in_favor)` para no sintetizar dos veces la misma jugada seguida. El ensamblado final coloca cada WAV aceptado en su `event_time_s` real, omite de la pista sonora los audios que se solaparían con otro ya aceptado y reexporta el vídeo como `H.264/AAC` para que el MP4 resultante se reproduzca bien en navegador. Si un audio de contexto marcado como interruptible se solapa con un tiro o gol posterior, el ensamblado conserva la acción peligrosa y descarta ese contexto de la pista sonora.

## Comportamiento del prompt

El prompt esta pensado para:

- sonar a retransmision futbolera;
- ser corto y conciso salvo en `gol`, donde puede ser algo mas largo y mucho mas emocional;
- obligar a incluir literalmente la accion del evento en el comentario;
- mantener los comentarios cortos y en una sola frase;
- mencionar el minuto solo en `gol`;
- permitir una apertura breve y libre cuando la accion es `intro`;
- permitir `contexto` como comentario de apoyo, sin narrar una accion tecnica concreta, usando datos tacticos de la alineacion y clasificacion simulada recibidos en el evento;
- arrancar un tiro o gol con una interrupcion natural cuando el evento llega con `intensity=interrupcion`;
- tratar las acciones de pase como acciones del jugador que da el pase, no del que lo recibe;
- reservar la mencion del equipo contrario para `gol`;
- dejar clarisimo que en `gol` el jugador marca para un equipo y se lo hace al otro;
- integrar la zona del campo si existe;
- obligar a mencionar el equipo a favor en `corner`, `fuera de banda` y `saque de puerta`;
- evitar alucinaciones sobre marcador, gol, parada o resultado si esos datos no estan en la entrada.

## Notas

- El backend por defecto del modulo principal sigue siendo Ollama y el modelo por defecto es `gemma4:e2b`.
- Si existe `OLLAMA_HOST` en el entorno, se usa como base URL automaticamente.
- Para evaluar solo el comportamiento del LLM sin la parte de voz existe `python -m football_ai.commentaries.eval_llm`, con `--backend ollama`, `--backend llama_cpp` o `--backend transformers`.
- El backend `llama_cpp` espera un `llama-server` compatible con OpenAI API, por ejemplo en `http://127.0.0.1:8001`.
- El fallback del comentario esta desactivado temporalmente: `FINAL_COMMENTARY` devuelve el texto del modelo tras la limpieza basica.
- El backend `transformers` esta pensado para pruebas locales de Hymba en una venv separada como `.venv-hymba`, para no romper el entorno principal del proyecto.
- La sintesis de voz usa por defecto `tts_models/multilingual/multi-dataset/xtts_v2`.
- `--tts-backend elevenlabs` usa `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` y opcionalmente `ELEVENLABS_FEMALE_VOICE_ID` desde `.env` y escribe MP3 por streaming.
- Tambien puedes probar `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` junto con `Qwen/Qwen3-TTS-12Hz-0.6B-Base` usando `--tts-backend qwen`.
- Si existe una referencia masculina de Qwen VoiceDesign en `output/commentaries/qwen_voices/`, el modulo la usa como voz masculina por defecto. `mi_Voz.wav` queda solo como fallback si no hay voz Qwen cacheada. Si quieres forzar otra referencia, pasala explicitamente con `--speaker-wav`.
- `--alternate-voices` permite usar una voz masculina y una femenina con seleccion aleatoria controlada; cada entrada de manifiesto guarda `voice_label` cuando se ha usado el selector de voces.
- Los artefactos de voz de Qwen se cachean en `output/commentaries/qwen_voices/` para no rediseñar ni reconstruir el prompt de clonacion en cada ejecucion.
- Las voces clonadas de XTTS se siguen cacheando en `output/commentaries/voices/`.
- La salida de audio se guarda por defecto en `output/commentaries/audio/`.
- Si quieres la menor latencia posible con Qwen en esta maquina, usa `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, sin FlashAttention, y manten el proceso vivo con `--jsonl-stdin` o `--http-server`.
- Si prefieres integracion por red local, usa `--http-server` y manda eventos a `POST /api/commentaries`.
