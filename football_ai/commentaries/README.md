# commentaries

Generacion de comentarios sinteticos de futbol a partir de eventos ya detectados o simulados.

## Objetivo

Tomar un evento estructurado en JSON y convertirlo en un comentario corto de narrador usando `llama.cpp` con Gemma 4 GGUF y convertirlo despues a audio con XTTS (`xtts_v2`).

## Solucion final (memoria)

- **LLM**: `gemma-4-E2B-it-Q4_K_S.gguf` cuantizado, servido via `llama.cpp` (`llama-server`) con alias `gemma4-q4ks-text`, contexto 512 tokens.
- **TTS local**: XTTS / `tts_models/multilingual/multi-dataset/xtts_v2`, con voces de referencia disenadas mediante Qwen VoiceDesign.
- **TTS alternativo (demo)**: ElevenLabs `eleven_multilingual_v2` via API.
- **TTS experimental**: Qwen3-TTS (VoiceDesign + Base), aislado en `experimental/qwen_voice.py`.

## Persistencia por run

Durante la ejecución, la fase de comentarios mantiene en memoria la cola, el caché de identidades y los eventos completados, pero los artefactos finales se persisten por run:

- manifiesto JSONL con los eventos aceptados;
- WAV individuales generados por TTS;
- `commentary_events.json` y, cuando aplica, `commentary_track.wav`.

En la interfaz esos archivos viven en `output/interfaz/runs/<run_id>/commentaries/`. En `scripts/track.py` sin interfaz se guardan en un subdirectorio único de la ejecución dentro de `output/actions/rolling_online/<video>/runs/<run_token>/commentary/` o dentro de `--output-root/commentary/<run_token>/` si se usa `--output-root`.

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

## Campos opcionales utiles

- `team_name`: equipo del jugador. Obligatorio en `gol`.
- `opponent_team_name`: equipo que encaja el gol. Obligatorio en `gol`.
- `team_in_favor`: obligatorio en `corner`, `fuera de banda` y `saque de puerta`.
- `field_zone`: zona del campo donde ocurre la accion.
- `action_target`: destinatario o objetivo del gesto tecnico.
- `play_context`: contexto corto de la jugada.
- `match_score`: marcador si algun dia quieres que el LLM lo tenga en cuenta.
- `intensity`: pista de tono para una narracion mas agresiva o calmada.
- `action_index`: indice de la accion dentro de la secuencia.

## Uso rapido

### Requisitos previos

Levantar `llama-server` con el modelo Gemma 4 GGUF:

```bash
llama-server \
  --model gemma-4-E2B-it-Q4_K_S.gguf \
  --alias gemma4-q4ks-text \
  --host 127.0.0.1 \
  --port 8001 \
  --ctx-size 512 \
  --no-thought
```

Generar comentario con demo integrada:

```bash
python -m football_ai.commentaries
```

La temperatura por defecto del backend de comentarios es `0.7`.

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

Si el servidor `llama.cpp` esta en otra URL:

```bash
python -m football_ai.commentaries.eval_llm \
  --base-url http://127.0.0.1:8001 \
  --model gemma4-q4ks-text \
  --event-json '{"action":"pase largo","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","field_zone":"medio campo","action_index":30}' \
  --show-prompts \
  --show-raw-response
```

Lanzar varias generaciones seguidas para comparar comportamiento:

```bash
python -m football_ai.commentaries.eval_llm \
  --event-json '{"action":"tiro","player_name":"Modric","player_position":"MC","event_time_s":245.0,"field_zone":"frontal del area","action_index":31}' \
  --repeat 5
```

En este modo `FINAL_COMMENTARY` devuelve directamente el texto del modelo ya limpiado y `TOTAL_DURATION_SEC` muestra el tiempo total del backend en segundos.

## Voz: Qwen VoiceDesign + XTTS

El enfoque mixto recomendado es:

1. Usar Qwen VoiceDesign para generar una voz de locutor deportivo.
2. Guardar esa voz como muestra de referencia.
3. Usar XTTS para sintetizar los comentarios finales clonando esa referencia.

Generar comentario y audio con la voz disenada por Qwen3-TTS:

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

Para preparar una segunda voz femenina y alternarla con la masculina:

```bash
python -m football_ai.commentaries \
  --tts-backend xtts \
  --alternate-voices \
  --generate-female-qwen-reference \
  --prepare-voice-only
```

Si quieres volver al backend estable:

```bash
python -m football_ai.commentaries --tts-backend xtts
```

## Backend ElevenLabs (demo)

Para demos de alta calidad con voces comerciales:

```bash
python -m football_ai.commentaries \
  --tts-backend elevenlabs \
  --elevenlabs-api-key TU_API_KEY \
  --elevenlabs-voice-id TU_VOICE_ID \
  --elevenlabs-model-id eleven_multilingual_v2 \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo"}'
```

## Modo caliente (stdin JSONL)

Para procesar muchos eventos sin recargar el backend TTS:

```bash
python -m football_ai.commentaries --jsonl-stdin
```

Envias un JSON por linea y recibes una respuesta JSON por linea con `commentary`, `audio_path`, `llm_seconds`, `tts_seconds` y `total_seconds`.

En `jsonl-stdin`, el proceso hace warmup real antes de anunciarse como listo:
- precalienta el LLM con un evento corto;
- precalienta el TTS sintetizando una frase breve.

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

- El backend LLM unico es `llama.cpp` con el alias `gemma4-q4ks-text`.
- Si existe `LLAMA_CPP_BASE_URL` en el entorno, se usa como base URL automaticamente.
- Para evaluar solo el comportamiento del LLM sin la parte de voz existe `python -m football_ai.commentaries.eval_llm`.
- La sintesis de voz usa por defecto `tts_models/multilingual/multi-dataset/xtts_v2`.
- `--alternate-voices` permite usar una voz masculina y una femenina con seleccion aleatoria controlada; cada entrada de manifiesto guarda `voice_label` cuando se ha usado el selector de voces.
- Los artefactos de voz de Qwen se cachean en `output/commentaries/qwen_voices/` para no redisenar ni reconstruir el prompt de clonacion en cada ejecucion.
- Las voces clonadas de XTTS se siguen cacheando en `output/commentaries/voices/`.
- La salida de audio se guarda por defecto en `output/commentaries/audio/`.
- El codigo experimental de Qwen esta aislado en `football_ai/commentaries/experimental/qwen_voice.py`.
- ElevenLabs esta disponible como alternativa externa para demos de alta calidad, usando `--tts-backend elevenlabs`.
