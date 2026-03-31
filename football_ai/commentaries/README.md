# commentaries

Generacion de comentarios sinteticos de futbol a partir de eventos ya detectados o simulados.

## Objetivo

Tomar un evento estructurado en JSON y convertirlo en un comentario corto de narrador usando un backend LLM local y convertirlo despues a audio con una voz clonada usando XTTS.

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
ollama pull qwen3:1.7b
```

Sintesis de voz:

```bash
pip install coqui-tts torchaudio torchcodec
```

Generar comentario con demo integrada:

```bash
python -m football_ai.commentaries
```

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

Lanzar varias generaciones seguidas para comparar comportamiento:

```bash
python -m football_ai.commentaries.eval_llm \
  --event-json '{"action":"tiro","player_name":"Modric","player_position":"MC","event_time_s":245.0,"field_zone":"frontal del area","action_index":31}' \
  --repeat 5
```

En este modo `FINAL_COMMENTARY` devuelve directamente el texto del modelo ya limpiado y `TOTAL_DURATION_SEC` muestra el tiempo total del backend en segundos.

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

Generar comentario y audio con la voz clonada:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --speaker-wav "football_ai/commentaries/mi_Voz.wav" \
  --audio-out output/commentaries/audio/demo.wav
```

Medir tiempos de una ejecucion puntual:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --audio-out output/commentaries/audio/demo.wav \
  --print-timings
```

Modo caliente para procesar muchos eventos y no recargar XTTS en cada comentario:

```bash
python -m football_ai.commentaries --jsonl-stdin
```

Luego envias un JSON por linea y recibes una respuesta JSON por linea con `commentary`, `audio_path`, `llm_seconds`, `tts_seconds` y `total_seconds`.

Servidor HTTP local para mandar eventos por `POST` sin recargar XTTS:

```bash
python -m football_ai.commentaries --http-server
```

El servidor escucha por defecto en `http://127.0.0.1:8788`.

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

## Comportamiento del prompt

El prompt esta pensado para:

- sonar a retransmision futbolera;
- ser corto y conciso salvo en `gol`, donde puede ser algo mas largo y mucho mas emocional;
- obligar a incluir literalmente la accion del evento en el comentario;
- mantener los comentarios cortos y en una sola frase;
- mencionar el minuto solo en `gol`;
- permitir una apertura breve y libre cuando la accion es `intro`;
- tratar las acciones de pase como acciones del jugador que da el pase, no del que lo recibe;
- reservar la mencion del equipo contrario para `gol`;
- dejar clarisimo que en `gol` el jugador marca para un equipo y se lo hace al otro;
- integrar la zona del campo si existe;
- obligar a mencionar el equipo a favor en `corner`, `fuera de banda` y `saque de puerta`;
- evitar alucinaciones sobre marcador, gol, parada o resultado si esos datos no estan en la entrada.

## Notas

- El backend por defecto del modulo principal sigue siendo Ollama y el modelo por defecto es `qwen3:1.7b`.
- Si existe `OLLAMA_HOST` en el entorno, se usa como base URL automaticamente.
- Para evaluar solo el comportamiento del LLM sin la parte de voz existe `python -m football_ai.commentaries.eval_llm`, con `--backend ollama` o `--backend transformers`.
- El fallback del comentario esta desactivado temporalmente: `FINAL_COMMENTARY` devuelve el texto del modelo tras la limpieza basica.
- El backend `transformers` esta pensado para pruebas locales de Hymba en una venv separada como `.venv-hymba`, para no romper el entorno principal del proyecto.
- La sintesis de voz usa por defecto `tts_models/multilingual/multi-dataset/xtts_v2`.
- Si existen varios `.wav` en `football_ai/commentaries/`, el modulo los usa todos por defecto como referencias de voz y prioriza `mi_Voz.wav` si está presente.
- Las voces clonadas se cachean en `output/commentaries/voices/` para no recalcular la referencia en cada ejecucion.
- La salida de audio se guarda por defecto en `output/commentaries/audio/`.
- Si quieres la menor latencia posible, usa `--jsonl-stdin` para mantener XTTS cargado entre eventos.
- Si prefieres integracion por red local, usa `--http-server` y manda eventos a `POST /api/commentaries`.
