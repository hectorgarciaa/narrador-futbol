# commentaries

Generacion de comentarios sinteticos de futbol a partir de eventos ya detectados o simulados.

## Objetivo

Tomar un evento estructurado en JSON y convertirlo en un comentario corto de narrador usando un modelo local servido por Ollama y convertirlo despues a audio con una voz clonada usando XTTS.

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

## Campos opcionales utiles

- `team_name`: equipo del jugador. Obligatorio en `gol`.
- `team_in_favor`: obligatorio en `corner`, `fuera de banda` y `saque de puerta`.
- `field_zone`: zona del campo donde ocurre la accion.
- `action_target`: destinatario o objetivo del gesto tecnico.
- `play_context`: contexto corto de la jugada.
- `match_score`: marcador si algun dia quieres que el LLM lo tenga en cuenta.
- `intensity`: pista de tono para una narracion mas agresiva o calmada.
- `action_index`: indice de la accion dentro de la secuencia. Si es multiplo de `30`, el comentario puede mencionar el minuto.

## Uso rapido

Servidor Ollama:

```bash
ollama serve
```

Modelo:

```bash
ollama pull tinyllama
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
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","field_zone":"frontal del area","action_index":30}'
```

Generar comentario y audio con la voz clonada:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","field_zone":"frontal del area","action_index":30}' \
  --speaker-wav "football_ai/commentaries/mi_Voz.wav" \
  --audio-out output/commentaries/audio/demo.wav
```

## Comportamiento del prompt

El prompt esta pensado para:

- sonar a retransmision futbolera;
- generar la frase completa sin prefijo fijo;
- sonar mas original y menos plantilla;
- mencionar el minuto solo una de cada 30 acciones, usando `action_index`;
- dejar clarisimo el equipo del jugador cuando la accion es `gol`;
- integrar la zona del campo si existe;
- obligar a mencionar el equipo a favor en `corner`, `fuera de banda` y `saque de puerta`;
- evitar alucinaciones sobre marcador, gol, parada o resultado si esos datos no estan en la entrada.

## Notas

- El modelo por defecto del modulo es `tinyllama:1.1b`.
- Si existe `OLLAMA_HOST` en el entorno, se usa como base URL automaticamente.
- La sintesis de voz usa por defecto `tts_models/multilingual/multi-dataset/xtts_v2`.
- Si existen varios `.wav` en `football_ai/commentaries/`, el modulo los usa todos por defecto como referencias de voz y prioriza `mi_Voz.wav` si está presente.
- La salida de audio se guarda por defecto en `output/commentaries/audio/`.
