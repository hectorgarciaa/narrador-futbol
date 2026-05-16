# scripts

Scripts ejecutables de línea de comandos que activan las distintas fases del pipeline. Cada script es un punto de entrada independiente que usa `football_ai` como librería. La mayoría leen toda su configuración de `config.yaml` a través de `get_config()`.

## `analyze_shirt_kmeans_convergence.py`

**Objetivo:** comparar si `max_iter=12` basta para el clustering LAB de camisetas frente a valores más altos usando crops reales extraídos de detecciones YOLO.

Qué hace:
1. Carga el modelo `paths.models.<model-key>` y un vídeo `paths.data.<video-key>`.
2. Muestrea frames, ejecuta detección y recorta la mitad superior del bbox para `player`, `goalkeeper` y `referee`.
3. Repite el clustering con varios `max_iter` sobre los mismos píxeles.
4. Guarda un resumen JSON y un CSV por crop con métricas de convergencia y diferencias de color respecto al baseline.

Ejemplo:

```bash
.venv/bin/python scripts/analyze_shirt_kmeans_convergence.py \
  --model-key modelo_base \
  --video-key video_prueba_corto \
  --frame-stride 25 \
  --max-frames 80 \
  --max-samples 200
```

## `analyze_shirt_kmeans_isolated_memory.py`

**Objetivo:** medir la huella aislada de RAM/VRAM del batch KMeans en un subprocess limpio, sin mezclarla con la memoria ya reservada por YOLO u otros módulos del proceso principal.

Ejemplos:

```bash
.venv/bin/python scripts/analyze_shirt_kmeans_isolated_memory.py \
  --model-key modelo_base \
  --video-key video_prueba_corto \
  --max-iter 12
```

```bash
.venv/bin/python scripts/analyze_shirt_kmeans_isolated_memory.py \
  --model-key modelo_base \
  --video-key video_prueba_corto \
  --max-iter 12 \
  --gpu
```

Todos los scripts se ejecutan desde la raíz del proyecto:

```bash
python scripts/<nombre>.py
```

---

## Scripts de detección

### `detect.py` — Detección simple por CLI

**Objetivo:** ejecutar detección YOLO sobre un vídeo indicando por CLI tanto el vídeo como el modelo, usando atajos de `config.yaml` o rutas relativas a la raíz del proyecto.

**CLI:**
```bash
.venv/bin/python scripts/detect.py video_prueba yolo_v11_m
```

```bash
.venv/bin/python scripts/detect.py data/partidoPrueba/partido.mp4 models/yolo/v11/yolo11m.pt
```

También acepta `--execution-mode runtime|debug`. Por defecto usa `runtime`.

**Flujo:**
1. Resuelve los dos argumentos como `paths.data.<atajo>` / `paths.models.<atajo>` o como rutas desde raíz.
2. Ejecuta `DetectionPhase` frame a frame usando el mismo contrato de packet que el pipeline de tracking.
3. Normaliza las clases a `player`, `referee`, `ball` y `goalkeeper`.
4. Dibuja `bbox`, confianza y la inicial de clase (`ball` solo muestra caja). Si estás en `runtime`, reconstruye los overlays a partir de `clean` sin depender de `trace`.
5. Guarda MP4 y JSON en `output/detect/<modelo>/<timestamp>/`.

---

### `homography.py` — Detección + homografía frame a frame

**Objetivo:** ejecutar YOLO y PnLCalib sobre un vídeo y guardar un JSON detallado con:
- detecciones por frame,
- `field_position_m` reproyectada,
- homografía usada,
- keypoints y líneas visibles,
- top de confidencias crudas de keypoints/líneas,
- frames donde la proyección colapsa en una vertical del campo.

**CLI básica:**
```bash
.venv/bin/python scripts/homography.py video_prueba_medio modelo_base
```

También acepta:
- `--execution-mode runtime|debug` (`debug` por defecto)
- `--max-frames 50`

**Salidas:**
- `output/homography/<modelo>/<timestamp>/homography.json`
- `output/homography/<modelo>/<timestamp>/homography.mp4`

**Notas:**
- Usa `DetectionPhase -> ProjectionPhase -> FilteringPhase`, igual que el tracker, pero sin ByteTrack ni identificación.
- Toma los thresholds y parámetros del proyector desde `tracking.projector.constructor` en `config.yaml`.
- Es útil para depurar clips donde PnLCalib encuentra pocos keypoints/líneas o genera homografías degeneradas. En `debug` conserva `reference_points.trace.attempts` y `reference_points.trace.diagnostics`.

### `export_homography_frame_pairs.py` — 4 imágenes (frame bueno/malo + campo 2D)

**Objetivo:** generar imágenes PNG para inspección rápida de calibración:
- Frame original con keypoints y líneas detectadas por PnLCalib pintadas.
- Campo 2D con proyección de `player/goalkeeper`.
- Para cada caso (`good` y `bad`), se genera ese par de imágenes.
- En el campo 2D, se oscurece la región del campo no visible desde la cámara del frame.

**CLI básica:**
```bash
.venv/bin/python scripts/export_homography_frame_pairs.py video_prueba_medio modelo_base
```

**Opciones útiles:**
- `--max-frames 900`: límite de frames a escanear.
- `--frame-step 3`: procesa 1 de cada N frames para acelerar.
- `--output-dir output/homography_frames/mi_run`: carpeta de salida personalizada.

**Salidas:**
- `good_frame_overlay.png`
- `good_field_2d.png`
- `bad_frame_overlay.png`
- `bad_field_2d.png`
- `summary.json` (frame elegido y métricas de calidad/jugadores)

---

## Scripts de tracking

### `track.py` — Pipeline completo de tracking

**Objetivo:** Ejecutar el pipeline de tracking completo sobre un vídeo, guardar los tracks en JSON y generar el video anotado.

**CLI:** admite un argumento posicional opcional para elegir vídeo:
```bash
python scripts/track.py video_prueba_ajustado
```
El valor puede ser una clave de `paths.data` en `config.yaml` o una ruta de vídeo directa.
También admite `--team-colors "{Equipo:color, Otro:color}"`, con colores en lenguaje natural (ej. `rojo`, `verde clarito`, `azul marino`) o en HEX/RGB. Si incluyes equipos no definidos en `config.yaml`, se aceptan y se usan para esa ejecución.
También admite:
- `--team-mode reference|auto-bootstrap`
- `--team-bootstrap-frames N`
- `--team-bootstrap-min-samples N`
- `--team-bootstrap-min-cluster-samples N`
- `--lineup-spec /ruta/al/lineup_spec.json`

`auto-bootstrap` aprende dos clusters de color de camiseta en los frames iniciales y fija esos equipos para todo el vídeo (nombres neutrales, p. ej. `Equipo 1`, `Equipo 2`). El color final de cada equipo se toma como la **mediana por cluster**.
Además, exige un mínimo de muestras por cluster (`--team-bootstrap-min-cluster-samples`, por defecto 4): si aparece un cluster pequeño (<=3), se re-clusteriza sobre el cluster grande.
Si pasas `--lineup-spec`, `track.py` usa por defecto el mismo comportamiento que el tracking normal: toma los colores de camiseta introducidos por el usuario como referencias directas de equipo. Si quieres bootstrap automático también con `lineup_spec`, pídeselo explícitamente con `--team-mode auto-bootstrap`. En ambos casos, el spec sobreescribe `expected_roles_by_team` con la formación elegida en la interfaz y, cuando un slot queda estable, intenta resolver `player_name` con el spec.

**Flujo:**
1. Carga toda la configuración de `config.yaml` (modelo, video, output, confianza, tracker, equipos).
2. Instancia `Tracker` y llama a `get_tracks()`.
   - Nada más entrar, normaliza las clases YOLO a las clases internas del proyecto.
   - Solo deja pasar detecciones de `player`, `goalkeeper`, `referee`, `ball` y alias comunes.
   - Si el modelo base devuelve `person`, esa detección se remapea automáticamente a `player`.
   - Si el modelo base devuelve `sports ball`, esa detección se remapea automáticamente a `ball`.
   - Cualquier otra clase YOLO se descarta antes de `supervision`, PnLCalib y ByteTrack.
   - Si `PnLCalib` devuelve una proyección válida para una detección y esa posición cae fuera del campo, la detección se descarta antes de identificación de equipos, ByteTrack y canonización, salvo dos excepciones: se permite un margen de 1 metro solo en las bandas laterales para conservar linieres y, además, cualquier caja fuera del campo se mantiene si solapa con un track activo de ByteTrack en ese frame.
3. Guarda los tracks en JSON con `json.dump` + `convert_to_serializable` en:
   - `output/tracks_json/tracker/<video_sanitizado>_tracks.json` (ruta principal para `experiments/positions`)
   - `output/tracks_json/tracker/tracks.json` (legacy, compatibilidad)
   - Si `tracking.execution_mode=debug`, también guarda:
     - `output/tracks_json/tracker/<video_sanitizado>_debug_frames.json` (metadatos por frame para depuración)
4. Genera el video anotado con `Drawer.draw_tracks()` e incluye `field_position_m` bajo los `player` cuando está disponible.
   - La salida se renderiza siempre como mosaico 2x2 (tracking compacto, mapa de campo, detecciones YOLO descartadas y vista con continuidad). En `runtime`, el panel de descartes queda en negro.
   - En ese mosaico, los paneles de tracking/campo/continuidad colorean `player/gk` por cluster de equipo, y el panel de descartes separa YOLO no trackeadas por ByteTrack de detecciones sí trackeadas pero descartadas al entrar en la capa canónica.
   - Si `tracking.possession.enabled=true`, resalta al poseedor con un segundo recuadro amarillo y muestra `POS: <equipo>` en overlays (modo 1 panel y 4 paneles).
5. El nombre del MP4 de salida se construye con el nombre del vídeo de entrada + `_tracking.mp4`.
6. Llama a `Evaluator` para imprimir métricas en consola.

**Análisis offline de descartes (four-panel):**
```bash
python scripts/analyze_debug_frames.py output/tracks_json/tracker/<video_sanitizado>_debug_frames.json
```

Cuando la instrumentación de depuración está activa, `discarded_yolo_not_tracked` también puede incluir `bytetrack_reason` y `bytetrack_stage`, útiles para saber si la detección cayó por umbral de activación, supresión de nuevos candidatos por solape o falta de matching. Además, cada frame puede guardar `frame_num` y `bytetrack_unconfirmed_association`, con el diagnóstico detallado del matching de tracks tentativos (`unconfirmed`): mejor candidato, IoU, penalizaciones de clase/tamaño/campo, conflicto de asignación y outcome final. En el packet `reference_points`, la traza conserva `attempts`, `selected_attempt_index`, `rejection_type` y `rejection_reasons`; si la homografía del frame no es usable, el pipeline ya no usa `field_position_m` y cae a bbox.

**Auditoría conjunta de homografía + ByteTrack + canónico:**
```bash
.venv/bin/python scripts/audit_tracking_run.py video_yt_30s --device cpu
```

Genera:
- `output/tracks_json/tracker/audits/<video>_audit_frames.json`
- `output/tracks_json/tracker/audits/<video>_audit_frames.csv`
- `output/tracks_json/tracker/audits/<video>_audit_summary.json`

**Cuello de botella de ByteTrack (`raw_detections` descartadas):**
```bash
.venv/bin/python scripts/analyze_bytetrack_bottleneck.py video_yt_30s
```

Genera:
- `output/tracks_json/tracker/audits/<video>_bytetrack_bottleneck_summary.json`
- `output/tracks_json/tracker/audits/<video>_bytetrack_bottleneck_frames.json`
- `output/tracks_json/tracker/audits/<video>_bytetrack_bottleneck_high_conf_examples.json`

**Matching de `unconfirmed` (por qué no llegan a confirmarse):**
```bash
.venv/bin/python scripts/analyze_unconfirmed_matching.py video_yt_30s_200f
```

Genera:
- `output/tracks_json/tracker/audits/<video>_unconfirmed_matching_summary.json`
- `output/tracks_json/tracker/audits/<video>_unconfirmed_matching_examples.json`

**Nota Linux/headless:** si `visualization.show_output=true` pero no hay entorno gráfico (`DISPLAY`/`WAYLAND_DISPLAY`), la ventana en tiempo real se desactiva automáticamente y el script sigue generando el MP4 de salida.
**Nota anti-ID-switch:** `track.py` aplica gate estadístico (`motion_std_*`) y reglas estrictas de reasignación desde `config.yaml`; con `require_field_position_for_reassign=true` no hay fallback a píxeles en reasignación y con `use_field_position_as_primary_cost=true` el matching base de `player/goalkeeper` se hace por campo.
**Nota jugadores fuera de plano:** si `tracking.reserve_penalty_spot_seed_players=true`, el tracker reserva dos IDs canónicos sintéticos en los puntos de penalti. Esos seeds no entran en `TeamDetector`, pero sí pueden ser heredados por una detección real compatible cuando el jugador aparece en pantalla. Hasta entonces también se escriben en el JSON como `synthetic_seed=true`.
**Nota IDs especiales 1-2:** si `tracking.special_seed_role_team_assignment_enabled=true`, `track.py` ejecuta el modelo de roles posicionales frame a frame durante el tracking solo para los jugadores normales y usa el defensa más cercano de ese frame para asignar equipo a esos IDs reservados. Los IDs `1-2` no entran al modelo y se fuerzan manualmente a `POR`. Si `tracking.expected_roles_by_team` está definido, el flujo online usa una doble pasada de Húngaro: primero genera un voto limpio por frame contra los `tracking_slots` esperados del equipo, y después resuelve el slot definitivo del frame usando la mayoría acumulada de cada segmento activo. Si el tracking detecta un relink fuerte o un salto estructural, abre un segmento nuevo para ese `canonical_id`.
**Salidas extra de roles:** cuando esa lógica online está activa, `track.py` escribe además los CSV y PNG de roles dentro de `output/tracker/<video>_role_artifacts/`. Ahí quedan `<video>_frame_role_predictions.csv`, `<video>_player_role_summary.csv` y `<video>_greedy_role_diagnostics.csv` como exporte diagnóstico del estado por segmento. También deja una copia de compatibilidad en `output/tracks_json/tracker/`.
Si la ejecución viene de la interfaz, `track.py` copia además el `lineup_spec.json` usado dentro de ese mismo directorio de artefactos.
**Nota reapariciones tardías:** para `player/goalkeeper` con homografía, la reasignación canónica usa exactamente el mismo gate de distancia en campo que ByteTrack (`field_position_match_distance_*`), sin suelo extra ni expansión por velocidad en la capa 2. `reassign_motion_growth_cap_frames` queda solo para clases sin homografía.
**Nota balón:** además del matching normal, `track.py` filtra el balón con restricciones específicas de trayectoria esperada y tamaño (configuradas en `tracking.ball.*`) para rechazar detecciones que se teletransportan o cambian de escala sin plausibilidad física, intentando mantener la cobertura original del detector. El umbral mínimo de confianza del balón se toma de `detection.ball_min_conf` en `config.yaml` y puede bajarse más que el general porque el filtro posterior ya elimina candidatas imposibles. Si el balón sale por un borde de la imagen, la búsqueda posterior queda anclada a ese límite; solo tras `tracking.ball.max_reassign_lost_frames` frames sin detección se permite redetección libre por confianza máxima. En el resumen final añade también `ball_coverage`.

Es el script principal del proyecto y sirve como referencia de cómo usar el paquete `football_ai` completo.

---

## Scripts de acciones

### `actions/convert_tracks_to_pathcrf.py` — Adaptador `tracks.json` → PathCRF

**Objetivo:** transformar la salida JSON del tracker actual en un parquet ancho compatible con el formato que espera `PathCRF`.

**CLI:**
```bash
python scripts/actions/convert_tracks_to_pathcrf.py output/tracks_json/tracker/partido_corto_tracks.json
```

**Opciones útiles:**
- `--output-path /ruta/salida.parquet`: permite guardar el parquet en una ruta concreta.
- `--fps 25`: controla los `timestamp`, velocidades y aceleraciones derivadas.

**Flujo:**
1. Lee el `tracks.json` generado por `scripts/track.py`.
2. Fusiona `player` y `goalkeeper` en 22 slots fijos (`home_1..11`, `away_1..11`) y conserva 3 árbitros (`referee_1..3`).
3. Mapea los slots directamente desde los IDs canónicos reservados: `1 -> home_1`, `2 -> away_1`, `3..12 -> home_2..11`, `13..22 -> away_2..11`, `23..25 -> referee_1..3`.
4. Interpola huecos internos con coordenadas de campo (`field_position_m`), rellena los slots que nunca aparecen con una plantilla simple de formación y suaviza las trayectorias con mediana móvil, Savitzky-Golay y limitación de jitter antes de recalcular movimiento.
5. Deja `ball_x/ball_y` vacío de forma deliberada para no introducir una señal de balón poco fiable en el parquet de PathCRF.
6. Por defecto deja `player_id` y `ball_owning_team_id` vacíos en el parquet final para no inyectar una señal de posesión heurística y ruidosa en PathCRF.
7. Exporta:
   - `football_ai/actions/pathcrf/data/narrador/tracking_processed/<video>.parquet`
   - `football_ai/actions/pathcrf/data/narrador/tracking_processed/<video>.summary.json`

**Limitación importante:** el parquet exportado para PathCRF no incluye señal de balón usable; el modelo trabaja solo con la geometría/kinemática de jugadores y nodos exteriores.

---

### `actions/run_pathcrf.py` — Conversión + inferencia + drawer PathCRF

**Objetivo:** pipeline **offline legacy**. Toma la salida ya guardada de `scripts/track.py`, convierte una sola vez a formato PathCRF y ejecuta una sola inferencia batch sobre toda la secuencia.

**CLI:**
```bash
python scripts/actions/run_pathcrf.py video_prueba_corto
```

También acepta una ruta directa al JSON:
```bash
python scripts/actions/run_pathcrf.py output/tracks_json/tracker/partido_corto_tracks.json
```

**Opciones útiles:**
- `--output-dir output/actions/pathcrf/<nombre>`: carpeta donde dejar todos los artefactos.
- `--trial 120`: selecciona el checkpoint de PathCRF.
- `--model-file state_dict_best_acc.pt`: checkpoint concreto dentro del trial.
- `--device auto|cpu|cuda:0`: dispositivo de inferencia.
- `--no-crf`: fuerza decodificación sin CRF.
- `--decode indep|greedy|viterbi`: modo de decodificación si `--no-crf`.
- `--no-render`: omite el MP4 del campo y deja solo parquet/json.
- `--render-width` / `--render-height`: tamaño del render de fallback 2D o del inset cuando hay broadcast real.

**Flujo:**
1. Resuelve la entrada: shortcut de `config.yaml`, vídeo, `tracks.json` o parquet ya convertido.
2. Si la entrada es `tracks.json`, la convierte a `*_tracking.parquet`.
3. Carga el checkpoint de PathCRF y ejecuta una inferencia offline sobre el parquet ancho.
4. Dentro de esa inferencia, PathCRF recorre internamente su ventana temporal deslizante para producir `edge` por frame.
5. Exporta:
   - `*_edge_sequence.parquet`
   - `*_events.parquet`
   - `*_macro_prev.parquet`
   - `*_macro_next.parquet`
   - `*_summary.json`
6. Si no se desactiva, genera `*_pitch_pathcrf.mp4` con el nuevo drawer. Con shortcut de vídeo o `tracks.json` asociado intenta renderizar sobre el vídeo original con `bbox` reales; con solo parquet, usa el fallback 2D.

**Nota de entorno:** los checkpoints `set_*` del repo clonado funcionan aunque falte `torch_geometric` en la `venv`; el wrapper local mete un stub mínimo porque ese import solo es imprescindible para la variante `gat`.

---

### `actions/run_live_snapshots_pathcrf.py` — Pipeline snapshot aislado

**Objetivo:** pipeline **snapshot/live bridge aislado**. Reprocesa `tracks` N veces: en cada snapshot acumulado ejecuta PathCRF legacy completo.

**CLI:**
```bash
python scripts/actions/run_live_snapshots_pathcrf.py video_prueba
```

**Semántica:**
1. Avanza frame a frame sobre `tracks.json`.
2. Solo cuando cumple warmup/cadencia (`--min-frames`, `--snapshot-interval-frames`) genera snapshot acumulado `0..frame_n`.
3. En cada snapshot corre `run_pathcrf_pipeline` legacy.
4. Guarda artefactos por snapshot y un resumen global `live_snapshots_summary.json`.

---

### `actions/run_incremental_pathcrf.py` — Pipeline incremental aislado

**Objetivo:** pipeline **incremental causal**. Actualiza estado cada frame y ejecuta inferencia de PathCRF cada N frames.

**CLI:**
```bash
python scripts/actions/run_incremental_pathcrf.py video_prueba
```

**Semántica:**
1. Actualiza `ActionsRuntime` en cada frame (`process_frame`).
2. Ejecuta inferencia solo cuando toca por `--min-frames-warmup` y `--cadence-frames`.
3. Mantiene postproceso causal (confirmación/cooldown).
4. Cada inferencia batch genera internamente un edge por frame del batch (`raw_edge_batch`).
5. Mantiene salida pública con un edge representativo por disparo de inferencia (`raw_edge`).
6. Exporta `tracking.parquet`, `edge_sequence.parquet`, `events_semantic.parquet`, `runtime_checkpoints.json` e `internal_edges_per_frame.parquet`.

---

### `actions/run_rolling_pathcrf.py` — Rolling con snapshots (paridad con live)

**Objetivo:** emular el bridge live con snapshots: en cada cadencia genera un snapshot acumulado `0..frame_n`, corre `run_pathcrf_pipeline` (adapter offline) y emite edges con delay.

**CLI:**
```bash
python scripts/actions/run_rolling_pathcrf.py video_prueba \
  --cadence-frames 10 \
  --emit-delay-frames 25 \
  --emit-frames 10
```

**Semántica (modo default):**
1. Mantiene un buffer en memoria de `tracks` frame a frame.
2. Cada `--cadence-frames` genera `tracks_snapshot_<frame>.json` y ejecuta PathCRF offline sobre ese snapshot.
3. Exporta:
   - `rolling_edges_per_frame.parquet`
   - `emitted_edges.parquet`
   - `runtime_checkpoints.json`
   - `summary.json`
4. Los artefactos por snapshot quedan en `output/actions/pathcrf_rolling/<video>/snapshots/`.

**Modo diagnostico (upper-bound):**
```bash
python scripts/actions/run_rolling_pathcrf.py video_prueba \
  --tracking-path output/actions/pathcrf/<video>/<video>_tracking.parquet
```
Este modo usa un `tracking.parquet` ya construido (no snapshots) y sirve solo como referencia/upper-bound.

**Debug de tracking:**
- `--tracking-diff` compara el tracking de cada snapshot contra el offline completo y guarda diffs por checkpoint en `tracking_diffs/`.

---

### `actions/evaluate_edge_similarity.py` — Similitud de edges contra offline

**Objetivo:** comparar `snapshot` e `incremental` contra `offline` usando IDs canónicos y tolerancia temporal.

**CLI:**
```bash
python scripts/actions/evaluate_edge_similarity.py \
  --offline-edge-path output/actions/pathcrf/<video>/<video>_edge_sequence.parquet \
  --snapshot-summary-path output/actions/pathcrf_live_snapshots/<video>/live_snapshots_summary.json \
  --incremental-internal-edges-path output/actions/pathcrf_incremental/<video>/internal_edges_per_frame.parquet \
  --tolerance-frames 5 \
  --output-dir output/actions/edge_eval/<video>
```

**Semántica de matching:**
1. Convierte `edge_src/edge_dst` desde slots PathCRF a IDs canónicos (`home_1 -> 1`, etc.).
2. Ignora edges que no puedan mapearse a IDs canónicos.
3. Cuenta match si existe mismo par dirigido `(src,dst)` en offline dentro de `frame ± 5`.
4. Reporta `edge_match_ratio` y también `precision/recall/F1`.

---

### `actions/run_cumulative_context_and_runtime_debug.py` — Ablation acumulativa + debug incremental

**Objetivo:** auditar paridad incremental vs snapshot/offline corrigiendo la referencia de snapshot a frame-level interno, ejecutando contexto acumulado en checkpoints incrementales y exportando debug por etapas del runtime.

**CLI:**
```bash
.venv/bin/python scripts/actions/run_cumulative_context_and_runtime_debug.py \
  --offline-dir output/actions/pipeline_audit/eval_full_offline \
  --snapshot-dir output/actions/pipeline_audit/eval_full_snapshot \
  --incremental-dir output/actions/pipeline_audit/eval_full_incremental \
  --snapshot-internal-path output/actions/pipeline_audit/eval_full_similarity/snapshot_internal_edges_per_frame.parquet \
  --output-dir output/actions/pipeline_audit/eval_full_cumulative_context_and_runtime_debug
```

**Salidas clave:**
1. `cumulative_context_summary.json` y `effective_config_audit.json`.
2. `snapshot_cumulative_at_incremental_checkpoints_edges.parquet`.
3. `incremental_stateful_explicit_config_edges.parquet` e `incremental_stateless_explicit_config_edges.parquet`.
4. `incremental_runtime_debug_by_frame.parquet` e `incremental_runtime_debug_by_batch.csv`.
5. Comparativas: `comparison_against_offline.csv`, `comparison_against_snapshot_internal.csv`, `comparison_by_stage.csv`, `comparison_by_batch_local_index.csv`.

---

### `track_partidos_posiciones.py` — Batch de tracking (partidosPosiciones o Kaggle DFL)

**Objetivo:** ejecutar `scripts/track.py` automáticamente en lote:
- shortcuts `video_test_*` de `data/partidosPosiciones` (vía `config.yaml`), o
- todos los `.mp4` del dataset Kaggle DFL (train+test).

**Uso básico:**
```bash
python scripts/track_partidos_posiciones.py
```
Por defecto usa `--video-source auto`: intenta Kaggle DFL y, si no está disponible, cae a `data/partidosPosiciones`.

**Opciones útiles:**
- `--dry-run`: muestra qué comandos ejecutaría sin lanzar el tracking.
- `--continue-on-error`: continúa con el siguiente vídeo si uno falla.
- `--team-colors "{Madrid:blanco, Wolsfburgo:verde-claro}"`: reenvía el override de colores a cada ejecución de `track.py`.
- `--team-mode auto-bootstrap`: fuerza bootstrap automático de equipos en cada vídeo.
- `--team-bootstrap-frames N`: controla cuántos frames iniciales usa ese bootstrap.
- `--team-bootstrap-min-cluster-samples N`: reenvía el mínimo de muestras por cluster para cerrar bootstrap.
- `--no-prompt-team-colors`: desactiva el modo interactivo para ejecución totalmente automática.
- `--video-source kaggle-all`: fuerza procesar el dataset Kaggle completo.
- `--kaggle-path /ruta/al/dataset`: usa un dataset local ya descargado.
- `--workers N`: paraleliza el lote con N procesos de `track.py` (recomendado `1` en GPU salvo pruebas controladas).
- `--allow-gpu-parallel`: permite `--workers > 1` aunque haya CUDA.

Ejemplos Kaggle:
```bash
# Descarga con kagglehub (si no está cacheado) y procesa train+test
python scripts/track_partidos_posiciones.py --video-source kaggle-all

# Reutiliza dataset local sin descargar
python scripts/track_partidos_posiciones.py --video-source kaggle-all --kaggle-path /ruta/dfl
```

El plan fijo de colores por vídeo (`video_test_*`) solo aplica a `data/partidosPosiciones`; para Kaggle puedes usar `--team-colors` global o `--prompt-team-colors`.
Si en Kaggle no pasas `--team-colors` ni `--team-mode`, el batch usa `auto-bootstrap` por defecto.

---

### `track_experiments.py` — Grid search de hiperparámetros

**Objetivo:** Ejecutar automáticamente múltiples experimentos variando los hiperparámetros del tracker (`conf`, `track_thresh`, `match_thresh`, `minimum_consecutive_frames`) y guardar todos los resultados para analizarlos después con `ExperimentVisualizer`.

**Flujo:**
1. Define listas de valores para cada hiperparámetro.
2. Hace un nested loop sobre todas las combinaciones.
3. Por cada combinación: crea el `Tracker`, ejecuta el tracking, añade el resultado a `all_tracks`.
4. Genera el video anotado de cada experimento.
5. Guarda el primer experimento como `tracks_sample.json` (referencia rápida).
6. Al terminar todos los experimentos, guarda todos los tracks en `tracks.json`.

Todas las rutas se leen desde `config.yaml`.

---

## Submóduló `data/`

### `data/download_models.py` — Descarga de modelos YOLO base

**Objetivo:** Descargar los modelos YOLO preentrenados de Ultralytics y guardarlos en la estructura de carpetas esperada por `config.yaml`.

**Funcionamiento:**
- Define un diccionario de modelos por versión (v8, v11) y nombre.
- Para cada modelo, instancia `YOLO(nombre)` (Ultralytics los descarga automáticamente si no están en caché) y los guarda en `models/yolo/<version>/<archivo>.pt`.

Ejecución:
```bash
python scripts/data/download_models.py
```

---

### `data/download_datasets.py` — Descarga de datasets desde Roboflow

**Objetivo:** Descargar el dataset de detección de fútbol en formato YOLOv11 desde Roboflow y colocarlo en `data/detection/`.

**Requisitos:**
- Tener un archivo `.env` en la raíz con `ROBOFLOW_API_KEY=<tu_clave>`.
- Tener `python-dotenv` y `roboflow` instalados.

Ejecución:
```bash
python scripts/data/download_datasets.py
```

---

## Submódulo `train/`

### `train/finetune_player.py` — Fine-tuning de YOLO

**Objetivo:** Entrenar un modelo YOLO preentrenado sobre el dataset de fútbol para las cuatro clases del proyecto.

**Flujo:**
1. Carga el modelo base definido en `config.yaml` (`paths.models.yolo_v11_m`).
2. Llama a `model.train()` con el dataset (`paths.data.dataset_football_ai`), épocas y batch configurables en `config.yaml` bajo `finetuning`.
3. YOLO guarda automáticamente los resultados en `models/finetuning/finetuning/weights/best.pt` (según `project` y `name` del script).

Ejecución:
```bash
python scripts/train/finetune_player.py
```

Los pesos resultantes se deben copiar manualmente como `models/finetuning/yolov11m.pt` para que los demás scripts los encuentren.

```bash
# Windows
copy models\finetuning\finetuning\weights\best.pt models\finetuning\yolov11m.pt
```

---

### `comparar_modelos.py` — Comparativa justa entre checkpoints YOLO

**Objetivo:** Evaluar dos checkpoints sobre el mismo dataset y con un espacio de clases común para evitar comparaciones engañosas cuando los modelos no usan exactamente la misma taxonomía.

**Criterio de comparación actual:**
- El espacio canónico de evaluación es de 3 clases: `ball`, `player`, `ref`.
- Si un modelo predice `goalkeeper`, esa clase se remapea a `player` antes de calcular métricas.
- Alias comunes como `referee`, `arbitro` o `sports ball` se normalizan al mismo espacio.
- Cualquier clase que no pueda mapearse de forma inequívoca se ignora explícitamente y se reporta como predicción descartada.

**Por qué es importante:**
- Evita colisiones por `class_id` entre modelos distintos.
- Hace comparable un modelo de 4 clases (`ball`, `goalkeeper`, `player`, `referee`) con otro de 3 clases (`ball`, `player`, `ref`).
- Impide que una clase extra del modelo se contabilice por accidente como otra clase del ground truth.

Ejecución:
```bash
python scripts/comparar_modelos.py
```
