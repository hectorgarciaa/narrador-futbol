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

**Flujo:**
1. Resuelve los dos argumentos como `paths.data.<atajo>` / `paths.models.<atajo>` o como rutas desde raíz.
2. Ejecuta YOLO frame a frame.
3. Normaliza las clases a `player`, `referee`, `ball` y `goalkeeper`.
4. Dibuja `bbox`, confianza y la inicial de clase (`ball` solo muestra caja).
5. Guarda MP4 y JSON en `output/detect/<modelo>/<timestamp>/`.

---

### `analyze_pnlcalib_on_detections.py` — Detección + homografía frame a frame

**Objetivo:** ejecutar YOLO y PnLCalib sobre un vídeo y guardar un JSON detallado con:
- detecciones por frame,
- `field_position_m` reproyectada,
- homografía usada,
- keypoints y líneas visibles,
- top de confidencias crudas de keypoints/líneas,
- frames donde la proyección colapsa en una vertical del campo.

**CLI básica:**
```bash
python scripts/analyze_pnlcalib_on_detections.py
```

**Opciones útiles:**
- `--video-key video_prueba_medio`
- `--model-key modelo_base`
- `--conf 0.01`
- `--max-frames 50`
- `--output-dir pnlcalib_analysis`

**Salidas:**
- `output/pnlcalib_analysis/<video>_pnlcalib_detections.json`
- `output/pnlcalib_analysis/<video>_pnlcalib_detections_summary.json`

**Notas:**
- Usa los thresholds y parámetros del proyector definidos en `tracking.projector.constructor` dentro de `config.yaml`.
- Es útil para depurar clips donde PnLCalib encuentra pocos keypoints/líneas o genera homografías degeneradas. Si el pipeline activo del tracker usa rescate adaptativo de thresholds, conviene contrastar este análisis con la salida real de `reference_points.trace.attempts` y `reference_points.trace.diagnostics` para ver en qué intento se aceptó o rechazó cada frame.

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
   - Si `visualization.four_panel_enabled=true`, también guarda:
     - `output/tracks_json/tracker/<video_sanitizado>_debug_frames.json` (metadatos por frame para depuración)
4. Genera el video anotado con `Drawer.draw_tracks()` e incluye `field_position_m` bajo los `player` cuando está disponible.
   - Si `visualization.four_panel_enabled=true`, la salida pasa a mosaico 2x2 (tracking compacto, mapa de campo, detecciones YOLO descartadas y vista con continuidad).
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
**Nota IDs especiales 1-2:** si `tracking.special_seed_role_team_assignment_enabled=true`, `track.py` ejecuta el modelo de roles posicionales frame a frame durante el tracking solo para los jugadores normales y usa el defensa más cercano de ese frame para asignar equipo a esos IDs reservados. Los IDs `1-2` no entran al modelo y se fuerzan manualmente a `POR`. Si `tracking.expected_roles_by_team` está definido, ese once esperado se usa al congelar la posición estable tras la ventana de estabilización, no para forzar la plaza de cada frame. El congelado se hace con la evidencia acumulada del propio jugador y luego se resuelven las plazas esperadas por equipo con la estrategia configurada en `tracking.role_stabilization_expected_roles_assignment` (`hungarian`, `greedy` o `ratio_priority`). En `ratio_priority`, `track.py` hace una foto global al llegar al frame de corte (`role_stabilization_window_frames`), usa solo esa evidencia para intentar cubrir plazas esperadas por equipo y, si un rol dominante no es válido para la alineación esperada o una plaza válida ya quedó ocupada, transfiere esa masa a la siguiente plaza válida libre del ranking del jugador. Después exige un umbral acumulado y otro umbral final en la plaza elegida. Si aun así quedan plazas libres, resuelve los descartes restantes con una asignación óptima entre jugadores pendientes y slots disponibles y, si todavía sobra algún slot del once, lo rellena con los jugadores aún congelados sin `expected_role_slot` usando la mejor combinación restante. Si el once esperado contiene dos `MC` o dos `DC`, una vez congelados esos dos jugadores el overlay del vídeo los desdobla a `MC_IZQ/MC_DCHO` o `DC_IZQ/DC_DCHO` usando la media acumulada de distancia a cada banda hasta ese momento y respetando la orientación del ataque. Los jugadores con menos de `role_stabilization_expected_roles_min_count` observaciones no compiten por plaza esperada en la fase fuerte y primero se congelan con su rol dominante de esa foto. El histórico previo ya no se backfillea en el JSON. Si reaparecen con un `raw_tracker_id` previamente ligado a otro canónico, solo pueden reclamarlo cuando ese canónico ya no estaba realmente activo y la geometría les favorece. Para ellos no se usa color de camiseta como criterio de recuperación.
**Salidas extra de roles:** cuando esa lógica online está activa, `track.py` escribe además los CSV y PNG de roles dentro de `output/tracker/<video>_role_artifacts/`. Ahí quedan `<video>_frame_role_predictions.csv`, `<video>_player_role_summary.csv`, `<video>_greedy_role_diagnostics.csv`, `<video>_role_assignment_vs_detected_pre<frame>.png` y, si la estrategia es `ratio_priority`, un PNG paso a paso por equipo con la simulación snapshot. También deja una copia de compatibilidad en `output/tracks_json/tracker/`.
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
3. Interpola huecos internos con coordenadas de campo (`field_position_m`) y rellena los slots que nunca aparecen con una plantilla simple de formación alineada al equipo visible.
4. Asigna los slots por cercanía a una plantilla espacial base de equipo, suaviza las trayectorias con mediana móvil, Savitzky-Golay y limitación de jitter, corrige picos aislados imposibles y filtra seeds/observaciones sintéticas antes de recalcular movimiento.
5. Deja `ball_x/ball_y` vacío de forma deliberada para no introducir una señal de balón poco fiable en el parquet de PathCRF.
6. Por defecto deja `player_id` y `ball_owning_team_id` vacíos en el parquet final para no inyectar una señal de posesión heurística y ruidosa en PathCRF.
7. Exporta:
   - `football_ai/actions/pathcrf/data/narrador/tracking_processed/<video>.parquet`
   - `football_ai/actions/pathcrf/data/narrador/tracking_processed/<video>.summary.json`

**Limitación importante:** el parquet exportado para PathCRF no incluye señal de balón usable; el modelo trabaja solo con la geometría/kinemática de jugadores y nodos exteriores.

---

### `actions/run_pathcrf.py` — Conversión + inferencia + drawer PathCRF

**Objetivo:** tomar la salida ya guardada de `scripts/track.py`, convertirla a formato PathCRF, ejecutar inferencia con el repo clonado en `football_ai/actions/repo/pathcrf/` y generar una visualización PathCRF. Si el script conoce el vídeo original y el `tracks.json`, el render sale sobre el broadcast real con `bbox` reales y un mini-mapa 2D incrustado; si no, cae al modo 2D puro.

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
3. Carga el checkpoint de PathCRF y ejecuta inferencia sobre el parquet ancho.
4. Exporta:
   - `*_edge_sequence.parquet`
   - `*_events.parquet`
   - `*_macro_prev.parquet`
   - `*_macro_next.parquet`
   - `*_summary.json`
5. Si no se desactiva, genera `*_pitch_pathcrf.mp4` con el nuevo drawer. Con shortcut de vídeo o `tracks.json` asociado intenta renderizar sobre el vídeo original con `bbox` reales; con solo parquet, usa el fallback 2D.

**Nota de entorno:** los checkpoints `set_*` del repo clonado funcionan aunque falte `torch_geometric` en la `venv`; el wrapper local mete un stub mínimo porque ese import solo es imprescindible para la variante `gat`.

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
