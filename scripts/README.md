# scripts

Scripts ejecutables de línea de comandos que activan las distintas fases del pipeline. Cada script es un punto de entrada independiente que usa `football_ai` como librería. La mayoría leen toda su configuración de `config.yaml` a través de `get_config()`.

Todos los scripts se ejecutan desde la raíz del proyecto:

```bash
python scripts/<nombre>.py
```

---

## Scripts de detección

### `detect.py` — Detección base con YOLO

**Objetivo:** Prueba rápida de detección usando un modelo YOLO sin fine-tuning, para verificar que el modelo base funciona correctamente sobre el vídeo.

**Flujo:**
1. Lee `paths.models.yolo_v11_m` y `paths.data.video_08fd33` de `config.yaml`.
2. Instancia YOLO directamente (sin `Detector`) y llama a `model()` con `save=True`.
3. Guarda el video anotado en `output/yoloDetectionTest/` (dentro del directorio base de output configurado).

---

### `detect_finetuned.py` — Detección con modelo fine-tuned

**Objetivo:** Verificar el modelo fine-tuned de jugadores sobre un clip del partido.

**Flujo:**
1. Carga el modelo desde `paths.models.finetuned_player` en `config.yaml`.
2. Ejecuta la detección sobre el vídeo configurado.
3. Guarda el video anotado en el directorio de salida configurado.

---

### `detect_ball.py` — Detección de balón con `DetectR8`

**Objetivo:** Probar el modelo fine-tuned de balón con la cabeza de detección personalizada `DetectR8`.

**Flujo:**
1. Carga el modelo fine-tuned de balón desde `paths.models.finetuned_ball`.
2. Sustituye la última capa del modelo por una instancia de `DetectR8` (esto es necesario porque el modelo fue entrenado con `reg_max=8` en lugar del estándar 16).
3. Ejecuta la detección sobre el vídeo de prueba.

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

`auto-bootstrap` aprende dos clusters de color de camiseta en los frames iniciales y fija esos equipos para todo el vídeo (nombres neutrales, p. ej. `Equipo 1`, `Equipo 2`). El color final de cada equipo se toma como la **mediana por cluster**.
Además, exige un mínimo de muestras por cluster (`--team-bootstrap-min-cluster-samples`, por defecto 4): si aparece un cluster pequeño (<=3), se re-clusteriza sobre el cluster grande.

**Flujo:**
1. Carga toda la configuración de `config.yaml` (modelo, video, output, confianza, tracker, equipos).
2. Instancia `Tracker` y llama a `get_tracks()`.
3. Guarda los tracks en JSON con `json.dump` + `convert_to_serializable` en:
   - `output/tracks_json/tracker/<video_sanitizado>_tracks.json` (ruta principal para `experiments/positions`)
   - `output/tracks_json/tracker/tracks.json` (legacy, compatibilidad)
4. Genera el video anotado con `Drawer.draw_tracks()` e incluye `field_position_m` bajo los `player` cuando está disponible.
5. El nombre del MP4 de salida se construye con el nombre del vídeo de entrada + `_tracking.mp4`.
6. Llama a `Evaluator` para imprimir métricas en consola.

**Nota Linux/headless:** si `visualization.show_output=true` pero no hay entorno gráfico (`DISPLAY`/`WAYLAND_DISPLAY`), la ventana en tiempo real se desactiva automáticamente y el script sigue generando el MP4 de salida.
**Nota anti-ID-switch:** `track.py` aplica gate estadístico (`motion_std_*`) y reglas estrictas de reasignación desde `config.yaml`; con `require_field_position_for_reassign=true` no hay fallback a píxeles en reasignación y con `use_field_position_as_primary_cost=true` el matching base de `player/goalkeeper` se hace por campo.
**Nota jugadores fuera de plano:** si `tracking.reserve_penalty_spot_seed_players=true`, el tracker reserva dos IDs canónicos sintéticos en los puntos de penalti. Esos seeds no entran en `TeamDetector`, pero sí pueden ser heredados por una detección real compatible cuando el jugador aparece en pantalla. Hasta entonces también se escriben en el JSON como `synthetic_seed=true`.
**Nota IDs especiales 1-2:** si `tracking.special_seed_role_team_assignment_enabled=true`, `track.py` ejecuta el modelo de roles posicionales frame a frame durante el tracking solo para los jugadores normales y usa el defensa más cercano de ese frame para asignar equipo a esos IDs reservados. Los IDs `1-2` no entran al modelo y se fuerzan manualmente a `POR`. Si reaparecen con un `raw_tracker_id` previamente ligado a otro canónico, solo pueden reclamarlo cuando ese canónico ya no estaba realmente activo y la geometría les favorece. Para todos los IDs visibles, el role queda congelado usando sus primeras observaciones (`tracking.role_stabilization_*`) para evitar cambios continuos de etiqueta. Para ellos no se usa color de camiseta como criterio de recuperación.
**Nota reapariciones tardías:** para `player/goalkeeper` con homografía, la reasignación canónica usa el mismo gate de distancia en campo que ByteTrack (`field_position_match_distance_*`). `reassign_motion_growth_cap_frames` queda como techo adicional solo para clases sin homografía.
**Nota balón:** además del matching normal, `track.py` filtra el balón con restricciones específicas de trayectoria esperada y tamaño (`ball_expected_position_*`, `ball_size_*`) para rechazar detecciones que se teletransportan o cambian de escala sin plausibilidad física, intentando mantener la cobertura original del detector. El umbral mínimo de confianza del balón se toma de `detection.ball_min_conf` en `config.yaml` y puede bajarse más que el general porque el filtro posterior ya elimina candidatas imposibles. Si el balón sale por un borde de la imagen, la búsqueda posterior queda anclada a ese límite; solo tras `ball_max_reassign_lost_frames` frames sin detección se permite redetección libre por confianza máxima. En el resumen final añade también `ball_coverage`.

Es el script principal del proyecto y sirve como referencia de cómo usar el paquete `football_ai` completo.

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
