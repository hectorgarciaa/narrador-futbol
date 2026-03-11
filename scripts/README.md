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
**Nota anti-ID-switch:** `track.py` aplica un gate estadístico configurable (`motion_std_*` en `config.yaml`) para bloquear reasignaciones con saltos de movimiento anómalos por frame.

Es el script principal del proyecto y sirve como referencia de cómo usar el paquete `football_ai` completo.

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
