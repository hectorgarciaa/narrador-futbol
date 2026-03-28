# 🎙️ AI Football Commentator

Sistema de inteligencia artificial para narración automática de partidos de fútbol, combinando detección de objetos, tracking multi-objeto, identificación de equipos, reconocimiento de acciones, generación de comentarios con LLM y síntesis de voz.

> **Estado actual:** Fase 1 completada (detección, tracking e identificación de equipos). Fases 2-5 pendientes.

---

## 📖 Descripción del Proyecto

El objetivo es construir un **pipeline completo de narración automática de fútbol** capaz de:

1. Detectar jugadores, árbitros y balón en cada frame del vídeo.
2. Hacer tracking multi-objeto para mantener IDs persistentes entre frames.
3. Identificar a qué equipo pertenece cada jugador por el color de camiseta.
4. Reconocer acciones del partido (pases, tiros, goles, faltas…).
5. Generar comentarios expresivos y contextualizados con un LLM.
6. Convertir los comentarios en audio para narración en tiempo real.

---

## 🏟️ Fases del Proyecto

### ✅ Fase 1: Detección, tracking e identificación de equipos
- Fine-tuning de YOLOv11 para las clases `player`, `goalkeeper`, `referee`, `ball`.
- Tracking multi-objeto con **ByteTrack** extendido con penalización por equipo.
- Proyección automática al campo 2D con **PnLCalib** para usar posiciones métricas de `player` y `goalkeeper` en el matching del tracker.
- Identificación de equipo mediante **KMeans en espacio LAB** sobre el crop de camiseta.
- Sistema de evaluación cuantitativo por track (cobertura, fragmentación, velocidad, etc.).

### 🚧 Fase 2: Detección de acciones
- Seleccionar y adaptar una red preentrenada para detectar acciones de fútbol (pase, tiro, gol, falta, tarjeta, penal…).

### 📅 Fase 3: Generación de comentarios con LLM
- Integrar información de tracking y acciones y enviarla a un LLM para generar comentarios expresivos y contextualizados.

### 📅 Fase 4: Conversión de texto a audio
- Transformar los comentarios a audio con una solución de TTS.

### 📅 Fase 5: Pipeline en tiempo real
- Integrar todas las fases en un pipeline eficiente para retransmisión en vivo.

---

## 🗂️ Estructura del Proyecto

```
narrador-futbol/
├── config.yaml             # Configuración central (rutas, hiperparámetros, equipos)
├── pyproject.toml          # Metadatos del paquete Python
├── requirements.txt        # Dependencias
├── clean_project.py        # Script de limpieza de archivos generados
├── verify_setup.py         # Script para verificar que todo esta listo
│
├── football_ai/            # Paquete principal (toda la lógica de negocio)
│   ├── core/               # Configuración, logging, serialización
│   ├── detection/          # Wrapper YOLO + cabeza DetectR8 para balón
│   ├── tracking/           # Tracker (orquestador) + ByteTrack extendido
│   ├── identification/     # ShirtDetector (KMeans LAB) + TeamDetector
│   ├── evaluation/         # Métricas por track y comparador de experimentos
│   └── visualization/      # Drawer: genera vídeo anotado
│
├── scripts/                # Scripts ejecutables de línea de comandos
│   ├── detect.py           # Detección base con YOLO sin fine-tuning
│   ├── detect_finetuned.py # Detección con modelo fine-tuned de jugadores
│   ├── detect_ball.py      # Detección de balón con DetectR8
│   ├── track.py            # Pipeline completo: tracking + evaluación + vídeo
│   ├── track_experiments.py# Grid search de hiperparámetros del tracker
│   ├── data/
│   │   ├── download_models.py    # Descarga modelos YOLO base
│   │   └── download_datasets.py  # Descarga datasets desde Roboflow
│   └── train/
│       └── finetune_player.py    # Fine-tuning de YOLO para fútbol
│
├── data/                   # Datos de entrada (ver data/README.md)
│   ├── detection/          # Datasets de detección (formato YOLOv11)
│   ├── partidoPrueba/      # Vídeos de partido para tracking/detección
│   └── partidosPosiciones/ # Clips para construir dataset de roles posicionales
│
├── models/                 # Pesos de modelos (no versionados, ver models/README.md)
│   ├── yolo/               # Modelos base YOLOv8 y YOLOv11
│   ├── finetuning/         # Modelo fine-tuned de jugadores
│   └── finetuning-balon/   # Modelo fine-tuned de balón
│
└── experiments/            # Notebooks de análisis y visualización
    ├── detection/
    ├── set_transformer.ipynb # Entrenamiento + aplicación de Set Transformer para roles
    ├── positions/          # Dataset supervisado de roles por posición (experimental)
    ├── reference_points/
    ├── tracking/
    └── visualization/
```

---

## 🛠️ Tecnologías

| Área | Tecnología |
|---|---|
| Detección | YOLOv8 / YOLOv11 (Ultralytics), fine-tuning con dataset Roboflow |
| Tracking | ByteTrack (supervision), extendido con restricción de equipo y posiciones 2D sobre el campo |
| Identificación de equipo | KMeans (scikit-learn), espacio de color LAB (OpenCV) |
| Evaluación | NumPy, pandas, Plotly, seaborn, matplotlib |
| Configuración | YAML (`config.yaml` centralizado) |
| Datasets | Roboflow (descarga automatizada) |

---

## 📦 Instalación

### Requisitos previos
- **Python 3.13.7** (entorno objetivo recomendado del repositorio)
- **Compatibilidad declarada del paquete:** `pyproject.toml` permite `>=3.8`
- **~10 GB de espacio libre** (modelos + datasets)
- **CUDA 12.4 + GPU NVIDIA** (opcional, para aceleración — CPU funciona pero es lento)

### Paso 1: Clonar el repositorio
```bash
git clone <repository-url>
cd narrador-futbol
```

### Paso 2: Crear entorno virtual limpio
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Verifica la versión activa (objetivo recomendado: 3.13.7)
python --version
```

### Paso 3: Actualizar pip e instalar PyTorch con CUDA support
```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar PyTorch 2.6.0 con soporte CUDA 12.4 (funciona en CPU también)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Si prefieres solo CPU (sin preparación para GPU futura):
# pip install torch torchvision
```

### Paso 4: Instalar dependencias del proyecto
```bash
# Instalar todos los paquetes del requirements.txt
pip install -r requirements.txt

# Instalar el paquete 'football_ai' en modo editable (importante!)
pip install -e .
```

### Paso 5: Configurar variables de entorno
```bash
# Copia .env.example a .env
cp .env.example .env

# Edita .env con tu API key de Roboflow
# ROBOFLOW_API_KEY=<tu_clave_aqui>
```

### Paso 6: Verificar la instalación
```bash
# Ejecuta el script de verificación
python verify_setup.py

# Debería mostrar 8/9 o 9/9 chequeos pasados (el .env needs API key es warning, no error)
```

### Paso 7: Descargar modelos y datos (opcional pero recomendado)
```bash
# Descargar modelos YOLO base desde Ultralytics
python scripts/data/download_models.py

# Descargar dataset de detección desde Roboflow (requiere ROBOFLOW_API_KEY válida)
python scripts/data/download_datasets.py
```

### Seleccionar intérprete en VS Code
1. **Ctrl+Shift+P** → **Python: Select Interpreter**
2. Si no aparece `.venv`, selecciona **"Enter interpreter path..."**
3. Escribe: `C:\Users\hecto\UNI\4\TFG\narrador-futbol\.venv\Scripts\python.exe`
4. Presiona Enter

---

## 🔍 Verificación de instalación

Siempre que hagas cambios en dependencias o uses el proyecto en una nueva terminal:

```bash
python verify_setup.py
```

Este script verifica:
- ✓ Python disponible (objetivo recomendado: 3.13.7)
- ✓ Venv activo
- ✓ Archivos de configuración (config.yaml, .env)
- ✓ Dependencias instaladas
- ✓ Módulos de football_ai importables
- ✓ PyTorch y estado de CUDA
- ✓ Variables de entorno configuradas

---

## 🚀 Uso

Todos los scripts se ejecutan desde la **raíz del proyecto**. La configuración se lee automáticamente de `config.yaml`.

### Pipeline completo de tracking
```bash
python scripts/track.py
```
También puedes indicar un shortcut de vídeo definido en `paths.data`:
```bash
python scripts/track.py video_prueba_ajustado
```
También puedes sobreescribir por terminal los colores de equipo y convertirlos a `LAB` de OpenCV automáticamente:
```bash
python scripts/track.py video_prueba_ajustado --team-colors "{Madrid:blanco, Wolsfburgo:verde-claro}"
```
`--team-colors` acepta:
- lenguaje natural de color (ej. `rojo`, `verde clarito`, `azul marino`, `rojo oscuro`)
- HEX (ej. `#90EE90`)
- RGB (ej. `255,255,255`)
Los nombres de equipo se mapean de forma flexible contra los equipos de `config.yaml` (mayúsculas/minúsculas, acentos y pequeñas erratas).
Si pasas un nombre que no exista en `config.yaml` (ej. `Stutgart`), no falla: ese nombre se usa como equipo nuevo para esa ejecución.
Si en `--team-colors` hay al menos un equipo nuevo, se usan exactamente los equipos indicados ahí para ese run.
Si no quieres depender de nombres de equipo predefinidos, puedes activar bootstrap automático por color:
```bash
python scripts/track.py video_prueba_ajustado --team-mode auto-bootstrap --team-bootstrap-frames 1
```
En `auto-bootstrap`, el sistema aprende clusters de color al inicio del vídeo y fija esos equipos durante todo el run (nombres neutrales como `Equipo 1`, `Equipo 2`).
En este modo, el color de cada equipo se calcula como la **mediana por cluster** (más robusta a outliers que la media).
También se exige un mínimo de muestras por cluster (por defecto 4): si aparece un cluster pequeño (<=3), se re-clusteriza sobre el cluster grande.
Para los clips de `data/partidosPosiciones/` hay shortcuts `video_test_*` (ejemplo: `video_test_1`, `video_test_29`).
Si quieres lanzar tracking en lote con autodetección de fuente de vídeo (`--video-source auto`):
```bash
python scripts/track_partidos_posiciones.py
```
En `auto`, el script intenta primero el dataset Kaggle DFL (train+test) y, si no está disponible, cae a `data/partidosPosiciones`.
Si quieres forzar cada modo:
```bash
# Solo clips de data/partidosPosiciones
python scripts/track_partidos_posiciones.py --video-source partidos-posiciones

# Dataset Kaggle DFL completo (train+test)
python scripts/track_partidos_posiciones.py --video-source kaggle-all
```
Si el dataset ya está descargado localmente:
```bash
python scripts/track_partidos_posiciones.py --video-source kaggle-all --kaggle-path /ruta/al/dataset
```
Por defecto solo procesa vídeos que **todavía no tienen** `output/tracks_json/tracker/<video_sanitizado>_tracks.json` (evita reejecutar).
Si quieres reprocesar todo:
```bash
python scripts/track_partidos_posiciones.py --force
```
Cuando el lote usa `data/partidosPosiciones` (modo `partidos-posiciones` o fallback de `auto`), el script aplica un plan fijo de colores por vídeo (`video_test_*`) y nombres de equipo consistentes por color.
Ejemplo de convención:
- `blanco` -> `Real Madrid`
- `negro` -> `Equipo Negro`
- `rojo` -> `Equipo Rojo`
- `amarillo` -> `Equipo Amarillo`
- `gris` -> `Equipo Gris`
- `azul` -> `Equipo Azul`

Si quieres introducir colores manualmente en cada vídeo:
```bash
python scripts/track_partidos_posiciones.py --prompt-team-colors
```
Si prefieres bootstrap automático por color en cada vídeo:
```bash
python scripts/track_partidos_posiciones.py --video-source kaggle-all --team-mode auto-bootstrap --team-bootstrap-frames 1
```
Para `kaggle-all`, si no pasas `--team-colors` ni `--team-mode`, el batch usa `auto-bootstrap` automáticamente.
Modo prueba (sin ejecutar):
```bash
python scripts/track_partidos_posiciones.py --dry-run
```
También puedes pasar colores de equipo para todo el lote:
```bash
python scripts/track_partidos_posiciones.py --team-colors "{Madrid:blanco, Wolsfburgo:verde-claro}"
```
Si quieres desactivar el plan fijo por vídeo:
```bash
python scripts/track_partidos_posiciones.py --no-default-color-plan --team-colors "{Madrid:blanco, Wolsfburgo:verde-claro}"
```
Paralelización opcional del lote:
```bash
# CPU o pruebas controladas
python scripts/track_partidos_posiciones.py --workers 2 --continue-on-error

# Forzar paralelo en GPU (usar con cautela por VRAM)
python scripts/track_partidos_posiciones.py --workers 2 --allow-gpu-parallel --continue-on-error
```
Ejecuta detección + identificación de equipo + ByteTrack, genera el vídeo anotado en `output/` y muestra métricas en consola.
Por defecto usa `paths.data.video_prueba_corto` (si existe) y, en caso contrario, `paths.data.video_prueba`.
El MP4 de salida se guarda en `output/pruebaTracker/` con el nombre del vídeo de entrada y sufijo `_tracking.mp4` (ejemplo: `partido_ajustado_tracking.mp4`).
El JSON de tracks se guarda en `output/tracks_json/tracker/<video_sanitizado>_tracks.json` (formato esperado por `experiments/positions`) y además en `output/tracks_json/tracker/tracks.json` como compatibilidad legacy.
También se guarda el resumen por vídeo en `output/tracks_json/tracker/<video_sanitizado>_summary.json`.
Y se actualiza automáticamente un dataset acumulado de métricas de tracking en `data/posiciones_etiquetadas/common/tracking_metrics.csv` (una fila por vídeo, con upsert por `video_source`). Ese resumen incluye también `ball_coverage`, para medir en qué fracción del clip el balón quedó trackeado.
Cuando `tracking.use_field_positions=true`, cada frame se calibra con `PnLCalib` y el tracker usa coordenadas 2D reales del campo para `player` y `goalkeeper`, reduciendo el efecto del paneo de cámara en el matching.
Si `tracking.reserve_penalty_spot_seed_players=true`, el tracker reserva además dos IDs canónicos sintéticos como `player` en los puntos de penalti. No participan en el clustering de equipos y solo sirven para que una detección real posterior pueda heredar esos IDs por geometría. Mientras no se absorban, también se escriben en el JSON con `synthetic_seed=true`.
Si `tracking.special_seed_role_team_assignment_enabled=true`, el tracking principal ejecuta además el modelo de `position_role` frame a frame durante el tracking para los jugadores normales y usa a los defensas detectados en ese frame para asignar equipo a los IDs reservados `1-2` por defensa más cercano. Esos dos IDs no entran al Set Transformer: se etiquetan manualmente como `POR`. Además, se congela un `role` estable por ID usando sus primeras observaciones visibles (`tracking.role_stabilization_*`) y la asignación estable final se resuelve por equipo con Hungarian para que no queden dos jugadores con la misma posición estable. Esos IDs no usan color de camiseta para recuperar identidad ni para fijar su equipo.
Para `player/goalkeeper` con homografía disponible, la reasignación canónica final usa exactamente el mismo gate de distancia en campo que ByteTrack (`field_position_match_distance_*`), sin suelo extra ni expansión por velocidad en la capa 2. Así un ID final no puede reaparecer con un salto mayor que el permitido en la capa base.
Si hay coordenadas de campo disponibles, el vídeo anotado muestra bajo cada `player` su posición `pos(m): x, y`.
Si en un frame `PnLCalib` falla (por ejemplo, homografía singular), el pipeline no aborta: ese frame se procesa con `field_position_m` no disponible y el tracking continúa.
En Linux headless, si `visualization.show_output=true` pero no hay `DISPLAY`/`WAYLAND_DISPLAY`, el sistema desactiva automáticamente la ventana de preview y continúa guardando el video de salida.
Si otra persona ya tiene este repositorio clonado, le basta con hacer `git pull`; no tiene que clonar `PnLCalib` manualmente. En la primera ejecución, el código clona `PnLCalib` en `models/reference_points/pnlcalib_repo/` y descarga sus pesos automáticamente. Si no tiene este repositorio principal en local, entonces sí tiene que clonar `narrador-futbol` una vez antes de hacer `git pull` en el futuro.

### Detección básica (sin fine-tuning)
```bash
python scripts/detect.py
```

## 🧪 Experimentos de geometría del campo

Para empezar a proyectar jugadores a coordenadas del campo hay un notebook de investigación en:

```bash
jupyter lab experiments/reference_points/reference_points.ipynb
```

Ese experimento implementa un enfoque clásico de visión por computador:
- segmentación del césped por color;
- extracción de líneas blancas con brillo, baja saturación y filtros para eliminar blobs compactos de jugadores;
- detección de segmentos con Hough y clustering por orientación;
- construcción de rectángulos candidatos a partir de pares de líneas del campo;
- estimación de homografía `imagen -> campo` y estabilización temporal frame a frame.

La longitud y anchura del campo están parametrizadas en el notebook para poder ajustar la plantilla a cada fuente de vídeo.

También hay una alternativa basada en modelo de keypoints del campo:

```bash
jupyter lab experiments/reference_points/soccana_keypoints.ipynb
```

Ese notebook descarga el modelo `Adit-jain/Soccana_Keypoint` desde Hugging Face, detecta 29 keypoints semánticos del campo y estima la homografía con RANSAC.

Y hay un tercer experimento basado en `PnLCalib`:

```bash
jupyter lab experiments/reference_points/pnlcalib_reference_points.ipynb
```

Ese notebook:
- clona `PnLCalib` bajo `models/reference_points/pnlcalib_repo/` si no existe;
- descarga los pesos `SV_kp` y `SV_lines` desde GitHub Releases;
- detecta keypoints y líneas del campo con los modelos originales del repositorio;
- estima homografía `imagen -> campo`, hace warp a bird-eye y proyecta tracks al campo 2D reescalando sus coordenadas a la resolución real del frame usado por la homografía.

Igual que en `scripts/track.py`, no hace falta clonar `PnLCalib` a mano en otra máquina: el notebook lo descarga automáticamente la primera vez.

Importante: `PnLCalib` está publicado con licencia `GPL-2.0`, así que si este método se fuese a integrar en un producto cerrado habría que revisar esa implicación legal antes.

## 🧪 Dataset de roles posicionales (experimental)

El repositorio incluye un flujo experimental para crear un dataset supervisado de rol nominal de jugador (por ejemplo, `POR`, `DFC_IZQ`, `MC`, `DC`) a partir de:
- clips en `data/partidosPosiciones/`
- tracking con homografía (`field_position_m`) en `output/tracks_json/tracker/`
- etiquetado manual periódico por JSON cada 6 segundos (`frame_id + team_id + player_id`)

Entrada principal:
- Notebook: `experiments/positions/position_role_dataset.ipynb`
- Utilidades: `experiments/positions/position_dataset.py`
- Entrenamiento/inferencia sobre dataset común: `experiments/set_transformer.ipynb`
- Pipeline reusable: `experiments/positions/set_transformer_pipeline.py`

Salida del notebook (por ejecución):
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/base_table.csv`
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/samples_metadata_and_obj_features.csv`
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/samples_teammates.npz`
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/dataset_meta.json`

Salida adicional acumulada (dataset común):
- `data/posiciones_etiquetadas/common/base_table.csv`
- `data/posiciones_etiquetadas/common/samples_metadata_and_obj_features.csv`
- `data/posiciones_etiquetadas/common/samples_teammates.npz`
- `data/posiciones_etiquetadas/common/dataset_meta.json`
- `data/posiciones_etiquetadas/common/sources.jsonl`
- `data/posiciones_etiquetadas/labels/<match_id>_labels_every_6s.json`

Ejecución:
```bash
jupyter lab experiments/positions/position_role_dataset.ipynb
```

## 🤖 Modelo Set Transformer para roles posicionales

Además del notebook de etiquetado, el repositorio incluye un notebook y un pipeline para entrenar un clasificador basado en Set Transformer a partir del dataset común acumulado en `data/posiciones_etiquetadas/common/base_table.csv` y aplicarlo después a `data/partidoPrueba/partido_ajustado.mp4`.

Antes de construir las features del modelo, cada equipo se lleva a una vista táctica canónica. Si un equipo ataca hacia `-x`, la representación se rota 180° a `(1 - x, 1 - y)` para conservar la semántica posicional `IZQ/DER` además de la profundidad del campo.

Entrenamiento por CLI:

```bash
python -m experiments.positions.set_transformer_pipeline train
```

Si has cambiado la canonización/ingeniería de features y quieres ignorar el cache derivado antiguo:

```bash
python -m experiments.positions.set_transformer_pipeline train --rebuild-from-base-table
```

Aplicación sobre `partido_ajustado`:

```bash
python -m experiments.positions.set_transformer_pipeline predict \
  --model-path models/positions/set_transformer/<timestamp>/set_transformer_checkpoint.pt \
  --video-path data/partidoPrueba/partido_ajustado.mp4
```

El notebook equivalente está en `experiments/set_transformer.ipynb` y ejecuta ese mismo flujo de forma interactiva. Soporta dos modos: reutilizar un checkpoint ya entrenado o reentrenar antes de predecir. Después de la predicción puede renderizar también el MP4 anotado con `role`. Además, fuerza una recarga explícita del módulo `set_transformer_pipeline.py`, para que los cambios recientes del pipeline se apliquen aunque el kernel de Jupyter siga vivo.

En el notebook puedes además fijar un once esperado por `team_id` y resolver la etiqueta estable con una asignación global tipo Hungarian. El modelo sigue produciendo probabilidades por jugador, pero el postproceso impone el multiconjunto de roles permitido para cada equipo; si faltan jugadores detectados, algunas plazas pueden quedar vacías, y si sobran jugadores detectados, los restantes caen a la mejor posición permitida dentro de ese once esperado.
Cuando esa restricción está activa, el vídeo anotado renderiza el rol estable restringido por equipo, no la etiqueta frame a frame libre del clasificador.

Artefactos generados:
- checkpoint y métricas en `models/positions/set_transformer/<timestamp>/`
- predicciones por frame en `output/predictions/positions/partido_ajustado_<timestamp>/frame_role_predictions.csv`
- resumen estable por jugador en `output/predictions/positions/partido_ajustado_<timestamp>/player_role_summary.csv`
- tracks enriquecidos con roles predichos en `output/predictions/positions/partido_ajustado_<timestamp>/tracks_with_predicted_roles.json`
- vídeo anotado con roles en `output/predictions/positions/partido_ajustado_<timestamp>/partido_ajustado_roles_annotated.mp4`

Para renderizar el MP4 anotado a partir del JSON enriquecido:

```bash
python -m experiments.positions.set_transformer_pipeline render-video \
  --video-path data/partidoPrueba/partido_ajustado.mp4 \
  --tracks-path output/predictions/positions/partido_ajustado_<timestamp>/tracks_with_predicted_roles.json
```

Limitación actual:
- el dataset común etiquetado no contiene muestras `POR`, así que el pipeline asigna `POR` por heurística a los tracks cuya clase es `goalkeeper`.

## ⚽ Posesión de balón por equipo (experimental)

También hay un experimento incremental para estimar por frame qué equipo tiene la posesión del balón a partir del tracking ya generado en `output/tracks_json/tracker/`.

Entrada principal:
- Script: `experiments/possession/team_possession.py`
- Vídeo de prueba habitual: `data/partidoPrueba/partido_ajustado.mp4`

La heurística actual:
- toma el centro del bbox del balón;
- busca el jugador o portero más cercano usando el pie aproximado del bbox;
- confirma toque/control cuando la proximidad coincide con una caída de velocidad, un cambio de dirección, un control del mismo jugador o una recuperación clara del rival;
- añade histéresis temporal para no cambiar de equipo con una única observación rival dudosa;
- rellena lagunas cortas y elimina segmentos mínimos espurios para estabilizar la posesión mostrada;
- mantiene siempre la posesión del último equipo que tocó el balón entre toques y solo cambia de equipo si el balón muestra señal de toque real en velocidad y/o dirección.

Ejecución:

```bash
python -m experiments.possession.team_possession \
  --video-path data/partidoPrueba/partido_ajustado.mp4
```

Artefactos generados:
- `output/predictions/possession/partido_ajustado_<timestamp>/frame_possession.csv`
- `output/predictions/possession/partido_ajustado_<timestamp>/summary.json`
- `output/predictions/possession/partido_ajustado_<timestamp>/partido_ajustado_possession_annotated.mp4`

### Detección con modelo fine-tuned de jugadores
```bash
python scripts/detect_finetuned.py
```

### Detección de balón (con DetectR8)
```bash
python scripts/detect_ball.py
```

### Fine-tuning del modelo
```bash
python scripts/train/finetune_player.py
```

### Grid search de hiperparámetros del tracker
```bash
python scripts/track_experiments.py
```
Genera `output/pruebaTracker/tracks.json` con todos los experimentos para analizar con `experiments/visualization/experiments_comparator.ipynb`.

---

## ⚙️ Configuración

Toda la configuración está centralizada en `config.yaml`. Los valores más relevantes a ajustar:

```yaml
paths:
  models:
    finetuned_player: "models/finetuning/yolov11m/weights/best.pt"
  data:
    video_prueba_ajustado: "data/partidoPrueba/partido_ajustado.mp4"
    video_prueba: "data/partidoPrueba/partido.mp4"
    video_test_1: "data/partidosPosiciones/test (1).mp4"

detection:
  conf_threshold: 0.01
  ball_min_conf: 0.0035

tracking:
  track_thresh: 0.15          # Confianza mínima para activar un track
  track_buffer: 90            # Frames que sobrevive un track sin ser visto
  match_thresh: 0.945         # IoU mínimo para asociar detección a track
  frame_rate: 25
  minimum_consecutive_frames: 5
  max_total_tracks: 25
  max_tracks_per_class:
    player: 22
    ball: 1
    referee: 3
  # Anti-ID-switch por clase y movimiento
  use_field_position_as_primary_cost: false
  use_bbox_center_for_matching: false
  bbox_center_distance_weight: 0.7
  bbox_center_distance_gate_px: 120.0
  lost_time_penalty_weight: 0.12
  lost_time_penalty_max_frames: 10
  field_position_match_distance_gate_m: 1.5
  field_position_match_distance_cap_m: 6.0
  field_position_match_distance_max_lost_frames: 10
  field_position_match_distance_growth_mode: linear_decay
  field_position_match_distance_lost_exponent: 0.5
  field_position_match_distance_decay_per_frame: 0.5
  reassign_motion_growth_cap_frames: 12
  strict_person_class_separation: true
  reserve_penalty_spot_seed_players: true
  reserve_penalty_spot_seed_match_distance_m: 12.0
  special_seed_role_team_assignment_enabled: true
  special_seed_role_model_path: "models/positions/set_transformer/20260317_211507/set_transformer_checkpoint.pt"
  special_seed_canonical_ids: [1, 2]
  special_seed_defender_roles: ["CD", "CI", "LD", "LI", "DFC_DER", "DFC_IZQ", "DFC_CENT"]
  require_field_position_for_reassign: true
  max_reassign_lost_frames: null  # null/0 = sin límite temporal de reaparición
  motion_std_gate_enabled: true
  motion_std_factor: 4.0
  motion_std_min_samples: 5
  motion_std_floor: 0.5
  ball_expected_position_gate_px: 90.0
  ball_expected_position_gate_growth_per_frame: 35.0
  ball_expected_position_confidence_relax: 1.4
  ball_size_ratio_per_frame: 1.8
  ball_size_min_samples: 5
  ball_size_std_factor: 3.0
  ball_size_std_floor: 1.0
  ball_max_reassign_lost_frames: 30
  ball_high_conf_override: 0.6

teams:
  Real Madrid:
    color_lab_opencv: [255, 127, 127]
  Wolfsburgo:
    color_lab_opencv: [224, 77, 196]
```

---

## 🎯 Límites de tracking en Fase 1

Para reducir creación de IDs nuevos y mantener estabilidad en el tracking, el sistema usa límites por clase sobre la salida final `tracks`:

- `player`: 22
- `ball`: 1
- `referee`: 3

Puntos importantes:

- La validación de límites se hace sobre `Tracker.get_tracks(...)`, no sobre el conteo crudo de detecciones YOLO por frame.
- Cuando se alcanza el máximo global de IDs visibles (`max_total_tracks`), se prioriza reasignar IDs previos compatibles antes de crear IDs nuevos.
- La reasignación mantiene coherencia por clase/equipo y aplica filtros de movimiento/cercanía para evitar saltos de identidad.

Parámetros relevantes de `TRACKER_CONF` (gestionados en `football_ai/tracking/tracker.py` y `football_ai/tracking/byte_tracker.py`):

- `max_total_tracks`
- `enforce_internal_class_limits`
- `team_mismatch_penalty`
- `second_match_threshold`
- `unconfirmed_match_threshold`
- `reassign_motion_factor`
- `reassign_min_distance`
- `reassign_min_samples`
- `reassign_motion_growth_cap_frames` (solo aplica a clases sin homografía)
- `use_field_position_as_primary_cost`
- `use_bbox_center_for_matching`
- `bbox_center_distance_weight`
- `bbox_center_distance_gate_px`
- `lost_time_penalty_weight`
- `lost_time_penalty_max_frames`
- `field_position_match_distance_gate_m`
- `field_position_match_distance_cap_m`
- `field_position_match_distance_max_lost_frames`
- `field_position_match_distance_growth_mode`
- `field_position_match_distance_lost_exponent`
- `field_position_match_distance_decay_per_frame`
- `strict_person_class_separation`
- `require_field_position_for_reassign`
- `max_reassign_lost_frames`
- `max_reassign_lost_frames_by_class`
- `motion_std_gate_enabled`
- `motion_std_factor`
- `motion_std_min_samples`
- `motion_std_floor`
- `referee_recovery_max_lost_frames`
- `referee_recovery_max_distance`

Si sigues viendo cambios de ID en clips largos, ajusta en este orden:
1. Activa `strict_person_class_separation`.
2. Activa `require_field_position_for_reassign` para `player/goalkeeper`.
3. Baja `motion_std_factor` (por ejemplo: `6 -> 5 -> 4`).
4. Baja `reassign_min_distance` (píxeles, para clases sin campo) o endurece `field_position_match_distance_*` si el problema está en `player/goalkeeper`.
5. Baja `motion_std_min_samples` para que el gate estadístico actúe antes.
6. Solo si quieres cortar reapariciones tardías, fija `max_reassign_lost_frames` (>0).

---

## 📊 Métricas de evaluación

El módulo `football_ai/evaluation/` calcula automáticamente estas métricas por track:

| Métrica | Descripción |
|---|---|
| `coverage` | % de frames del vídeo en que el track fue visible |
| `fragments` | Número de interrupciones en el track |
| `mean_speed` | Velocidad media de movimiento (px/frame) |
| `team_flip_rate` | Tasa de cambios incorrectos de equipo asignado |
| `entropy` | Entropía de la distribución de equipos del track |
| `color_var` | Varianza del color de camiseta detectado a lo largo del tiempo |
| `bbox_size_cv` | Coeficiente de variación del tamaño del bounding box |

---

## 🧹 Limpieza del proyecto

```bash
python clean_project.py --all       # Limpieza completa
python clean_project.py --cache     # Solo caché de Python
python clean_project.py --output    # Solo vídeos y JSONs generados
python clean_project.py --datasets  # Solo datasets (regenerables)
python clean_project.py --models    # Solo modelos base (regenerables)
```

> ⚠️ Los modelos **fine-tuned** no se eliminan en ningún caso (requieren horas de entrenamiento).

---

## 🐛 Solución de problemas

**`FileNotFoundError: config.yaml`** — Ejecuta los scripts desde la raíz del proyecto, no desde el directorio del script.

**`Could not open video`** — Verifica que la ruta en `config.yaml → paths.data` es correcta y el archivo existe.

**`CUDA out of memory`** — Reduce el batch en `config.yaml → finetuning.batch` o añade `device='cpu'` al script.

**Imports fallando** — Asegúrate de que el entorno virtual está activado y `pip install -r requirements.txt` se ejecutó correctamente.

---

## ❓ Preguntas Frecuentes

### ¿Qué es `football_ai.egg-info/` y por qué aparece?

`football_ai.egg-info/` es una **carpeta de metadatos** generada automáticamente por `pip install -e .` (instalación en modo editable). Contiene:
- `METADATA`: Información del paquete (versión, dependencias, autor)
- `WHEEL`: Información de la distribución
- `RECORD`: Lista de archivos instalados
- `entry_points.txt`: Scripts ejecutables del paquete

**¿Necesito regenerarla?** No. Se regenera automáticamente cuando:
- Cambias `pyproject.toml` o `setup.py`
- Ejecutas `pip install -e .` de nuevo
- Cambias dependencias en `setup.py`

**Para cambios normales en código Python**, no necesitas hacer nada. El modo editable permite cambios sin reinstalar.

**Si quieres hacer una limpieza completa:**
```bash
rm -r football_ai.egg-info/      # Linux/Mac
rmdir /s football_ai.egg-info/   # Windows
pip install -e .                  # Regenera
```

---

### ¿Puedo instalar NVIDIA CUDA Toolkit 12.4 ahora?

**Sí, es recomendable.** PyTorch ya está preparado con soporte CUDA 12.4 (`torch 2.6.0+cu124`).

#### Pasos para instalar CUDA 12.4:

1. **Verifica tu GPU NVIDIA:**
   ```bash
   nvidia-smi        # Si funciona, tienes NVIDIA instalado
   ```

2. **Descarga CUDA Toolkit 12.4:**
   - Ir a: https://developer.nvidia.com/cuda-12-4-0-download-archive
   - Seleccionar OS (Windows/Linux) y arquitectura
   - Descargar e instalar

3. **Descarga cuDNN (acelerador para redes neuronales):**
   - Ir a: https://developer.nvidia.com/cudnn
   - Descargar cuDNN para CUDA 12.4
   - Seguir instrucciones de instalación del archivo README

4. **Verifica que PyTorch detecta CUDA:**
   ```bash
   python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
   ```
   Debería mostrar: `CUDA disponible: True` y `CUDA version: 12.4`

5. **Benchmark (opcional):**
   ```bash
   python -c "import torch; t = torch.randn(10000, 10000, device='cuda'); print((t @ t).sum())"
   ```
   Debería ejecutarse rápidamente (GPU) en lugar de lentamente (CPU).

**Beneficio:** Los fine-tunings y detecciones serán **10-100x más rápidos**.

---

### ¿El venv se va a romper con actualizaciones?

No. El venv es **completamente independiente** del Python global:
- Actualizaciones de Windows o sistema no afectan
- Otros proyectos pueden usar otros venvs sin conflictos
- Si algo falla, simplemente `rm -r .venv` y crea uno nuevo

---

### ¿Cómo actualizar dependencias sin romper nada?

```bash
# Ver qué versiones hay disponibles
pip index versions ultralytics

# Actualizar una dependencia específica
pip install --upgrade ultralytics

# Actualizar todas las dependencias
pip install --upgrade -r requirements.txt

# Después, regenera el lock (recomendado):
pip freeze > requirements-lock.txt
```

---

## 📋 Mantenimiento del entorno

### Limpiar caché y archivos temporales
```bash
python clean_project.py --cache
```

### Verificar integridad periódicamente
```bash
python verify_setup.py
```

### Actualizar football_ai si editaste código
```bash
# No es necesario. El modo editable permite cambios inmediatos.
# Solo si cambiaste setup.py o pyproject.toml:
pip install -e .
```

---

## ✅ Validación mínima al cambiar código

Si cambias código ejecutable (especialmente en detección/tracking), ejecuta al menos el pipeline principal afectado y verifica que arranca sin excepción inicial.

Ejemplo:

```bash
python scripts/track.py
```

Si ejecutas en entorno headless, puedes dejar `show_output: true`: la visualización en ventana se desactiva sola cuando no hay display. Si quieres forzarlo desde configuración, usa `show_output: false`.

---

## 📝 Estado del proyecto

| Fase | Estado |
|---|---|
| Fase 1: Detección, tracking e identificación | ✅ Completada |
| Fase 2: Detección de acciones | 📅 Pendiente |
| Fase 3: Generación de comentarios (LLM) | 📅 Pendiente |
| Fase 4: Síntesis de voz | 📅 Pendiente |
| Fase 5: Pipeline en tiempo real | 📅 Pendiente |

---

## 👤 Autor

Héctor García y Carlos Mantilla  
Universidad Complutense de Madrid  
Trabajo de Fin de Grado — 2026
