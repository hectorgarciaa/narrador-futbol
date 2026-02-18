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
├── data/                   # Datos de entrada (no versionados, ver data/README.md)
│   ├── detection/          # Datasets de detección (formato YOLOv11)
│   └── partidoPrueba/      # Vídeos de partido para desarrollo
│
├── models/                 # Pesos de modelos (no versionados, ver models/README.md)
│   ├── yolo/               # Modelos base YOLOv8 y YOLOv11
│   ├── finetuning/         # Modelo fine-tuned de jugadores
│   └── finetuning-balon/   # Modelo fine-tuned de balón
│
└── experiments/            # Notebooks de análisis y visualización
    ├── detection/
    ├── tracking/
    └── visualization/
```

---

## 🛠️ Tecnologías

| Área | Tecnología |
|---|---|
| Detección | YOLOv8 / YOLOv11 (Ultralytics), fine-tuning con dataset Roboflow |
| Tracking | ByteTrack (supervision), extendido con restricción de equipo |
| Identificación de equipo | KMeans (scikit-learn), espacio de color LAB (OpenCV) |
| Evaluación | NumPy, pandas, Plotly, seaborn, matplotlib |
| Configuración | YAML (`config.yaml` centralizado) |
| Datasets | Roboflow (descarga automatizada) |

---

## 📦 Instalación

### Requisitos previos
- Python 3.10+
- CUDA (opcional, para aceleración GPU)

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd narrador-futbol
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar variables de entorno
Crea un archivo `.env` en la raíz basándote en `.env.example`:
```env
ROBOFLOW_API_KEY=tu_clave_aqui
ROBOFLOW_PUBLISHABLE_KEY=tu_clave_publicable_aqui
```

### 4. Descargar modelos y datos
```bash
# Modelos YOLO base desde Ultralytics
python scripts/data/download_models.py

# Dataset de detección desde Roboflow (requiere .env)
python scripts/data/download_datasets.py
```

---

## 🚀 Uso

Todos los scripts se ejecutan desde la **raíz del proyecto**. La configuración se lee automáticamente de `config.yaml`.

### Pipeline completo de tracking
```bash
python scripts/track.py
```
Ejecuta detección + identificación de equipo + ByteTrack, genera el vídeo anotado en `output/` y muestra métricas en consola.

### Detección básica (sin fine-tuning)
```bash
python scripts/detect.py
```

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
Genera `scripts/tracks.json` con todos los experimentos para analizar con `experiments/visualization/experiments_comparator.ipynb`.

---

## ⚙️ Configuración

Toda la configuración está centralizada en `config.yaml`. Los valores más relevantes a ajustar:

```yaml
paths:
  models:
    finetuned_player: "models/finetuning/v11/yolov11m/weights/best.pt"
  data:
    video_prueba: "data/partidoPrueba/partido.mp4"

detection:
  conf_threshold: 0.01
  ball_min_conf: 0.01

tracking:
  track_thresh: 0.5           # Confianza mínima para activar un track
  track_buffer: 90            # Frames que sobrevive un track sin ser visto
  match_thresh: 0.945         # IoU mínimo para asociar detección a track
  frame_rate: 25
  minimum_consecutive_frames: 5

teams:
  Real Madrid:
    color_rgb: [255, 127, 127]
  Wolfsburgo:
    color_rgb: [224, 77, 196]
```

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