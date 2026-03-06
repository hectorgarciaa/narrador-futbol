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
Ejecuta detección + identificación de equipo + ByteTrack, genera el vídeo anotado en `output/` y muestra métricas en consola.
Si existe `paths.data.video_prueba_ajustado` en `config.yaml`, ese vídeo se usa por defecto.
Cuando `tracking.use_field_positions=true`, cada frame se calibra con `PnLCalib` y el tracker usa coordenadas 2D reales del campo para `player` y `goalkeeper`, reduciendo el efecto del paneo de cámara en el matching.
Puedes forzar un vídeo específico sin editar config con `TRACK_VIDEO_KEY` (ejemplo: `TRACK_VIDEO_KEY=video_prueba_corto`) y forzar visualización con `TRACK_SHOW_OUTPUT=0|1`.
Además, si `tracking.timing_enabled=true`, se guardan tiempos por fase para detectar cuellos de botella en `output/timing_reports/`:
- `<run_id>_timing_summary.json` con tiempos globales del script y agregados del tracker.
- `<run_id>_tracking_frame_times.csv` con tiempos por frame (si `timing_save_per_frame=true`).
El reporte por frame incluye desglose detallado de detección, identificación de equipos (`team_kmeans_s`/`team_clustering_s`), tracking (`bytetrack_*`, reasignación canónica) y transformación imagen→campo 2D (`field_homography_estimation_s`, `field_project_points_s`, y detalle interno `field_pnl_*` para inferencia kp/líneas, votación geométrica y suavizado temporal).
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
  output:
    timing_reports: "output/timing_reports"

detection:
  conf_threshold: 0.01
  ball_min_conf: 0.01

tracking:
  track_thresh: 0.5           # Confianza mínima para activar un track
  track_buffer: 90            # Frames que sobrevive un track sin ser visto
  match_thresh: 0.945         # IoU mínimo para asociar detección a track
  frame_rate: 25
  minimum_consecutive_frames: 5
  max_total_tracks: 25
  max_tracks_per_class:
    player: 22
    ball: 1
    referee: 3
  timing_enabled: true
  timing_log_every_n_frames: 25
  timing_save_per_frame: true

teams:
  Real Madrid:
    color_rgb: [255, 127, 127]
  Wolfsburgo:
    color_rgb: [224, 77, 196]
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
- `referee_recovery_max_lost_frames`
- `referee_recovery_max_distance`

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

Si ejecutas en entorno headless, usa `show_output: false` en `config.yaml`.

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
