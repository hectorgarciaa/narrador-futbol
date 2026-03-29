# 🚀 Guía de Inicio Rápido (Quick Start)

Sigue estos pasos para poner en marcha el proyecto desde cero.

## 1. Configuración del Entorno Python
Se recomienda usar Python 3.10+.

```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno (Windows)
.\.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 2. Configuración de Variables de Entorno
Copia el archivo de ejemplo y añade tu API Key de Roboflow si deseas descargar los datasets.

```powershell
cp .env.example .env
```
*Edita el archivo `.env` y rellena `ROBOFLOW_API_KEY`.*

## 3. Preparación de la Estructura y Assets
Ejecuta el script de validación para crear las carpetas necesarias y luego descarga los modelos base.

```powershell
# Validar estructura de carpetas
python verify_setup.py

# Descargar modelos base (YOLOv8/v11)
python scripts/data/download_models.py
```

## 4. Descarga de Datasets (Opcional - Para entrenamiento)
Si necesitas re-entrenar los modelos de detección, el proyecto usa el dataset de DFL Bundesliga en Roboflow por defecto:

```powershell
python scripts/data/download_datasets.py
```

**Dataset sintético (SoccerSynth/SpiideoSynLoc)** (altamente recomendado para pre-entrenamiento):
Al requerir aceptación de licencia, la descarga se hace manualmente:
1. Crea una cuenta en [research.spiideo.com](https://research.spiideo.com/).
2. Ve a la página del dataset *Spiideo SoccerNet SynLoc*.
3. Descarga `annotations.zip` y los zips de **FullHD Images** (`train.zip`, `val.zip`, `test.zip`).
4. Descomprímelos todos dentro de la nueva carpeta: `data/detection/SoccerSynth/SpiideoSynLoc`.

## 5. Ejecución del Pipeline de Tracking (Inferencia)
Para procesar un vídeo de prueba, realizar el tracking y generar el output anotado:

```powershell
python scripts/track.py
```
*Los resultados se guardarán en `output/pruebaTracker/`:*
- `video_anotado.mp4`: Vídeo con boxes e IDs.
- `tracks.json`: Datos de los tracks para análisis posterior.

## 6. Visualización de Resultados
Abre el notebook de comparación para analizar las métricas de los tracks generados:
1. Abre VS Code.
2. Navega a `experiments/visualization/track_evolution.ipynb`.
3. Selecciona el kernel `.venv`.
4. Ejecuta las celdas.

## 7. Comandos Útiles
- **Limpiar proyecto**: `python clean_project.py` (borra archivos temporales y logs).
- **Verificar estado**: `python verify_setup.py` (comprueba si falta algún asset crítico).
