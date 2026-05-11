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
Copia el archivo de ejemplo solo si vas a descargar datasets desde Roboflow, usar ElevenLabs o apuntar a un `llama.cpp` remoto ya levantado.

```powershell
cp .env.example .env
```
Notas:
- `ROBOFLOW_*` solo hace falta para `scripts/data/download_datasets.py`.
- `ELEVENLABS_*` solo hace falta para TTS con ElevenLabs.
- `LLAMA_CPP_BASE_URL` es opcional y sirve si quieres reutilizar un `llama.cpp` remoto/supervisado sin crear `external/llama.cpp/config.yaml`.

## 3. Preparación de la Estructura y Assets
Ejecuta el script de validación y deja presentes los assets que usa el runtime real.

```powershell
# Validar estructura de carpetas
python verify_setup.py

# Descargar modelos YOLO base de referencia (opcionales para fallback/pruebas)
python scripts/data/download_models.py
```

Nota:
- `paths.models.modelo_base` apunta por defecto a `models/finetuning/yolov11m/weights/best.pt`.
- Ese checkpoint pesa ~39 MB y se puede versionar en GitHub sin Git LFS.
- El `.gitignore` ya permite trackear exactamente ese fichero si queréis dejarlo como artefacto oficial del runtime.
- `download_models.py` no deja listo por sí solo el runtime por defecto.
- Si no vais a versionar ese checkpoint, hay que copiarlo manualmente o sobrescribir `paths.models.modelo_base` antes de arrancar el tracking.

Con eso ya puedes arrancar el tracking base con:

```powershell
python scripts/track.py
```

## 3.1 Assets de runtime

### PathCRF (`external/pathcrf`)

Con la configuración actual es obligatorio, porque `tracking.actions.enabled: true` y el pipeline espera ese checkout para la detección de acciones.

Hoy no basta con mover un `.pt` a `models/`: el runtime también necesita el código del repo externo y `saved/<trial>/args.json`. Por eso, ahora mismo lo correcto es mantener `PathCRF` como repo clonado en `external/pathcrf`.

Ejemplo:

```powershell
git clone <ruta-o-fork-de-pathcrf> external/pathcrf
```

Después comprueba que exista el trial configurado en `config.yaml`, por ejemplo:

```text
external/pathcrf/saved/120/model/state_dict_best_acc.pt
```

### `llama.cpp` local (`external/llama.cpp/config.yaml`)

Con la configuración actual es obligatorio salvo que sobrescribas `tracking.commentary.llm_base_url` o arranques la interfaz con otro backend explícito.

Ejemplo mínimo:

```powershell
mkdir -p external/llama.cpp
@"
server:
  executable: C:/ruta/a/llama-server.exe
  host: 127.0.0.1
  port: 8081
model:
  path: C:/ruta/al/modelo.gguf
"@ | Set-Content external/llama.cpp/config.yaml
```

Luego ajusta en ese YAML:
- `server.executable`: ruta al binario `llama-server`
- `model.path`: ruta al GGUF que quieras servir

Si no quieres usar `external/llama.cpp`, puedes:
- usar `--commentary-backend ollama`, o
- exportar `LLAMA_CPP_BASE_URL=http://host:puerto`

### Assets que no conviene versionar en git normal

- `models/pnlcalib/SV_kp` y `models/pnlcalib/SV_lines`: los descarga el runtime automáticamente
- `external/pnlcalib`: el runtime lo clona automáticamente
- `models/gguf/**/*.gguf`: demasiado pesados para GitHub normal
- `external/pathcrf` completo: mejor mantenerlo como repo externo con su trial/checkpoint

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
