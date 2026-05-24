# SETUP

Guia corta para dejar el proyecto listo y ejecutar el tracking principal.

## 1. Python

La version objetivo del repo es `Python 3.13.7`.

Compruebalo con:

```bash
python --version
```

## 2. Entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Configuracion minima

Revisa `config.yaml` y confirma al menos estas rutas:
- `paths.models.modelo_base`
- `paths.data.video_prueba`
- `paths.data.video_prueba_medio`

El tracking principal necesita un peso YOLO valido en la ruta configurada por `paths.models.modelo_base`.

## 4. Verificacion

Puedes lanzar:

```bash
python verify_setup.py
```

Ese script revisa dependencias, configuracion y assets del runtime por defecto, incluyendo `PathCRF`, `llama.cpp` y credenciales de `ElevenLabs` cuando la configuracion actual los exige. Si no lo quieres usar, al menos comprueba manualmente que existe:

```text
config.yaml
models/finetuning/yolov11m/weights/best.pt
data/partidoPrueba/partido_medio.mp4
external/pathcrf/saved/120/model/state_dict_best_acc.pt
external/llama.cpp/config.yaml
```

## 5. Ejecucion rapida

Desde la raiz del repo:

```bash
python scripts/track.py video_prueba_medio
```

Con el `config.yaml` actual, ese comando espera:
- `actions.enabled: true` -> `external/pathcrf` listo;
- `commentary.enabled: true` -> `external/llama.cpp/config.yaml` o un `LLAMA_CPP_BASE_URL` remoto;
- si el backend TTS es `elevenlabs`, credenciales validas en `.env` o en el entorno.

Tambien puedes probar:

```bash
python scripts/detect.py video_prueba_medio modelo_base
python scripts/homography.py video_prueba_medio modelo_base
python interfaz/app.py
```

## 6. Assets opcionales

### Modelos base YOLO

```bash
python scripts/data/download_models.py
```

### Datasets de deteccion

Solo hace falta si vas a entrenar o repetir fine-tuning.

```bash
python scripts/data/download_datasets.py
```

### PathCRF y comentarios

Con la configuracion actual del repo, si ejecutas `scripts/track.py` tal cual, si son necesarias:
- `external/pathcrf` para acciones;
- `external/llama.cpp/config.yaml` o `LLAMA_CPP_BASE_URL` para comentarios.

Clonado recomendado:

```bash
git clone https://github.com/hyunsungkim-ds/pathcrf.git external/pathcrf
git clone https://github.com/ggml-org/llama.cpp.git external/llama.cpp
```

Despues, para `PathCRF`, comprueba que existe al menos:

```text
external/pathcrf/saved/120/args.json
external/pathcrf/saved/120/model/state_dict_best_acc.pt
```

Y para `llama.cpp`, prepara `external/llama.cpp/config.yaml` apuntando al binario `llama-server` y a tu modelo GGUF.

Si solo quieres tracking visual basico, desactiva `actions.enabled` y `commentary.enabled` en `config.yaml`. Tambien puedes lanzar `--no-commentary`, pero eso no desactiva `actions`.

## 7. Variables de entorno

`.env` es opcional, pero se vuelve necesario en estos casos:
- `ROBOFLOW_*`: descarga de datasets;
- `ELEVENLABS_*`: audio con ElevenLabs;
- `LLAMA_CPP_BASE_URL`: si reutilizas un `llama.cpp` remoto en vez del config local.

## 8. Limpieza

Para borrar caches, logs y artefactos regenerables:

```bash
python clean_project.py --help
```
