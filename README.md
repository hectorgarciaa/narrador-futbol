# narrador-futbol

Proyecto para analizar video de futbol y construir un narrador automatico apoyado en IA.

Hoy el codigo que esta realmente operativo se centra en:
- deteccion de jugadores, arbitros y balon con YOLO;
- tracking multiobjeto con IDs persistentes;
- identificacion de equipo por color de camiseta;
- proyeccion al campo, posesion, roles posicionales y render de video.

Las partes de acciones, LLM y TTS existen en el repo, pero todavia no forman un pipeline productivo completo de narracion end-to-end.

## Estructura

- `football_ai/`: paquete principal del proyecto.
- `scripts/`: puntos de entrada por linea de comandos.
- `interfaz/`: interfaz web para lanzar ejecuciones con alineaciones.
- `data/`: videos y datasets.
- `models/`: pesos y checkpoints.
- `experiments/`: notebooks y pruebas de investigacion.
- `output/`: artefactos generados.

## Requisitos

- `Python 3.13.7` recomendado para este repo.
- Dependencias de [requirements.txt](requirements.txt).
- Un video en `data/partidoPrueba/` y pesos configurados en `config.yaml`.

## Instalacion

Orden recomendado para arrancar el proyecto por primera vez:

1. instalar dependencias Python;
2. clonar los repos externos necesarios;
3. revisar `config.yaml`;
4. ejecutar `verify_setup.py`;
5. lanzar el pipeline.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`verify_setup.py` comprueba tambien el runtime por defecto definido en `config.yaml`, incluyendo:
- modelo YOLO principal;
- video de prueba;
- `external/pathcrf` y su checkpoint si `actions.enabled: true`;
- el bloque `llama_cpp` en `config.yaml` o un endpoint remoto si `commentary.enabled: true`;
- credenciales de `ElevenLabs` si el TTS configurado las necesita.

## Ejecucion rapida

Desde la raiz del proyecto:

```bash
python scripts/track.py video_prueba_medio
```

Importante:
- con la configuracion actual del repo, ese comando usa tambien acciones (`PathCRF`) y comentarios;
- por tanto, para la ejecucion por defecto deben existir `external/pathcrf`, `external/llama.cpp` y el bloque `llama_cpp` en `config.yaml`, o bien un backend remoto para comentarios;
- si quieres una ejecucion mas simple, desactiva `actions.enabled` y `commentary.enabled` en `config.yaml`, o al menos lanza `--no-commentary` para evitar la fase de comentarios.

Ese comando ejecuta el pipeline principal de tracking y genera:
- `output/tracks_json/tracker/<video>_tracks.json`
- `output/tracking/<video>_tracking.mp4`

Otros comandos utiles:

```bash
python scripts/detect.py video_prueba_medio modelo_base
python scripts/homography.py video_prueba_medio modelo_base
python interfaz/app.py
```

## Configuracion

La configuracion vive en `config.yaml`. Ahi se definen:
- rutas de videos, datasets y modelos;
- parametros de deteccion y tracking;
- colores de equipos;
- opciones de visualizacion y salidas.

Si trabajas en un entorno sin interfaz grafica, usa `visualization.show_output: false`.

## Variables de entorno

`.env` no es obligatorio para todo el proyecto.

Solo hace falta si vas a usar alguna de estas opciones:
- descarga de datasets desde Roboflow;
- TTS con `ElevenLabs`;
- backend remoto de `llama.cpp` mediante `LLAMA_CPP_BASE_URL`.

Si usas `llama.cpp` local con el bloque `llama_cpp` dentro de `config.yaml`, no necesitas `LLAMA_CPP_BASE_URL`.

## Repos externos

Con el `config.yaml` actual, antes de ejecutar `python scripts/track.py video_prueba_medio` necesitas clonar estos repos externos:

### PathCRF

```bash
git clone https://github.com/hyunsungkim-ds/pathcrf.git external/pathcrf
```

Despues confirma que existe el trial configurado:

```text
external/pathcrf/saved/120/args.json
external/pathcrf/saved/120/model/state_dict_best_acc.pt
```

### llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git external/llama.cpp
```

Luego necesitas configurar el bloque `llama_cpp` dentro de `config.yaml` y tener un modelo GGUF accesible. En este repo, el chequeo por defecto espera esa configuracion local salvo que uses:
- `commentary.llm_base_url` en `config.yaml`, o
- `LLAMA_CPP_BASE_URL` en el entorno.

## Verificacion

Cuando ya tengas dependencias y repos externos:

```bash
python verify_setup.py
```

`verify_setup.py` comprueba tambien el runtime por defecto definido en `config.yaml`, incluyendo:
- modelo YOLO principal;
- video de prueba;
- `external/pathcrf` y su checkpoint si `actions.enabled: true`;
- el bloque `llama_cpp` en `config.yaml` o un endpoint remoto si `commentary.enabled: true`;
- credenciales de `ElevenLabs` si el TTS configurado las necesita.

Si no quieres depender de estos repos externos, desactiva `actions.enabled` y `commentary.enabled` en `config.yaml`.

## Modulos principales

- [football_ai/tracking](football_ai/tracking/README.md): pipeline de tracking por fases.
- [football_ai/actions](football_ai/actions/README.md): adaptadores e inferencia de acciones con PathCRF.
- [football_ai/commentaries](football_ai/commentaries/README.md): generacion de texto y audio para comentarios.
- [football_ai/positions](football_ai/positions/README.md): roles posicionales y alineaciones.
- [football_ai/posession](football_ai/posession/README.md): heuristica de posesion.
- [football_ai/visualization](football_ai/visualization/README.md): render de videos anotados.
