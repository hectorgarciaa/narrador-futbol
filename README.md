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

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_setup.py
```

## Ejecucion rapida

Desde la raiz del proyecto:

```bash
python scripts/track.py video_prueba_medio
```

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

## Modulos principales

- [football_ai/tracking](football_ai/tracking/README.md): pipeline de tracking por fases.
- [football_ai/actions](football_ai/actions/README.md): adaptadores e inferencia de acciones con PathCRF.
- [football_ai/commentaries](football_ai/commentaries/README.md): generacion de texto y audio para comentarios.
- [football_ai/positions](football_ai/positions/README.md): roles posicionales y alineaciones.
- [football_ai/posession](football_ai/posession/README.md): heuristica de posesion.
- [football_ai/visualization](football_ai/visualization/README.md): render de videos anotados.
