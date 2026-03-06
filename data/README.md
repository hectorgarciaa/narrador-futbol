# data

Datos de entrada del sistema: vídeos de prueba y datasets de entrenamiento para la detección de objetos.

## Estructura

```
data/
├── detection/                        # Datasets de detección (no versionados)
│   ├── football-ai-2/                # Dataset principal (Roboflow, formato YOLOv11)
│   │   └── data.yaml               # Referenciado en config.yaml → paths.data.dataset_football_ai
│   └── FootBall-Detection-2/         # Dataset alternativo
│       └── data.yaml
├── metadata/
│   └── teams.json                    # Equipos del partido + color de camiseta
├── partidoPrueba/                    # Vídeos de prueba (no versionados)
│   ├── partido.mp4                   # Vídeo completo
│   ├── partido_medio.mp4             # Mitad del vídeo (para pruebas rápidas)
│   ├── 08fd33_4.mp4                  # Clip específico de partido
│   ├── 08fd33_4_medio.mp4            # Versión reducida del clip
│   └── cortarVideo.py                # Script auxiliar para recortar vídeos
└── partidosPosiciones/               # Vídeos para dataset de roles/posiciones (no versionados)
    ├── test (1).mp4
    ├── test (2).mp4
    ├── ...
    └── .gitkeep                      # Mantiene la carpeta versionada
```

---

## Datasets de detección (`data/detection/`)

Descargados automáticamente desde Roboflow con `scripts/data/download_datasets.py`. No se versionan en git por su tamaño.

**Formato:** YOLOv11 (imágenes + anotaciones en formato YOLO, con `data.yaml` que describe clases y rutas).

**Clases:** `player`, `goalkeeper`, `referee`, `ball`.

**Descarga:**
```bash
# Requiere ROBOFLOW_API_KEY en .env
python scripts/data/download_datasets.py
```

Las rutas a los `data.yaml` están configuradas en `config.yaml` bajo `paths.data.dataset_football_ai` y `paths.data.dataset_football_detection`.

---

## Vídeos de prueba (`data/partidoPrueba/`)

Vídeos de partido reales usados para desarrollo y validación del sistema. **No se versionan en git** (añadir a `.gitignore`). Deben copiarse manualmente.

Las rutas están configuradas en `config.yaml` bajo `paths.data`:

| Clave en config | Vídeo |
|---|---|
| `video_prueba` | `partido.mp4` |
| `video_prueba_medio` | `partido_medio.mp4` |
| `video_08fd33` | `08fd33_4.mp4` |
| `video_08fd33_medio` | `08fd33_4_medio.mp4` |

### `cortarVideo.py`
Script auxiliar para generar versiones recortadas de un vídeo (por ejemplo, la primera mitad). Las versiones `_medio` permiten ejecutar experimentos más rápidamente durante el desarrollo sin procesar el vídeo completo.

---

## Vídeos de posiciones (`data/partidosPosiciones/`)

Carpeta para los clips usados al construir dataset de **rol nominal** por jugador (`POR`, `LI`, `MC`, `DC`, etc.) en `experiments/positions/position_role_dataset.ipynb`.

Estos vídeos también se usan con `scripts/track.py` mediante `TRACK_VIDEO_KEY`, usando las claves definidas en `config.yaml`:

| Clave en config | Vídeo |
|---|---|
| `video_posiciones_test_1` | `test (1).mp4` |
| `video_posiciones_test_2` | `test (2).mp4` |
| `video_posiciones_test_11` | `test (11).mp4` |
| `video_posiciones_test_12` | `test (12).mp4` |
| `video_posiciones_test_13` | `test (13).mp4` |
| `video_posiciones_test_14` | `test (14).mp4` |
| `video_posiciones_test_15` | `test (15).mp4` |
| `video_posiciones_test_16` | `test (16).mp4` |
| `video_posiciones_test_18` | `test (18).mp4` |
| `video_posiciones_test_20` | `test (20).mp4` |
| `video_posiciones_test_22` | `test (22).mp4` |
| `video_posiciones_test_26` | `test (26).mp4` |
| `video_posiciones_test_27` | `test (27).mp4` |
| `video_posiciones_test_29` | `test (29).mp4` |

Ejemplo de ejecución:

```bash
TRACK_VIDEO_KEY=video_posiciones_test_1 TRACK_SHOW_OUTPUT=0 python scripts/track.py
```

---

## Metadata de equipos (`data/metadata/`)

Aquí vive `teams.json`, que contiene nombre de equipo y color de camiseta para el detector de equipos.
Se referencia desde `config.yaml` en `teams.source_file`.

Ejemplo:

```json
{
  "default_color_space": "lab",
  "teams": [
    {"name": "Equipo A", "color": [255, 127, 127]},
    {"name": "Equipo B", "color": [224, 77, 196]}
  ]
}
```

---

## Notas

- Los archivos `.gitkeep` en `detection/` mantienen la estructura de carpetas en el repositorio aunque no haya datos descargados.
- Los vídeos `.mp4` no se versionan porque `.gitignore` tiene una regla global `*.mp4`. Por eso `data/partidosPosiciones/` no aparecía en Git hasta añadir `.gitkeep`.
- El directorio `output/` (generado por los scripts) tampoco se versiona. Se crea automáticamente en tiempo de ejecución.
