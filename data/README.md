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
├── partidoPrueba/                    # Vídeos de prueba para tracking/detección
│   ├── partido.mp4                   # Vídeo completo
│   ├── partido_medio.mp4             # Mitad del vídeo (para pruebas rápidas)
│   ├── 08fd33_4.mp4                  # Clip específico de partido
│   ├── 08fd33_4_medio.mp4            # Versión reducida del clip
│   └── cortarVideo.py                # Script auxiliar para recortar vídeos
└── partidosPosiciones/               # Clips para dataset de roles posicionales
    ├── test (1).mp4
    ├── ...
    └── test (29).mp4
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

Vídeos de partido reales usados para desarrollo y validación del sistema. En general se recomienda no versionarlos en git por tamaño y mantenerlos fuera del repositorio cuando sea posible.

Las rutas están configuradas en `config.yaml` bajo `paths.data`:

| Clave en config | Vídeo |
|---|---|
| `video_prueba` | `partido.mp4` |
| `video_prueba_medio` | `partido_medio.mp4` |
| `video_08fd33` | `08fd33_4.mp4` |
| `video_08fd33_medio` | `08fd33_4_medio.mp4` |
| `video_test_*` | Clips en `data/partidosPosiciones/test (N).mp4` |

### `cortarVideo.py`
Script auxiliar para generar versiones recortadas de un vídeo (por ejemplo, la primera mitad). Las versiones `_medio` permiten ejecutar experimentos más rápidamente durante el desarrollo sin procesar el vídeo completo.

---

## Clips para posiciones (`data/partidosPosiciones/`)

Conjunto de clips usado por el flujo de `experiments/positions` para construir dataset supervisado de roles posicionales por jugador.

Datos actuales en el repositorio:
- 14 vídeos `.mp4` (nombres tipo `test (N).mp4`)
- Tamaño aproximado total: `~282 MB`

Estos vídeos se consumen desde `football_ai.positions.data` mediante `list_position_videos(...)`.

### Relación con tracks

El notebook `experiments/positions/position_role_dataset.ipynb` necesita, para cada vídeo, un JSON de tracking con `field_position_m` por jugador:
- esperado: `output/tracks_json/tracker/<video_sanitizado>_tracks.json`
- fallback legacy: `output/tracks_json/tracker/tracks.json`

Ejemplo de sanitización de nombre:
- `test (1).mp4` → `test_1_tracks.json`
- `test (11).mp4` → `test_11_tracks.json`

---

## Notas

- Los archivos `.gitkeep` en `detection/` mantienen la estructura de carpetas en el repositorio aunque no haya datos descargados.
- El directorio `output/` (generado por los scripts) tampoco se versiona. Se crea automáticamente en tiempo de ejecución.
