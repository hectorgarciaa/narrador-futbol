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
└── partidoPrueba/                    # Vídeos de prueba (no versionados)
    ├── partido.mp4                   # Vídeo completo
    ├── partido_medio.mp4             # Mitad del vídeo (para pruebas rápidas)
    ├── 08fd33_4.mp4                  # Clip específico de partido
    ├── 08fd33_4_medio.mp4            # Versión reducida del clip
    └── cortarVideo.py                # Script auxiliar para recortar vídeos
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

## Notas

- Los archivos `.gitkeep` en `detection/` mantienen la estructura de carpetas en el repositorio aunque no haya datos descargados.
- El directorio `output/` (generado por los scripts) tampoco se versiona. Se crea automáticamente en tiempo de ejecución.
