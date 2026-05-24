# scripts

Puntos de entrada por linea de comandos del proyecto.

## Scripts principales

- `track.py`: ejecuta el pipeline completo de tracking.
- `detect.py`: prueba rapida de deteccion YOLO sobre video.
- `homography.py`: depura deteccion y proyeccion al campo.
- `comparar_modelos.py`: compara pesos/modelos de deteccion.
- `profile_gpu_run.py`: perfila una ejecucion en GPU.

## Subcarpetas

- `data/`: descarga y conversion de datasets o modelos.
- `train/`: entrenamiento y fine-tuning.

## Uso

Ejecuta siempre desde la raiz del repo:

```bash
python scripts/track.py video_prueba_medio
python scripts/detect.py video_prueba_medio modelo_base
python scripts/homography.py video_prueba_medio modelo_base
```

