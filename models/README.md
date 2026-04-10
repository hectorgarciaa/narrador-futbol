# models

Directorio de modelos de machine learning. En general, los archivos de pesos grandes (`.pt`) no se versionan en git por su tamaño. La excepción actual es el checkpoint del Set Transformer de roles, que sí se puede versionar porque su tamaño es reducido.

## Estructura esperada

```
models/
├── yolo/
│   ├── v8/
│   │   ├── yolov8m.pt
│   │   ├── yolov8l.pt
│   │   └── yolov8x.pt
│   └── v11/
│       ├── yolov11m.pt
│       ├── yolov11l.pt
│       └── yolov11x.pt
├── finetuning/
│   ├── yolov11m.pt          ← modelo fine-tuned de jugadores
│   └── yolov11x.pt          ← alternativa con modelo más grande
├── positions/
│   └── set_transformer/
│       └── <timestamp>/     ← checkpoint + métricas del clasificador de roles
└── finetuning-balon/
    └── yolov11m.pt          ← modelo fine-tuned de balón
```

## Tipos de modelos

### Modelos base YOLO (`yolo/`)

Modelos preentrenados de Ultralytics en COCO (80 clases). Se usan como punto de partida para el fine-tuning y para detecciones de referencia sin fine-tuning.

- **v8**: YOLOv8m, YOLOv8l, YOLOv8x (diferentes tamaños, velocidad/precisión)
- **v11**: YOLOv11m, YOLOv11l, YOLOv11x (arquitectura más reciente)

Referenciados en `config.yaml` como `paths.models.yolo_v8_m`, `yolo_v11_m`, etc.

### Modelo fine-tuned de jugadores (`finetuning/`)

Modelo YOLO fine-tuned sobre el dataset de Roboflow para detectar las 4 clases específicas: `player`, `goalkeeper`, `referee`, `ball`. Usado por `Tracker` como detección principal.

Referenciado en `config.yaml` como `paths.models.finetuned_player`.

### Modelo fine-tuned de balón (`finetuning-balon/`)

Modelo YOLO fine-tuned específicamente para detección de balón, entrenado con `DetectR8` (cabeza con `reg_max=8` en lugar de 16). Optimizado para detectar objetos pequeños con mayor precisión.

Referenciado en `config.yaml` como `paths.models.finetuned_ball`.

### Modelo posicional Set Transformer (`positions/set_transformer/`)

Checkpoint entrenado para clasificar roles nominales a partir del dataset etiquetado en `data/posiciones_etiquetadas/common/base_table.csv`.

Este checkpoint sí puede versionarse en git cuando sea el modelo vigente, porque pesa poco frente al resto de pesos del repositorio.

Cada run guarda:
- `set_transformer_checkpoint.pt`
- `metrics.json`
- `training_history.csv`
- `split.json`

El checkpoint se usa desde `experiments/positions/set_transformer_pipeline.py`.

## Cómo poblar el directorio

### Modelos base (descarga automática)

```bash
python scripts/data/download_models.py
```

Este script usa `YOLO(nombre)` de Ultralytics que descarga automáticamente desde los servidores de Ultralytics si el modelo no está en caché, y lo guarda en la ruta correcta.

### Modelos fine-tuned (generación manual)

1. Ejecutar el fine-tuning:
   ```bash
   python scripts/train/finetune_player.py
   ```
2. YOLO guarda los pesos en `models/finetuning/finetuning/weights/best.pt`.
3. Copiar manualmente al directorio correspondiente:
   ```bash
   # Windows
   copy models\finetuning\finetuning\weights\best.pt models\finetuning\yolov11m.pt
   ```

### Alternativa: copiar desde entrenamiento previo

Si los modelos ya fueron entrenados y están disponibles externamente, simplemente copiarlos a las rutas indicadas arriba.

## Configuración en `config.yaml`

Todas las rutas están bajo `paths.models` en `config.yaml`. Modificar ese archivo si se quiere usar un modelo diferente sin tocar el código.

En la configuración actual, el tracking principal lee `paths.models.modelo_base` y esa ruta apunta por defecto a:

```text
models/finetuning/yolov11m/weights/best.pt
```

Los pesos base de `models/yolo/` se mantienen para descargas/referencias, pero no son el detector por defecto del pipeline de tracking.
