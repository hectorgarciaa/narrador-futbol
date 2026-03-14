# experiments/positions

Utilidades y notebook para construir un dataset supervisado de rol nominal de jugador a partir de tracking proyectado al campo.

## Archivos

- `position_dataset.py`: funciones para preparar observaciones, validar etiquetas de rol, inferir orientación de ataque y construir samples con features tabulares + tensor de compañeros.
- `position_role_dataset.ipynb`: flujo end-to-end de creación del dataset.
- `position_role_workflow.py`: implementación equivalente en script (`.py`) del flujo completo (preparación, etiquetado interactivo en Jupyter y exportación).

## Uso recomendado en `.py` (sin depender del notebook)

### 1) Preparar schedule + JSON de etiquetas

```bash
cd /Users/carloscole/narrador-futbol
python -m experiments.positions.position_role_workflow prepare \
  --project-root /Users/carloscole/narrador-futbol \
  --video-filename "A1606b0e6_0 (50).mp4" \
  --label-every-seconds 6 \
  --label-window-frames 20
```

### 2) Etiquetar con UI estable desde Jupyter (solo 1 imagen: frame actual)

```python
from pathlib import Path
from experiments.positions.position_role_workflow import WorkflowConfig, create_labeler_ui

config = WorkflowConfig(
    project_root=Path("/Users/carloscole/narrador-futbol"),
    video_filename="A1606b0e6_0 (50).mp4",
    label_every_seconds=6.0,
    label_window_frames=20,
)

context, labeler, summary = create_labeler_ui(config)
print(summary)
labeler.display()
```

Notas:
- El preview se pinta en un único `widgets.Image` (no usa `matplotlib` inline), para evitar acumulación de imágenes.
- Si re-ejecutas la celda de creación de UI, la instancia anterior se cierra automáticamente.

### 3) Exportar dataset desde labels guardadas

```bash
cd /Users/carloscole/narrador-futbol
python -m experiments.positions.position_role_workflow export \
  --project-root /Users/carloscole/narrador-futbol \
  --video-filename "A1606b0e6_0 (50).mp4"
```

## Notebook mínimo

`position_role_dataset.ipynb` se ha simplificado a 2 celdas:
1. Setup de `sys.path` + imports del workflow.
2. Configuración de vídeo + creación de UI (`create_labeler_ui(config)`).

Todo el pipeline (schedule, labels JSON, UI, propagación, export y upsert a `common`) lo ejecuta `position_role_workflow.py`.

## Labels v1

`POR, CI, LI, DFC_IZQ, DFC_CENT, DFC_DER, LD, CD, MC, MI, MD, EI, ED, DC`
