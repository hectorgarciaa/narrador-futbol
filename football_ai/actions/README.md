# actions

Modulo para deteccion de acciones a partir del tracking ya generado.

## Que hace

- adapta `tracks.json` al formato que espera PathCRF;
- ejecuta inferencia offline o rolling;
- postprocesa eventos para obtener acciones mas legibles;
- deja los eventos listos para consumo por el modulo de comentarios.

## Archivos clave

- `adapter.py`: conversion `tracks -> dataframe/parquet` para PathCRF.
- `inference.py`: ejecucion de PathCRF.
- `model.py`: carga y cache del modelo.
- `rolling.py`: fase online `RollingActionsPhase`.
- `postprocess.py`: consolidacion de eventos emitidos.
- `pathcrf_*.py`: reglas semanticas y enriquecimiento de eventos.

## Uso tipico

Este modulo se usa desde `scripts/actions/run_pathcrf.py` o desde el pipeline rolling del proyecto.

Si `actions.enabled=true` en `config.yaml`, el runtime espera un checkout de PathCRF en `external/pathcrf`, con el trial y checkpoint configurados en `actions.trial` y `actions.model_file`.
