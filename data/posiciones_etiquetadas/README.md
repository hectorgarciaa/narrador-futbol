# data/posiciones_etiquetadas

Dataset etiquetado para roles posicionales.

Estructura:
- `labels/`: plantillas JSON de etiquetado manual por vídeo.
- `common/`: dataset acumulado (`base_table.csv`, samples derivados, métricas y metadata).
- `<match_id>_<timestamp>/`: export de una ejecución concreta del workflow.

Estos artefactos se generan localmente y están ignorados en git.
