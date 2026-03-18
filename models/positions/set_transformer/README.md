# models/positions/set_transformer

Runs del clasificador de roles por Set Transformer.

Cada carpeta de timestamp contiene:
- `set_transformer_checkpoint.pt`
- `metrics.json`
- `training_history.csv`
- `split.json`

Solo se conserva el último run local salvo que se necesite comparar experimentos.
