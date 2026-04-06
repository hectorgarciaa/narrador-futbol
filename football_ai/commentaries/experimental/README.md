# experimental

Codigo experimental relacionado con TTS y benchmarks de comentarios.

Contenido actual:

- `qwen_voice.py`: backends opcionales `qwen` y `qwen_cpp`, separados de `voice.py` para no sobrecargar la ruta estable basada en XTTS.

Notas:

- la ruta de produccion sigue siendo `football_ai/commentaries/voice.py`;
- estos modulos se mantienen para comparativas, benchmarking y futuras pruebas;
- el informe tecnico completo de decisiones y resultados esta en `football_ai/report/qwen_tts_evaluation_report.md`.
