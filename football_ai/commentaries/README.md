# commentaries

Modulo de generacion de comentarios y audio a partir de eventos estructurados.

## Que hace

- transforma eventos de partido en texto breve de narracion;
- sintetiza audio con el backend TTS configurado;
- guarda manifiestos y audios por ejecucion;
- expone una fase reutilizable para la interfaz y el pipeline.

## Archivos clave

- `generator.py`: construccion del comentario en texto.
- `voice.py`: sintesis de voz.
- `phase.py`: fase `CommentaryPhase` para el pipeline.
- `deferred_media.py`: ensamblado de audio diferido.
- `launcher.py` y `__main__.py`: utilidades de ejecucion.
- `eval_llm.py`: evaluacion manual del backend LLM.

## Nota

En este repo la parte de comentarios existe y se puede probar, pero todavia no representa un pipeline de produccion completamente cerrado.

