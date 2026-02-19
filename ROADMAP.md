# 🗺️ Hoja de Ruta Detallada del Proyecto (Fase 1)

## 1. Visión General e Infraestructura
*   [ ] **Guía Maestro**: Lee el [README.md](README.md) principal para entender la arquitectura en 5 fases. Actualmente estamos centrados en la **Fase 1**.
*   [ ] **Gestión de Artefactos**: Revisa [.gitignore](.gitignore). Observarás que `data/`, `models/` y `output/` están mayoritariamente excluidos para evitar subir binarios pesados.
*   [ ] **Orquestación**: Analiza [config.yaml](config.yaml). Es el cerebro del proyecto; controla desde las rutas de los modelos hasta los umbrales de confianza del tracker.

## 2. Configuración del Entorno de Desarrollo
*   [ ] **Dependencias**: Revisa [pyproject.toml](pyproject.toml) y [requirements.txt](requirements.txt). Usamos `ultralytics` para YOLO y `scikit-learn` para el clustering de equipos.
*   [ ] **Validación**: Ejecuta [verify_setup.py](verify_setup.py). Este script asegura que tienes la estructura de carpetas necesaria y las variables de entorno correctas.
*   [ ] **Variables de Entorno**: Crea tu `.env` basado en `.env.example` (necesario para descargar datasets de Roboflow).

## 3. Inmersión en `football_ai` (Arquitectura Core)
*   [ ] **Configuración Dinámica**: Lee [football_ai/core/config.py](football_ai/core/config.py). Entiende cómo la función `get_config()` carga el YAML y resuelve las rutas absolutas.
*   [ ] **Detección Especializada**:
    *   Revisa [football_ai/detection/detector.py](football_ai/detection/detector.py) para ver el wrapper genérico de YOLO.
    *   Mira [football_ai/detection/ball_detector.py](football_ai/detection/ball_detector.py) para entender cómo el modelo DetectR8 maneja específicamente la pelota.
*   [ ] **Tracking y Lógica de Negocio**:
    *   **ByteTrack**: Estudia [football_ai/tracking/byte_tracker.py](football_ai/tracking/byte_tracker.py), una extensión del algoritmo original integrada con `supervision`.
    *   **Penalización**: Analiza cómo se usa la información de equipo para mejorar la persistencia de IDs en [football_ai/tracking/tracker.py](football_ai/tracking/tracker.py).
*   [ ] **Identificación de Equipos**:
    *   [football_ai/identification/shirt_detector.py](football_ai/identification/shirt_detector.py): El núcleo donde se usa KMeans sobre colores LAB para segmentar la camiseta.
    *   [football_ai/identification/team_detector.py](football_ai/identification/team_detector.py): Decide a qué equipo pertenece un track basándose en el color predominante.
*   [ ] **Evaluación y Visualización**:
    *   [football_ai/evaluation/evaluator.py](football_ai/evaluation/evaluator.py): Calcula métricas como fragmentación y cobertura de tracks.
    *   [football_ai/visualization/drawer.py](football_ai/visualization/drawer.py): La clase encargada de pintar los "boxes" y IDs sobre el vídeo original.

## 4. Scripts Ejecutables (Flujo de Trabajo)
*   [ ] **Preparación de Datos**:
    *   Ejecuta [scripts/data/download_models.py](scripts/data/download_models.py) para bajar los pesos base.
    *   Ejecuta [scripts/data/download_datasets.py](scripts/data/download_datasets.py) para obtener los datos de entrenamiento (requiere API Key).
*   [ ] **Entrenamiento**:
    *   Entiende el proceso de fine-tuning en [scripts/train/finetune_player.py](scripts/train/finetune_player.py).
*   [ ] **Ejecución del Pipeline**:
    *   [scripts/track.py](scripts/track.py): El comando principal. Orquesta detección, tracking, identificación y guarda los resultados en `output/` tanto en vídeo como en JSON.

## 5. Experimentación y Análisis
*   [ ] **Optimización**: Revisa [scripts/track_experiments.py](scripts/track_experiments.py). Se usa para buscar los mejores hiperparámetros del tracker (Grid Search).
*   [ ] **Notebooks**: Explora [experiments/visualization/experiments_comparator.ipynb](experiments/visualization/experiments_comparator.ipynb) para comparar cómo rinden distintas configuraciones visualmente.

## 6. Mantenimiento
*   [ ] **Limpieza**: [clean_project.py](clean_project.py) es vital para borrar caches de notebooks y salidas de `output/` antes de realizar nuevas pruebas limpias.
