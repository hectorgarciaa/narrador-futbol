# 🎙️ AI Football Commentator

Sistema de inteligencia artificial capaz de narrar partidos de fútbol de manera automática, combinando detección de jugadores, tracking, reconocimiento de acciones, generación de comentarios con LLM y síntesis de voz.

---

## 📖 Descripción del Proyecto

El objetivo de este proyecto es crear un **pipeline completo de narración de partidos de fútbol** usando IA. El sistema es capaz de:

1. Detectar jugadores, árbitros y balón.
2. Identificar a cada jugador con su nombre y dorsal.
3. Reconocer las acciones del partido (pases, tiros, goles, faltas, tarjetas, penales…).
4. Generar comentarios expresivos y contextualizados usando un LLM.
5. Convertir los comentarios en audio para narración en tiempo real.

---

## 🏟️ Fases del Proyecto

### **Fase 1: Detección de jugadores y objetos**
- Entrada: Video del partido y un documento con información de los jugadores (nombre, dorsal, posición, color de piel, altura, peso, posición en el campo).  
- Tareas:
  - Detectar los 22 jugadores, árbitros y balón.
  - Fine-tuning de un modelo YOLO para clases específicas de fútbol: jugador, portero, árbitro, pelota.
  - Tracking de jugadores para asignar un ID único y mantener seguimiento durante el partido.
  - Identificación de jugadores y equipos mediante técnicas clásicas de visión por computador (análisis de pixels del bounding box, clustering de colores de camiseta con KMeans, etc.).

### **Fase 2: Detección de acciones**
- Objetivo: Detectar acciones como pase, pase largo, tiro, gol, falta, tarjeta, penal, etc.  
- Enfoque:
  - Seleccionar una red preentrenada para detección de acciones humanas.
  - Reentrenarla para acciones de fútbol mediante fine-tuning.

### **Fase 3: Generación de comentarios con LLM**
- Integrar la información de detección de jugadores y acciones.
- Enviar los datos al LLM para generar comentarios expresivos y coherentes con el contexto del partido.

### **Fase 4: Conversión de texto a audio**
- Transformar los comentarios generados en audio para crear una experiencia de narración completa.

### **Fase 5: Optimización para transmisión en tiempo real**
- Integrar todas las fases en un pipeline eficiente capaz de hacer inferencia en directo como si fuese una retransmisión en vivo.

---

## 🛠️ Tecnologías y Herramientas

- **Detección de Objetos:** YOLOv8 / YOLOv11, fine-tuning para clases de fútbol.  
- **Tracking de Jugadores:** Asignación de IDs, seguimiento y mapeo a nombre/dorsal.  
- **Reconocimiento de Equipos y Jugadores:** Análisis de pixels, clustering de colores (KMeans) y ML clásico.  
- **Detección de Acciones:** Redes preentrenadas para acción humana, adaptadas a fútbol.  
- **Generación de Comentarios:** LLM contextual y expresivo.  
- **Síntesis de Voz:** Conversión de texto a audio.  
- **Optimización:** Pipeline para transmisión en tiempo real.

---

## 🗂️ Estructura del Proyecto

data/
├─ detection/FootBall-Detection-2/
│ ├─ train/ # Imágenes y labels para entrenamiento
│ ├─ valid/ # Imágenes y labels para validación
│ ├─ test/ # Imágenes y labels para pruebas
│ └─ data.yaml # Configuración de dataset
├─ partidoPrueba/
│ └─ 08fd33_4.mp4 # Video de prueba

models/
├─ finetuning/v11/yolov11m/weights/best.pt
├─ yolo/v8/yolov8m.pt
├─ yolo/v8/yolov8x.pt
├─ yolo/v11/yolov11m.pt
└─ yolo/v11/yolov11x.pt

output/
├─ pruebaDeteccionYolo/08fd33_4.avi
└─ pruebaDeteccionFinetuning/08fd33_4.avi

src/
├─ detection/
│ ├─ detection/ # Scripts YOLO
│ ├─ tracking/ # Scripts de seguimiento de jugadores
│ └─ identificacion/ # Scripts de identificación de jugadores (nombre, dorsal, equipo)
├─ deteccionAcciones/ # Scripts para la segunda fase (detección de acciones)
├─ llm/ # Scripts para generar comentarios
└─ audio/ # Scripts para conversión de texto a voz

---

## 📦 Instalación

### **Requisitos Previos**
- Python 3.8 o superior
- CUDA (opcional, para aceleración GPU)
- Git

### **1. Clonar el repositorio**
```bash
git clone <repository-url>
cd narrador-futbol
```

### **2. Crear entorno virtual**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### **3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

### **4. Configurar variables de entorno**
Crea un archivo `.env` en la raíz del proyecto basándote en `.env.example`:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edita el archivo `.env` y añade tus credenciales de Roboflow:
```env
ROBOFLOW_API_KEY=tu_clave_aqui
ROBOFLOW_PUBLISHABLE_KEY=tu_clave_publicable_aqui
```

### **5. Descargar modelos YOLO (opcional)**
Si necesitas descargar los modelos base de YOLO:
```bash
python src/detection/descargarModelosYolo.py
```

### **6. Verificar instalación**
```bash
python -c "from src.utils import get_config; config = get_config(); print('✓ Configuración cargada correctamente')"
```

---

## 🚀 Uso

### **Tracking de Jugadores**
Para ejecutar el sistema completo de detección y tracking:

```bash
cd src/tracking
python main.py
```

El script:
1. Carga el modelo fine-tuned de YOLO
2. Procesa el video especificado en `config.yaml`
3. Detecta y rastrea jugadores, porteros, árbitros y balón
4. Identifica equipos mediante clustering de colores
5. Genera un video con las anotaciones
6. Muestra métricas de evaluación

### **Detección con YOLO Base**
Para probar detección con modelos YOLO preentrenados:

```bash
cd src/detection
python pruebaDeteccionYolo.py
```

### **Fine-tuning de YOLO**
Para entrenar un modelo en tu propio dataset:

```bash
cd src/detection
python finetuning.py
```

Configura los parámetros en `config.yaml`:
```yaml
finetuning:
  epochs: 50
  batch: 16
  imgsz: 640
```

### **Detección de Balón**
Para detección especializada del balón con arquitectura DetectR8:

```bash
cd src/detection
python detectionBall.py
```

---

## 🧹 Limpieza del Proyecto

El proyecto incluye un script de limpieza para eliminar archivos generados y liberar espacio (~5 GB):

### **Limpieza completa**
```bash
python clean_project.py --all
```

### **Limpieza selectiva**
```bash
# Solo caché de Python
python clean_project.py --cache

# Solo archivos de output (videos procesados, JSONs)
python clean_project.py --output

# Solo datasets (pueden descargarse de nuevo)
python clean_project.py --datasets

# Solo modelos YOLO base (pueden descargarse de nuevo)
python clean_project.py --models

# Combinaciones
python clean_project.py --cache --output --logs
```

### **¿Qué se elimina?**

| Categoría | Tamaño | Regenerable | Comando |
|-----------|--------|-------------|---------|
| Caché Python (`__pycache__`, `*.pyc`) | < 1 MB | Automático | `--cache` |
| Output (videos, JSONs) | ~1.3 GB | Reejecutando scripts | `--output` |
| Datasets | ~3.3 GB | `descargarDataSetDeteccion.py` | `--datasets` |
| Modelos YOLO base | ~500 MB | `descargarModelosYolo.py` | `--models` |
| Logs | < 1 MB | Automático | `--logs` |

**⚠️ Nota:** Los modelos fine-tuned **NO se eliminan** (requieren horas de entrenamiento).

### **Regenerar archivos eliminados**

Después de la limpieza, puedes regenerar lo necesario:

```bash
# Descargar datasets
python src/detection/descargarDataSetDeteccion.py

# Descargar modelos YOLO base
python src/detection/descargarModelosYolo.py

# Generar output (ejecutar tracking)
cd src/tracking
python main.py
```

---

## ⚙️ Configuración

Toda la configuración del sistema se gestiona desde `config.yaml` en la raíz del proyecto:

```yaml
# Rutas de modelos y datos
paths:
  models:
    finetuned_player: "models/finetuning/v11/yolov11m/weights/best.pt"
  data:
    video_prueba: "data/partidoPrueba/partido.mp4"

# Parámetros de tracking
tracking:
  track_thresh: 0.5
  match_thresh: 0.945
  minimum_consecutive_frames: 5

# Colores de equipos (RGB)
teams:
  Real Madrid:
    color_rgb: [255, 127, 127]
  Wolfsburgo:
    color_rgb: [224, 77, 196]
```

Modifica estos valores según tus necesidades.

---

## 📊 Métricas de Evaluación

El sistema incluye evaluación automática de tracking con métricas:

- **Coverage**: Porcentaje de frames donde aparece cada track
- **Fragments**: Número de gaps en cada track
- **Mean Speed**: Velocidad promedio de movimiento
- **Team Flip Rate**: Tasa de cambios incorrectos de equipo
- **Color Consistency**: Coherencia en el color de camiseta detectado

---

## 🐛 Solución de Problemas

### Error: "Archivo de configuración no encontrado"
Asegúrate de que `config.yaml` existe en la raíz del proyecto.

### Error: "Could not open video"
Verifica que la ruta del video en `config.yaml` es correcta y el archivo existe.

### Error: "CUDA out of memory"
Reduce el tamaño del batch en `config.yaml` o usa CPU:
```python
device = 'cpu'  # en lugar de 'cuda'
```

### Imports fallando
Asegúrate de ejecutar los scripts desde sus directorios correspondientes:
```bash
cd src/tracking  # antes de ejecutar main.py
cd src/detection # antes de ejecutar finetuning.py
```

---

## 📝 Estado del Proyecto

✅ **Completado (Fase 1)**
- Detección de objetos con YOLO
- Fine-tuning para clases de fútbol
- Tracking multi-objeto con ByteTrack
- Identificación de equipos
- Sistema de evaluación

🚧 **En desarrollo**
- Mejoras en identificación de jugadores individuales
- Optimización de rendimiento

📅 **Pendiente (Fases 2-5)**
- Detección de acciones
- Generación de comentarios con LLM
- Síntesis de voz
- Pipeline en tiempo real

---

## 🤝 Contribuciones

Este es un proyecto de TFG. Para reportar bugs o sugerencias, por favor contacta al autor.

---

## 📄 Licencia

[Especificar licencia]

---

## 👤 Autor

Héctor García  y Carlos Mantilla 
Universidad Complutense de Madrid 
Trabajo de Fin de Grado 2026