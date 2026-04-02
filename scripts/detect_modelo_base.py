#!/usr/bin/env python3
"""
Script para detectar objetos con el modelo fine-tuned modelo_base (DFL-Bundesliga)
sobre el video video_yt_30s.
Detecta todas las clases: player, goalkeeper, referee, ball.
"""

import sys
from pathlib import Path
import cv2
import json
from datetime import datetime

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from football_ai.core import get_config, get_logger, Logger


def detect_on_video(video_key, output_dir="output/deteccion_finetuned"):
    """
    Procesa un video con detecciones YOLO fine-tuned (modelo_base) y guarda resultados.
    Muestra todas las clases detectadas por el modelo (player, referee, ball, goalkeeper).
    
    Args:
        video_key: Clave en config.yaml (ej: 'video_yt_30s')
        output_dir: Directorio de salida
    """
    
    # Cargar configuración
    config = get_config()
    video_path = config.get_path('paths', 'data', video_key)
    model_path = config.get_path('paths', 'models', 'modelo_base')
    
    # Resolver rutas
    output_dir = Path(config.get_path('paths', 'output', 'base', create_if_missing=True)) / "deteccion_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Detectando con modelo fine-tuned (modelo_base)")
    print(f"Video: {video_path}")
    print(f"Modelo: {model_path}")
    print(f"Salida: {output_dir}")
    print("-" * 70)
    
    # Cargar modelo
    print("Cargando modelo fine-tuned...")
    model = YOLO(str(model_path))
    
    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {video_path}")
        return
    
    # Propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.2f}fps ({total_frames} frames)")
    
    # Crear writer para video de salida
    output_video = output_dir / f"{video_path.stem}_finetuned_detections.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Almacenar detecciones
    all_detections = []
    
    frame_count = 0
    print(f"\nProcesando frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Ejecutar detección
        results = model(frame, conf=0.01, verbose=False)
        
        # Crear resultado filtrado para dibujar (todas las clases)
        annotated_frame = frame.copy()
        
        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            
            # Color por clase
            color_map = {
                'player': (0, 255, 0),        # Verde
                'goalkeeper': (0, 255, 255), # Cyan
                'referee': (255, 0, 0),      # Azul
                'ball': (0, 0, 255)          # Rojo
            }
            color = color_map.get(class_name, (255, 255, 255))
            
            # Dibujar bbox y label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Guardar frame
        out.write(annotated_frame)
        
        # Almacenar detecciones para JSON (todas las clases)
        frame_detections = {
            'frame': frame_count,
            'detections': []
        }
        
        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            
            detection = {
                'class': class_name,
                'class_id': int(box.cls),
                'confidence': float(box.conf),
                'bbox': {
                    'x1': float(box.xyxy[0][0]),
                    'y1': float(box.xyxy[0][1]),
                    'x2': float(box.xyxy[0][2]),
                    'y2': float(box.xyxy[0][3]),
                },
                'center': {
                    'x': float((box.xyxy[0][0] + box.xyxy[0][2]) / 2),
                    'y': float((box.xyxy[0][1] + box.xyxy[0][3]) / 2),
                }
            }
            frame_detections['detections'].append(detection)
        
        all_detections.append(frame_detections)
        
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progreso: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    # Guardar JSON
    output_json = output_dir / f"{video_path.stem}_finetuned_detections.json"
    with open(output_json, 'w') as f:
        json.dump({
            'video': str(video_path),
            'model': str(model_path),
            'model_type': 'fine-tuned (DFL-Bundesliga)',
            'timestamp': datetime.now().isoformat(),
            'fps': fps,
            'resolution': {'width': width, 'height': height},
            'total_frames': total_frames,
            'frames': all_detections
        }, f, indent=2)
    
    # Resumen
    print("-" * 70)
    print(f"\n✓ Procesamiento completado")
    print(f"Frames procesados: {frame_count}")
    print(f"Video guardado: {output_video}")
    print(f"JSON guardado: {output_json}")
    
    # Estadísticas
    total_detections = sum(len(f['detections']) for f in all_detections)
    avg_per_frame = total_detections / frame_count if frame_count > 0 else 0
    
    print(f"\nEstadísticas:")
    print(f"  Total detecciones: {total_detections}")
    print(f"  Promedio por frame: {avg_per_frame:.1f}")
    
    # Contar por clase
    class_counts = {}
    for frame_det in all_detections:
        for det in frame_det['detections']:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"  Por clase:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cls}: {count}")


if __name__ == "__main__":
    detect_on_video(video_key="video_yt_30s")
