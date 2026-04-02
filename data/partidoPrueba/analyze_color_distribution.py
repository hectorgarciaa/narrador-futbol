#!/usr/bin/env python3
"""
Script alternativo para inspeccionar características de color del video.
Analiza píxeles reales del video para detectar patrones de color.
"""

import cv2
import numpy as np

video_path = "AD Ferroviaria-Canillejas CF_clip_30s.mp4"

print(f"Analizando: {video_path}")
print("=" * 70)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: No se pudo abrir el video")
    exit(1)

# Obtener propiedades
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Resolución: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print()

# Analizar varios frames para detectar rangos de color
frames_to_check = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4]

print("ANÁLISIS DE VALORES DE COLOR:")
print("-" * 70)

all_bgr_values = []

for idx, frame_num in enumerate(frames_to_check):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    # Obtener estadísticas de color por canal
    b_mean, g_mean, r_mean = cv2.mean(frame)[:3]
    
    # Analizar distribución
    b_min, b_max = frame[:,:,0].min(), frame[:,:,0].max()
    g_min, g_max = frame[:,:,1].min(), frame[:,:,1].max()
    r_min, r_max = frame[:,:,2].min(), frame[:,:,2].max()
    
    print(f"Frame {frame_num}:")
    print(f"  B: min={b_min}, max={b_max}, mean={b_mean:.1f}")
    print(f"  G: min={g_min}, max={g_max}, mean={g_mean:.1f}")
    print(f"  R: min={r_min}, max={r_max}, mean={r_mean:.1f}")
    
    all_bgr_values.append((b_mean, g_mean, r_mean))

cap.release()

print()
print("=" * 70)
print("INTERPRETACIÓN:")
print("-" * 70)
print("OpenCV SIEMPRE carga videos en BGR (Blue, Green, Red)")
print()
print("El video fue comprimido en:")
print("  - Codec: H.264/H.265 (típicamente)")
print("  - Espacio de color: YUV420p (estándar para vídeo)")
print()
print("Flujo:")
print("  Video almacenado (YUV) → OpenCV convierte a BGR")
print("  OpenCV BGR → YOLO convierte a RGB")
print()
print("RECOMENDACIONES PARA YOLO:")
print("-" * 70)
print("✓ YOLO está entrenado con normalización estándar [0, 255]")
print("✓ OpenCV maneja la conversión YUV→BGR automáticamente")
print("✓ YOLO hace BGR→RGB internamente si es necesario")
print("✓ Esto es NORMAL y está contemplado en ultralytics")
print()
print("Si hay degradación de rendimiento, probablemente es por:")
print("  - Diferencia en resolución de entrenamiento vs inferencia")
print("  - Diferencia en aumentación de datos")
print("  - Diferencia en objeto detection (clases, FOV, etc.)")
print("  - NO por el espacio de color (eso está normalizado)")
print()
print("Para máxima compatibilidad:")
print("  - Usar imágenes RGB normalizadas [0, 1] o [0, 255]")
print("  - YOLO de ultralytics maneja ambas automáticamente")
