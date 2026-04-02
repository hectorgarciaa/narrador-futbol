#!/usr/bin/env python3
"""
Script para extraer un clip de 30 segundos del clip de 8 minutos.
"""

import cv2

# Ruta del video a procesar
input_video = "AD Ferroviaria-Canillejas CF_clip_2-10.mp4"
output_video = "AD Ferroviaria-Canillejas CF_clip_30s.mp4"

print(f"Creando clip de 30 segundos desde: {input_video}")
print("-" * 60)

# Abrir video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print(f"Error: No se pudo abrir {input_video}")
    exit(1)

# Obtener características
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calcular frames para 30 segundos desde el minuto 2:00
start_sec = 120  # 2:00 minutos (en medio del clip de 8 minutos)
end_sec = start_sec + 30  # 30 segundos después
start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

print(f"FPS: {fps}")
print(f"Resolución: {width} x {height}")
print(f"Extrayendo desde {start_sec}s a {end_sec}s ({end_frame - start_frame} frames)")
print("-" * 60)

# Crear writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    print(f"Error: No se pudo crear el archivo de salida")
    cap.release()
    exit(1)

# Saltar al frame de inicio
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_frame > end_frame:
        break
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print(f"✓ Clip de 30s guardado en: {output_video}")
print(f"Frames procesados: {frame_count}")
print(f"Duración: {frame_count / fps:.2f}s")
