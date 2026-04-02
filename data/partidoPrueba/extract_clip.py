#!/usr/bin/env python3
"""
Script para extraer un clip de video entre los minutos 2:00 y 10:00 sin audio.
También analiza y reporta las características del video.
"""

import cv2

# Ruta del video a procesar
input_video = "AD Ferroviaria-Canillejas CF.mp4"
output_video = "AD Ferroviaria-Canillejas CF_clip_2-10.mp4"

print(f"Analizando video: {input_video}")
print("-" * 60)

# Abrir video para análisis
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print(f"Error: No se pudo abrir {input_video}")
    exit(1)

# Obtener características
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps if fps > 0 else 0
duration_min = int(duration_sec // 60)
duration_sec_remainder = int(duration_sec % 60)

# Leer primer frame para analizar colores
ret, frame = cap.read()
if ret:
    # OpenCV usa BGR por defecto
    bgr_mean = frame.mean(axis=(0, 1))
    rgb_mean = bgr_mean[::-1]
    
    # LAB (convertir BGR a LAB)
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab_mean = lab_frame.mean(axis=(0, 1))
    
    # Hex color
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(rgb_mean[0]),
        int(rgb_mean[1]),
        int(rgb_mean[2])
    )
    
    print(f"FPS: {fps}")
    print(f"Resolución: {width} x {height}")
    print(f"Total frames: {total_frames}")
    print(f"Duración: {duration_min:02d}:{duration_sec_remainder:02d} ({duration_sec:.2f}s)")
    print(f"\nColores (primer frame):")
    print(f"  BGR (OpenCV): {tuple(bgr_mean.astype(int))}")
    print(f"  RGB: {tuple(rgb_mean.astype(int))}")
    print(f"  LAB (OpenCV): {tuple(lab_mean.astype(int))}")
    print(f"  HEX: {hex_color}")
    print("-" * 60)

# Calcular frames de inicio y fin (minuto 2:00 a 10:00)
start_min = 2
end_min = 10
start_frame = int(start_min * 60 * fps)
end_frame = int(end_min * 60 * fps)

print(f"\nExtrayendo clip de {start_min}:00 a {end_min}:00...")
print(f"Frames: {start_frame} a {end_frame} ({end_frame - start_frame} frames)")

# Crear writer sin audio
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
    
    if frame_count % 100 == 0:
        progress = ((frame_count) / (end_frame - start_frame)) * 100
        print(f"  Progreso: {frame_count}/{end_frame - start_frame} ({progress:.1f}%)")

cap.release()
out.release()

print(f"\n✓ Clip guardado en: {output_video}")
print(f"Duración del clip: {frame_count / fps:.2f}s")
