#!/usr/bin/env python3
"""
Script para inspeccionar el espacio de color y características de codificación del video.
Requiere: ffprobe (viene con ffmpeg)
"""

import subprocess
import json
import sys

def inspect_video_color_space(video_path):
    """Inspecciona el espacio de color del video usando ffprobe."""
    
    try:
        # Ejecutar ffprobe para obtener información detallada
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=color_space,color_range,color_transfer,color_primaries,codec_name,codec_long_name,width,height,r_frame_rate',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        if not data.get('streams'):
            print("Error: No se encontró stream de video")
            return None
        
        stream = data['streams'][0]
        return stream
    
    except FileNotFoundError:
        print("Error: ffprobe no está instalado.")
        print("En Linux: sudo apt install ffmpeg")
        return None
    except Exception as e:
        print(f"Error al inspeccionar video: {e}")
        return None


def main():
    video_path = "AD Ferroviaria-Canillejas CF_clip_30s.mp4"
    
    print(f"Inspeccionando: {video_path}")
    print("=" * 70)
    
    info = inspect_video_color_space(video_path)
    
    if info is None:
        sys.exit(1)
    
    print(f"Códec: {info.get('codec_long_name', 'N/A')}")
    print(f"Resolución: {info.get('width', 'N/A')} x {info.get('height', 'N/A')}")
    
    fps_str = info.get('r_frame_rate', 'N/A')
    if fps_str != 'N/A':
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
        print(f"FPS: {fps:.2f}")
    
    print("\n*** ESPACIO DE COLOR ***")
    color_space = info.get('color_space', 'desconocido')
    print(f"Color Space: {color_space}")
    
    # Interpretar el espacio de color
    if color_space:
        if color_space.lower() == 'yuv420p' or color_space.lower() == 'yuv':
            print("  → YUV (estándar para vídeo comprimido)")
            print("  → OpenCV lo convierte a BGR automáticamente")
            print("  → YOLO necesita RGB, hay conversión BGR→RGB")
        elif color_space.lower() == 'rgb':
            print("  → RGB (óptimo para YOLO)")
        elif color_space.lower() == 'bgr':
            print("  → BGR (formato nativo de OpenCV)")
            print("  → YOLO necesita RGB, hay conversión BGR→RGB")
    
    print(f"\nColor Range: {info.get('color_range', 'N/A')}")
    print(f"Color Transfer: {info.get('color_transfer', 'N/A')}")
    print(f"Color Primaries: {info.get('color_primaries', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("\nRECOMENDACIONES:")
    print("- YOLO típicamente espera imágenes RGB en rango [0, 255]")
    print("- OpenCV carga en BGR y hace conversión automática")
    print("- Los videos MP4 suelen estar en YUV420p (compresión estándar)")
    print("- Esto es normal y YOLO está preparado para ello")
    print("- El verdadero problema sería si hay incoherencia en el entramiento")


if __name__ == "__main__":
    main()
