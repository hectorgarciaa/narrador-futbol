import cv2
import argparse
from pathlib import Path


def generate_output_filename(input_video: str, start_min: float, end_min: float) -> str:
    """
    Genera un nombre de archivo de salida basado en el video de entrada y los minutos.
    Si el archivo ya existe, agrega (1), (2), etc. hasta encontrar un nombre disponible.
    """
    input_path = Path(input_video)
    stem = input_path.stem
    suffix = input_path.suffix
    
    base_name = f"{stem}_{int(start_min)}-{int(end_min)}{suffix}"
    output_path = input_path.parent / base_name
    
    if not output_path.exists():
        return str(output_path)
    
    # Si existe, agregar (1), (2), etc.
    counter = 1
    while True:
        new_name = f"{base_name}({counter}){suffix}"
        output_path = input_path.parent / new_name
        if not output_path.exists():
            return str(output_path)
        counter += 1


def main():
    parser = argparse.ArgumentParser(description="Recorta un video desde un minuto de inicio a un minuto final.")
    parser.add_argument("--video", required=True, help="Ruta del video de entrada (ej: partido.mp4)")
    parser.add_argument("--s", "--start", dest="start", type=float, required=True, help="Minuto de inicio (ej: 1.5 para 1 minuto 30 segundos)")
    parser.add_argument("--e", "--end", dest="end", type=float, required=True, help="Minuto final (ej: 3.87 para 3 minutos 52 segundos)")
    parser.add_argument("--output",default=None,help="Ruta del video de salida (opcional, genera un nombre por defecto si no se especifica)")
    
    args = parser.parse_args()
    
    # Validaciones
    if args.start < 0 or args.end < 0:
        print("Error: Los minutos de inicio y final deben ser positivos")
        return
    
    if args.start >= args.end:
        print("Error: El minuto de inicio debe ser menor que el minuto final")
        return
    
    if not Path(args.video).exists():
        print(f"Error: El video '{args.video}' no existe")
        return
    
    # Abrir video y obtener propiedades
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convertir minutos a frames
    start_frame = int(args.start * 60 * fps)
    end_frame = int(args.end * 60 * fps)
    
    # Validar que los frames estén dentro del rango
    if start_frame >= total_frames:
        print(f"Error: El frame de inicio ({start_frame}) está fuera del rango del video (total: {total_frames})")
        cap.release()
        return
    
    if end_frame > total_frames:
        print(f"Advertencia: El frame final ({end_frame}) excede el total del video ({total_frames}). Ajustando...")
        end_frame = total_frames
    
    # Generar nombre de salida si no se proporciona
    output_video = args.output if args.output else generate_output_filename(args.video, args.start, args.end)
    
    print(f"Recortando video: {args.video}")
    print(f"Inicio: {args.start} minutos (frame {start_frame})")
    print(f"Final: {args.end} minutos (frame {end_frame})")
    print(f"Salida: {output_video}")
    
    # Definir el códec y crear el objeto de escritura
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Leer y guardar los frames en el rango especificado
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    # Liberar recursos
    cap.release()
    out.release()
    
    print(f"✓ Video recortado exitosamente. {frame_count} frames procesados.")


if __name__ == "__main__":
    main()
