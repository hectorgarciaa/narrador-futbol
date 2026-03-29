import os
import json
import zipfile
import shutil
import random
import yaml
from pathlib import Path

def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convierte [x_min, y_min, width, height] (COCO)
    a [x_center, y_center, width, height] normalizado (YOLO)
    """
    x_min, y_min, w, h = bbox
    
    x_center = x_min + (w / 2.0)
    y_center = y_min + (h / 2.0)
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    
    return [x_center, y_center, w, h]

def process_soccersynth_split(split_name, base_dir, out_dir, target_class_id=1):
    """
    Procesa un split (train, val, test) de SoccerSynth.
    Extrae imágenes y genera txt en formato YOLO.
    """
    zip_path = base_dir / f"{split_name}.zip"
    annotations_zip_path = base_dir / "annotations.zip"
    
    if not zip_path.exists():
        print(f"Skipping {split_name}, image ZIP not found at {zip_path}")
        return
        
    # Directorios destino
    out_images_dir = out_dir / split_name / "images"
    out_labels_dir = out_dir / split_name / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations for {split_name}...")
    coco_data = None
    
    # Intenta leer desde el ZIP primero para entornos limpios como el clúster
    if annotations_zip_path.exists():
        with zipfile.ZipFile(annotations_zip_path, 'r') as z:
            # Buscar el JSON independientemente de la estructura interna de subdirectorios
            target_filename = f"{split_name}.json"
            json_file = next((name for name in z.namelist() if name.endswith(target_filename)), None)
            
            if json_file:
                with z.open(json_file) as f:
                    coco_data = json.load(f)
            else:
                print(f"Skipping {split_name}, {target_filename} not found inside annotations.zip")
                return
    else:
        # Fallback si ya lo tienen descomprimido localmente
        json_path = base_dir / "annotations" / "annotations" / f"{split_name}.json"
        
        # Intentar ruta alternativa sin un "annotations" extra por si acaso
        if not json_path.exists():
            json_path = base_dir / "annotations" / f"{split_name}.json"
            
        if not json_path.exists():
            print(f"Skipping {split_name}, neither annotations.zip nor plain JSON found.")
            return
            
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
    # Mapear imágenes por id
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Agrupar anotaciones por image_id
    annotations_by_img = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)
        
    print(f"Extracting images from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extraer solo archivos (ignorando la estructura interna si la hubiera)
        for member in zip_ref.namelist():
            if not member.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Extraer y mover al directorio correcto
            filename = os.path.basename(member)
            if not filename:
                continue
                
            source = zip_ref.open(member)
            target = open(out_images_dir / filename, "wb")
            with source, target:
                shutil.copyfileobj(source, target)
                
    print(f"Generating YOLO labels for {split_name}...")
    for img_id, img_info in images_info.items():
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # El nombre del txt correspondiente
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = out_labels_dir / txt_filename
        
        anns = annotations_by_img.get(img_id, [])
        
        with open(txt_filepath, 'w') as f:
            for ann in anns:
                if 'bbox' not in ann:
                    continue
                
                # En SoccerSynth person es 1, pero queremos pasarlo a target_class_id
                # En YOLO de la config default: 0 es ball, 1 es player, 2 es ref
                # Así que lo mapearemos a 1 por defecto (player)
                
                yolo_bbox = convert_coco_bbox_to_yolo(ann['bbox'], width, height)
                
                # Escribir línea: class x_center y_center width height
                line = f"{target_class_id} " + " ".join([f"{v:.6f}" for v in yolo_bbox]) + "\n"
                f.write(line)
                
def main():
    base_dir = Path("data/detection/SoccerSynth")
    out_dir = Path("data/detection/SoccerSynth_YOLO")
    
    # Crear data.yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    
    yaml_content = {
        'train': '../train/images',
        'val': '../val/images',
        'test': '../test/images',
        'nc': 3,
        'names': ['ball', 'player', 'ref']
    }
    
    # Escribir el yaml, compatible con la estructura existente
    with open(out_dir / "data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    # En la configuración original, `player` tiene el índice 1
    PLAYER_CLASS_ID = 1
    
    for split in ['train', 'val', 'test']:
        process_soccersynth_split(split, base_dir, out_dir, target_class_id=PLAYER_CLASS_ID)
        
    print(f"\nConversion finished! Dataset ready at {out_dir}")
    print(f"You can train using this dataset by pointing to {out_dir}/data.yaml")

if __name__ == "__main__":
    main()
