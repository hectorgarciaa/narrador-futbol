"""
Convierte el dataset SoccerSynth (formato COCO, clase person) a un dataset YOLO
con solo 2 clases: ball=0, player=1 (sin clase 'ref').

Muy similar a convertir_soccersynth_yolo.py, pero el data.yaml resultante
solo declara 2 clases en lugar de 3.

Uso:
    cd /ruta/al/proyecto
    python scripts/data/convertir_soccersynth_2clases.py

Salida: data/detection/SoccerSynth_YOLO_2clases/
"""
import os
import json
import zipfile
import shutil
from pathlib import Path
import yaml


def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convierte [x_min, y_min, width, height] (COCO)
    a [x_center, y_center, width, height] normalizado (YOLO).
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_width
    y_center  = (y_min + h / 2.0) / img_height
    w  /= img_width
    h  /= img_height
    return [x_center, y_center, w, h]


def process_split(split_name: str, base_dir: Path, out_dir: Path, player_class_id: int = 1):
    """
    Procesa un split (train, val, test) de SoccerSynth.
    Todas las personas se convierten a player_class_id (por defecto 1).
    """
    zip_path             = base_dir / f"{split_name}.zip"
    annotations_zip_path = base_dir / "annotations.zip"

    if not zip_path.exists():
        print(f"  ⤸ Skip '{split_name}': ZIP de imágenes no encontrado en {zip_path}")
        return

    out_images_dir = out_dir / split_name / "images"
    out_labels_dir = out_dir / split_name / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar anotaciones COCO ---
    print(f"  Cargando anotaciones para '{split_name}'...")
    coco_data = None

    if annotations_zip_path.exists():
        with zipfile.ZipFile(annotations_zip_path, "r") as z:
            target_filename = f"{split_name}.json"
            json_file = next(
                (name for name in z.namelist() if name.endswith(target_filename)), None
            )
            if json_file:
                with z.open(json_file) as f:
                    coco_data = json.load(f)
            else:
                print(f"    ✗ {target_filename} no encontrado en annotations.zip. Skip.")
                return
    else:
        # Fallback: JSON descomprimido localmente
        for candidate in [
            base_dir / "annotations" / "annotations" / f"{split_name}.json",
            base_dir / "annotations" / f"{split_name}.json",
        ]:
            if candidate.exists():
                with open(candidate, "r") as f:
                    coco_data = json.load(f)
                break
        if coco_data is None:
            print(f"    ✗ No se encontró anotación para '{split_name}'. Skip.")
            return

    images_info = {img["id"]: img for img in coco_data["images"]}

    annotations_by_img: dict[int, list] = {}
    for ann in coco_data["annotations"]:
        annotations_by_img.setdefault(ann["image_id"], []).append(ann)

    # --- Extraer imágenes ---
    print(f"  Extrayendo imágenes de {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            filename = os.path.basename(member)
            if not filename:
                continue
            dst = out_images_dir / filename
            if not dst.exists():
                with zf.open(member) as src, open(dst, "wb") as tgt:
                    shutil.copyfileobj(src, tgt)

    # --- Generar labels YOLO ---
    print(f"  Generando labels YOLO para '{split_name}'...")
    for img_id, img_info in images_info.items():
        filename = img_info["file_name"]
        width    = img_info["width"]
        height   = img_info["height"]
        txt_path = out_labels_dir / (os.path.splitext(filename)[0] + ".txt")

        anns = annotations_by_img.get(img_id, [])
        with open(txt_path, "w") as f:
            for ann in anns:
                if "bbox" not in ann:
                    continue
                yolo_bbox = convert_coco_bbox_to_yolo(ann["bbox"], width, height)
                # Todas las personas → clase player (índice player_class_id)
                f.write(f"{player_class_id} " + " ".join(f"{v:.6f}" for v in yolo_bbox) + "\n")

    print(f"    ✓ Split '{split_name}' completado.")


def main():
    base_dir = Path("data/detection/SoccerSynth")
    out_dir  = Path("data/detection/SoccerSynth_YOLO_2clases")

    print("=== Conversión SoccerSynth → YOLO 2 clases (ball, player) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # data.yaml con SOLO 2 clases (ball=0, player=1)
    yaml_content = {
        "train": "../train/images",
        "val":   "../val/images",
        "test":  "../test/images",
        "nc":    2,
        "names": ["ball", "player"],
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml escrito en {out_dir / 'data.yaml'}")

    # Índice de player en el esquema nuevo de 2 clases: 0=ball, 1=player
    PLAYER_CLASS_ID = 1

    for split in ["train", "val", "test"]:
        print(f"\n[Split: {split}]")
        process_split(split, base_dir, out_dir, player_class_id=PLAYER_CLASS_ID)

    print(f"\n✅ Dataset listo en: {out_dir.resolve()}")
    print(f"   Úsalo en config.yaml apuntando a: {out_dir}/data.yaml")


if __name__ == "__main__":
    main()
