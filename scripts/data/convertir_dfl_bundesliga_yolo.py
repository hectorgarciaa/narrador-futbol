"""
Convierte el dataset DFL-Bundesliga (formato YOLO, 3 clases: ball=0, player=1, ref=2)
a un nuevo dataset YOLO con solo 2 clases: ball=0, player=1.

Los árbitros (clase 2) se remapean a jugadores (clase 1).
Las imágenes NO se copian (se usan symlinks o rutas relativas), solo se regeneran los labels.

Uso:
    cd /ruta/al/proyecto
    python scripts/data/convertir_dfl_bundesliga_yolo.py

Salida: data/detection/DFL-Bundesliga-2clases/
"""
import os
import shutil
from pathlib import Path
import yaml

# Clases originales en DFL: 0=ball, 1=player, 2=ref
# Mapeo a las nuevas 2 clases:
#   0 (ball)   -> 0 (ball)    ← sin cambios
#   1 (player) -> 1 (player)  ← sin cambios
#   2 (ref)    -> 1 (player)  ← árbitros pasan a ser jugadores
REMAP = {0: 0, 1: 1, 2: 1}
NEW_CLASSES = ['ball', 'player']

SRC_BASE  = Path("data/detection/DFL-Bundesliga,-soccer,-football-1")
DST_BASE  = Path("data/detection/DFL-Bundesliga-2clases")

# Los splits del DFL original usan 'valid' como nombre de carpeta en validación
SPLITS = {
    "train": "train",
    "valid": "valid",   # carpeta original de DFL (si cambia, ajusta aquí)
    "test":  "test",
}


def convert_labels_split(split_src_name: str, split_dst_name: str):
    src_labels = SRC_BASE / split_src_name / "labels"
    dst_labels = DST_BASE / split_dst_name / "labels"
    src_images = SRC_BASE / split_src_name / "images"
    dst_images = DST_BASE / split_dst_name / "images"

    if not src_labels.exists():
        print(f"  ⤸ Skip '{split_src_name}': carpeta de labels no encontrada en {src_labels}")
        return

    dst_labels.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    label_files = list(src_labels.glob("*.txt"))
    print(f"  Procesando {len(label_files)} archivos de labels en '{split_src_name}'...")

    skipped = 0
    converted = 0

    for src_txt in label_files:
        lines_out = []
        with open(src_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                orig_cls = int(parts[0])
                new_cls = REMAP.get(orig_cls)
                if new_cls is None:
                    # Clase desconocida: ignorar silenciosamente
                    skipped += 1
                    continue
                lines_out.append(f"{new_cls} " + " ".join(parts[1:]))

        dst_txt = dst_labels / src_txt.name
        with open(dst_txt, "w") as f:
            f.write("\n".join(lines_out) + ("\n" if lines_out else ""))
        converted += 1

    # Enlace simbólico a las imágenes originales (evita duplicar GBs de datos)
    if src_images.exists():
        for img in src_images.iterdir():
            dst_img = dst_images / img.name
            if not dst_img.exists():
                os.symlink(img.resolve(), dst_img)
    
    print(f"    ✓ {converted} labels convertidos, {skipped} anotaciones desconocidas ignoradas.")


def write_data_yaml():
    yaml_content = {
        "train": "../train/images",
        "val":   "../valid/images",
        "test":  "../test/images",
        "nc":    len(NEW_CLASSES),
        "names": NEW_CLASSES,
    }
    out_yaml = DST_BASE / "data.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml escrito en {out_yaml}")


def main():
    print(f"=== Conversión DFL-Bundesliga → 2 clases (ball, player) ===")
    DST_BASE.mkdir(parents=True, exist_ok=True)

    for src_split, dst_split in SPLITS.items():
        print(f"\n[Split: {src_split}]")
        convert_labels_split(src_split, dst_split)

    print("\n[data.yaml]")
    write_data_yaml()

    print(f"\n✅ Dataset listo en: {DST_BASE.resolve()}")
    print(f"   Úsalo en config.yaml apuntando a: {DST_BASE}/data.yaml")


if __name__ == "__main__":
    main()
