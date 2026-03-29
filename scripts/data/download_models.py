import os
from ultralytics import YOLO

from football_ai.core import get_config

import shutil

def download_yolo_models(base_dir, versions):
    """Downloads and saves YOLO models according to the configuration."""
    for version, model_names in versions.items():
        for name in model_names:
            filename = f"{name}.pt"
            save_path = os.path.join(base_dir, version, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if not os.path.exists(save_path):
                # YOLO por defecto descarga el archivo en el directorio actual (la raíz)
                model = YOLO(filename)
                
                # Movemos el archivo descargado desde la raíz a la carpeta correcta
                if os.path.exists(filename):
                    shutil.move(filename, save_path)
                    print(f"Moved {filename} to {save_path}")
            else:
                print(f"Model {name} already exists at {save_path}")


if __name__ == "__main__":
    config = get_config()
    base_dir = str(config.get_path('paths', 'models_download', 'base_dir', create_if_missing=True))
    versions = config.get('paths', 'models_download', 'versions')
    download_yolo_models(base_dir, versions)
