import os
from ultralytics import YOLO

from football_ai.core import get_config


def download_yolo_models(base_dir, versions):
    """Descarga y guarda modelos YOLO según la configuración."""
    for version, model_names in versions.items():
        for nombre in model_names:
            modelo = YOLO(nombre)
            archivo = f"{nombre}.pt"
            ruta_guardado = os.path.join(base_dir, version, archivo)
            os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
            modelo.save(ruta_guardado)
            print(f"{nombre} guardado en {ruta_guardado}")


if __name__ == "__main__":
    config = get_config()
    base_dir = str(config.get_path('paths', 'models_download', 'base_dir', create_if_missing=True))
    versions = config.get('paths', 'models_download', 'versions')
    download_yolo_models(base_dir, versions)
