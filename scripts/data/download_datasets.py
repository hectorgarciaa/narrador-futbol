from roboflow import Roboflow
import shutil
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener API keys y configuración desde variables de entorno
API_KEY = os.getenv("ROBOFLOW_API_KEY")
publishableApiKey = os.getenv("ROBOFLOW_PUBLISHABLE_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")

# Validar que las claves estén configuradas
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY no está configurada. Crea un archivo .env basado en .env.example")
if not WORKSPACE:
    raise ValueError("ROBOFLOW_WORKSPACE no está configurado en el archivo .env")
if not PROJECT:
    raise ValueError("ROBOFLOW_PROJECT no está configurado en el archivo .env")

output_dir = "../../data/detection"

def downloadDataSet(ouput_dir):
    # Inicializa Roboflow con tu API key
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(1).download("yolov11")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.move(dataset.location, output_dir)

if __name__ == "__main__":
    downloadDataSet(output_dir)
