from roboflow import Roboflow
import shutil
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener API keys desde variables de entorno
API_KEY = os.getenv("ROBOFLOW_API_KEY")
publishableApiKey = os.getenv("ROBOFLOW_PUBLISHABLE_KEY")

# Validar que las claves estén configuradas
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY no está configurada. Crea un archivo .env basado en .env.example")

output_dir = "../../data/detection"

def downloadDataSet(ouput_dir):
    # Inicializa Roboflow con tu API key
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("work-ejvtm").project("football-ai-ikmty")
    dataset = project.version(2).download("yolov11")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.move(dataset.location, output_dir)

if __name__ == "__main__":
    downloadDataSet(output_dir)
