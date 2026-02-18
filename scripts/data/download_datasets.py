from roboflow import Roboflow
import shutil
import os
from dotenv import load_dotenv

from football_ai.core import get_config

# Load environment variables from .env
load_dotenv()

# Get API keys and configuration from environment variables
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")

# Validate that the keys are configured
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY is not configured. Create a .env file based on .env.example")
if not WORKSPACE:
    raise ValueError("ROBOFLOW_WORKSPACE is not configured in the .env file")
if not PROJECT:
    raise ValueError("ROBOFLOW_PROJECT is not configured in the .env file")


def download_dataset(output_dir):
    """Downloads the dataset from Roboflow and moves it to the output directory."""
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(1).download("yolov11")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shutil.move(dataset.location, output_dir)


if __name__ == "__main__":
    config = get_config()
    output_dir = str(config.get_path('paths', 'data', 'download_detection_output', create_if_missing=True))
    download_dataset(output_dir)
