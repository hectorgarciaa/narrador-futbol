from roboflow import Roboflow
import shutil
import os

API_KEY = "bypTTZl8JCEAKE6N4DDN"
publishableApiKey = "rf_pi97jJ4I4Ng35oEBDuSqxAPpRVb2"

output_dir = "../../data/detection"

def downloadDataSet(ouput_dir):
    # Inicializa Roboflow con tu API key
    rf = Roboflow(api_key="KSNZNjhrcZN1cq1zmo33")
    project = rf.workspace("yolo-atnlh").project("football-detection-wfhdh")
    dataset = project.version(2).download("yolov11")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    shutil.move(dataset.location, output_dir)

if __name__ == "__main__":
    downloadDataSet(output_dir)