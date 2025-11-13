import json
from evaluators import Evaluator

archivos = ["tracks_m_vmedio.json", "tracks_x_vmedio.json", "tracks_m.json", "tracks_x.json"]
for file in archivos:
    with open("./experimentos/" + file, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluator = Evaluator()
    metrics, summary = evaluator.evaluate(data)
    print(file)
    print(summary)
    print("="*60)
    print
    print()
    print()
