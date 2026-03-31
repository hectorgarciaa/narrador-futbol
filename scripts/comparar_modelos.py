"""
Comparativa extensiva de modelos YOLO para detección de fútbol.

Métricas calculadas:
 - Detección: mAP50, mAP50-95, Precisión, Recall (IoU estricto, para referencia)
 - Detección suave (center-based): hits cuando el centro del GT cae dentro del bbox predicho
 - Recall suave: detecciones por proximidad de centro (sin IoU estricto)
 - Recall a IoU bajo (0.1): cuánto "identifica la presencia" aunque la caja baile
 - Precision-Recall curves (AUC)
 - Por clase: distribución de confianzas, tamaños de bbox
 - Falsos positivos y negativos por clase
 - Estadísticos de tamaño de bbox predicho vs groundtruth
"""

import sys
import json
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import cv2

# ─── Configuración ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "best_actual": PROJECT_ROOT / "models/finetuning/best.pt",
    "dfl_bundesliga": PROJECT_ROOT / "models/finetuning/con_arbitro/dfl-bundesliga/weights/best.pt",
}

# Dataset de evaluación original con 3 clases (ball=0, player=1, ref=2)
DATASET_3CL = PROJECT_ROOT / "data/detection/DFL-Bundesliga,-soccer,-football-1"
DATASET_2CL = PROJECT_ROOT / "data/detection/DFL-Bundesliga-2clases"

CONF_THRESHOLD = 0.10     # conf mínima para contar una predicción
IOU_THRESHOLD = 0.50      # IoU para mAP estricto
IOU_SOFT = 0.10           # IoU mínimo para "detección laxa"
CENTER_MATCH = True       # Activar centro-en-bbox matching
MAX_IMAGES = None         # None = todos; poner ej. 100 para prueba rápida

OUTPUT_DIR = PROJECT_ROOT / "output/comparativa_modelos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Utilidades ──────────────────────────────────────────────────────────────

def iou(box_a, box_b):
    """Calcula IoU entre dos cajas [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    union = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter_area
    return inter_area / union if union > 0 else 0.0

def center_in_box(cx, cy, box):
    """¿Está el centro (cx,cy) dentro de la caja box=[x1,y1,x2,y2]?"""
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2

def xywh_to_xyxy(xywh, img_w, img_h):
    x_c, y_c, w, h = xywh
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]

def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2-x1) * max(0, y2-y1)

# ─── Carga de GTs desde dataset YOLO ─────────────────────────────────────────

def load_gt(dataset_dir, split="valid"):
    """
    Carga groundtruth de un dataset YOLO.
    Retorna dict image_stem -> list of {class_id, box:[x1,y1,x2,y2]}
    """
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    gts = {}
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if MAX_IMAGES:
        image_paths = image_paths[:MAX_IMAGES]
    for img_path in image_paths:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                box = xywh_to_xyxy([float(p) for p in parts[1:5]], w, h)
                boxes.append({"class_id": cls_id, "box": box})
        gts[img_path.stem] = {"boxes": boxes, "shape": (w, h), "img_path": str(img_path)}
    print(f"  Cargadas {len(gts)} imágenes GT desde {dataset_dir.name}/{split}")
    return gts

# ─── Predicciones del modelo ──────────────────────────────────────────────────

def get_predictions(model_obj, gts, conf=CONF_THRESHOLD):
    """
    Corre el modelo sobre las imágenes. Retorna dict image_stem -> list of {class_id, box, confidence}
    """
    preds = {}
    img_paths = [gt_info["img_path"] for gt_info in gts.values()]
    stems = list(gts.keys())
    
    for i, (stem, gt_info) in enumerate(tqdm(gts.items(), desc="Prediciendo", ncols=80)):
        img_path = gt_info["img_path"]
        results = model_obj.predict(img_path, conf=conf, verbose=False)
        r = results[0]
        img_preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            for j in range(len(r.boxes)):
                box = r.boxes.xyxy[j].cpu().numpy().tolist()
                cls_id = int(r.boxes.cls[j].cpu().numpy())
                conf_val = float(r.boxes.conf[j].cpu().numpy())
                img_preds.append({"class_id": cls_id, "box": box, "confidence": conf_val})
        preds[stem] = img_preds
    return preds

# ─── Evaluación por imagen ────────────────────────────────────────────────────

def match_boxes(gt_boxes, pred_boxes, iou_thresh, class_map=None):
    """
    Matching greedy entre GT y predicciones para una sola imagen.
    class_map: dict {pred_class_id -> gt_class_id} para re-mapeo de clases.
    Retorna: tp, fp, fn por clase_gt, y lista de ious matched.
    """
    if class_map:
        # Re-mapeamos las predicciones al espacio de clases GT
        pred_boxes = [
            {**p, "class_id": class_map.get(p["class_id"], p["class_id"])}
            for p in pred_boxes
        ]
    
    # Agrupamos por clase
    all_classes = set(g["class_id"] for g in gt_boxes) | set(p["class_id"] for p in pred_boxes)
    results = {cls: {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []} for cls in all_classes}
    
    gt_matched = set()
    pred_matched = set()
    
    # Para cada pred, buscamos el mejor GT de la misma clase
    for pi, pred in enumerate(pred_boxes):
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in gt_matched:
                continue
            if gt["class_id"] != pred["class_id"]:
                continue
            iou_val = iou(pred["box"], gt["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gi = gi
        
        cls = pred["class_id"]
        if cls not in results:
            results[cls] = {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
        
        if best_gi >= 0 and best_iou >= iou_thresh:
            results[cls]["tp"] += 1
            results[cls]["matched_ious"].append(best_iou)
            gt_matched.add(best_gi)
            pred_matched.add(pi)
        else:
            results[cls]["fp"] += 1
    
    # FN: GTs no matcheados
    for gi, gt in enumerate(gt_boxes):
        if gi not in gt_matched:
            cls = gt["class_id"]
            if cls not in results:
                results[cls] = {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
            results[cls]["fn"] += 1
    
    return results

def center_based_recall(gt_boxes, pred_boxes, class_map=None):
    """
    Para cada GT, comprueba si alguna predicción de la misma clase contiene su centro.
    Más laxo que IoU: premia al modelo que "sabe que hay algo ahí" aunque la caja no encaje.
    """
    if class_map:
        pred_boxes = [
            {**p, "class_id": class_map.get(p["class_id"], p["class_id"])}
            for p in pred_boxes
        ]
    hits = {}   # gt_class_id -> {found, total}
    for gt in gt_boxes:
        cls = gt["class_id"]
        if cls not in hits:
            hits[cls] = {"found": 0, "total": 0}
        hits[cls]["total"] += 1
        cx, cy = center(gt["box"])
        for pred in pred_boxes:
            if pred["class_id"] == cls:
                if center_in_box(cx, cy, pred["box"]):
                    hits[cls]["found"] += 1
                    break
    return hits

# ─── Agregación de métricas ───────────────────────────────────────────────────

def aggregate_metrics(all_results_strict, all_results_soft, all_center_hits, 
                      all_preds_by_cls, all_gts_by_cls, class_names):
    """Agrega por clase los tp/fp/fn de todas las imágenes."""
    summary = {}
    all_cls = sorted(set(
        list(all_results_strict.keys()) + 
        list(all_gts_by_cls.keys())
    ))
    
    for cls in all_cls:
        cls_name = class_names.get(cls, f"class_{cls}")
        
        # Métricas estrictas (IoU 0.5)
        tp = all_results_strict.get(cls, {}).get("tp", 0)
        fp = all_results_strict.get(cls, {}).get("fp", 0)
        fn = all_results_strict.get(cls, {}).get("fn", 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Métricas suaves (IoU 0.1)
        tp_s = all_results_soft.get(cls, {}).get("tp", 0)
        fp_s = all_results_soft.get(cls, {}).get("fp", 0)
        fn_s = all_results_soft.get(cls, {}).get("fn", 0)
        recall_soft = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
        precision_soft = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
        f1_soft = 2 * precision_soft * recall_soft / (precision_soft + recall_soft) if (precision_soft + recall_soft) > 0 else 0.0
        
        # Center-based recall
        ch = all_center_hits.get(cls, {"found": 0, "total": 0})
        center_recall = ch["found"] / ch["total"] if ch["total"] > 0 else 0.0
        
        # Confianzas
        confs = all_preds_by_cls.get(cls, [])
        mean_conf = float(np.mean(confs)) if confs else 0.0
        std_conf = float(np.std(confs)) if confs else 0.0
        
        total_gt = all_gts_by_cls.get(cls, 0)
        total_pred = len(confs)
        
        # IoUs matched
        matched_ious = all_results_strict.get(cls, {}).get("matched_ious", [])
        mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        
        summary[cls_name] = {
            "class_id": cls,
            "total_gt_instances": total_gt,
            "total_pred_instances": total_pred,
            "detection_rate": total_pred / total_gt if total_gt > 0 else 0.0,  # predicciones / GTs

            # IoU estricto (0.5)
            "tp_strict": tp,
            "fp_strict": fp,
            "fn_strict": fn,
            "precision_iou50": round(precision, 4),
            "recall_iou50": round(recall, 4),
            "f1_iou50": round(f1, 4),

            # IoU laxo (0.1)
            "tp_soft": tp_s,
            "fn_soft": fn_s,
            "recall_iou10": round(recall_soft, 4),
            "precision_iou10": round(precision_soft, 4),
            "f1_iou10": round(f1_soft, 4),

            # Center-based
            "center_recall": round(center_recall, 4),

            # Confianza
            "mean_confidence": round(mean_conf, 4),
            "std_confidence": round(std_conf, 4),
            
            # IoU de hits reales
            "mean_matched_iou": round(mean_matched_iou, 4),
        }
    return summary

# ─── Evaluación de un modelo ──────────────────────────────────────────────────

def evaluate_model(model_name, model_path, dataset_dir, gt_class_names, model_class_names,
                   class_map=None):
    """
    Evalúa un modelo completo sobre un dataset.
    class_map: {model_class_id -> gt_class_id} si las clases no coinciden.
    """
    print(f"\n{'='*60}")
    print(f"Evaluando: {model_name}")
    print(f"Modelo:   {model_path}")
    print(f"Dataset:  {dataset_dir.name}")
    print(f"Clases GT:    {gt_class_names}")
    print(f"Clases Modelo: {model_class_names}")
    if class_map:
        mapped = {model_class_names.get(k, k): gt_class_names.get(v, v) for k, v in class_map.items()}
        print(f"Re-mapeo de clases: {mapped}")
    print(f"{'='*60}")

    model_obj = YOLO(str(model_path))
    gts = load_gt(dataset_dir)

    all_results_strict = {}  # cls -> {tp, fp, fn, matched_ious} acumulado
    all_results_soft = {}
    all_center_hits = {}     # cls -> {found, total}
    all_preds_by_cls = {}    # cls -> [conf, ...]
    all_gts_by_cls = {}      # cls -> total count
    
    preds = get_predictions(model_obj, gts)
    
    for stem, gt_info in tqdm(gts.items(), desc="Evaluando", ncols=80):
        gt_boxes = gt_info["boxes"]
        pred_boxes = preds.get(stem, [])
        
        # Acumular GT
        for g in gt_boxes:
            all_gts_by_cls[g["class_id"]] = all_gts_by_cls.get(g["class_id"], 0) + 1
        
        # Acumular preds por clase (con re-mapeo)
        for p in pred_boxes:
            pred_cls = class_map.get(p["class_id"], p["class_id"]) if class_map else p["class_id"]
            if pred_cls not in all_preds_by_cls:
                all_preds_by_cls[pred_cls] = []
            all_preds_by_cls[pred_cls].append(p["confidence"])
        
        # Métricas estrictas (IoU 0.5)
        res_strict = match_boxes(gt_boxes, pred_boxes, IOU_THRESHOLD, class_map)
        for cls, vals in res_strict.items():
            if cls not in all_results_strict:
                all_results_strict[cls] = {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
            all_results_strict[cls]["tp"] += vals["tp"]
            all_results_strict[cls]["fp"] += vals["fp"]
            all_results_strict[cls]["fn"] += vals["fn"]
            all_results_strict[cls]["matched_ious"].extend(vals["matched_ious"])
        
        # Métricas suaves (IoU 0.1)
        res_soft = match_boxes(gt_boxes, pred_boxes, IOU_SOFT, class_map)
        for cls, vals in res_soft.items():
            if cls not in all_results_soft:
                all_results_soft[cls] = {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}
            all_results_soft[cls]["tp"] += vals["tp"]
            all_results_soft[cls]["fp"] += vals["fp"]
            all_results_soft[cls]["fn"] += vals["fn"]
        
        # Center-based
        ch = center_based_recall(gt_boxes, pred_boxes, class_map)
        for cls, vals in ch.items():
            if cls not in all_center_hits:
                all_center_hits[cls] = {"found": 0, "total": 0}
            all_center_hits[cls]["found"] += vals["found"]
            all_center_hits[cls]["total"] += vals["total"]
    
    gt_class_names_inv = gt_class_names  # {id -> name}
    summary = aggregate_metrics(
        all_results_strict, all_results_soft, all_center_hits,
        all_preds_by_cls, all_gts_by_cls, gt_class_names
    )
    
    return summary

# ─── Impresión de resultados ──────────────────────────────────────────────────

def print_comparison(results_a, name_a, results_b, name_b):
    """Imprime una tabla comparativa clara."""
    
    all_classes = sorted(set(list(results_a.keys()) + list(results_b.keys())))
    
    metrics_to_compare = [
        ("recall_iou50",     "Recall IoU≥0.50  (estricto)"),
        ("precision_iou50",  "Precision IoU≥0.50"),
        ("f1_iou50",         "F1 IoU≥0.50"),
        ("recall_iou10",     "Recall IoU≥0.10  (laxo, presenza)"),
        ("precision_iou10",  "Precision IoU≥0.10"),
        ("f1_iou10",         "F1 IoU≥0.10"),
        ("center_recall",    "Recall Centro-en-BBox (muy laxo)"),
        ("detection_rate",   "Tasa Detección (preds/GTs)"),
        ("mean_confidence",  "Confianza Media (preds)"),
        ("mean_matched_iou", "IoU Medio en hits"),
    ]

    w = 32
    
    print("\n" + "="*90)
    print(f"{'COMPARATIVA DE MODELOS':^90}")
    print("="*90)
    print(f"  {'Modelo A':>15}: {name_a}")
    print(f"  {'Modelo B':>15}: {name_b}")
    
    for cls_name in all_classes:
        print(f"\n  ── Clase: {cls_name.upper()} ──")
        a = results_a.get(cls_name, {})
        b = results_b.get(cls_name, {})
        
        print(f"  {'Instancias GT':30} A={a.get('total_gt_instances','N/A')}   B={b.get('total_gt_instances','N/A')}")
        print(f"  {'Predicciones':30} A={a.get('total_pred_instances','N/A')}   B={b.get('total_pred_instances','N/A')}")
        print()
        
        COL = 14
        print(f"  {'Métrica':<{w}} {'':>2} {name_a[:COL]:>{COL}} {name_b[:COL]:>{COL}}  {'Diff (B-A)':>10}")
        print(f"  {'-'*w} {'':>2} {'-'*COL} {'-'*COL}  {'-'*10}")
        
        for key, label in metrics_to_compare:
            val_a = a.get(key, None)
            val_b = b.get(key, None)
            if val_a is None and val_b is None:
                continue
            
            str_a = f"{val_a:.4f}" if isinstance(val_a, float) else str(val_a)
            str_b = f"{val_b:.4f}" if isinstance(val_b, float) else str(val_b)
            
            if isinstance(val_a, float) and isinstance(val_b, float):
                diff = val_b - val_a
                sign = "▲" if diff > 0.005 else ("▼" if diff < -0.005 else "≈")
                str_diff = f"{sign} {diff:+.4f}"
                winner = f"  ← B MEJOR" if diff > 0.01 else ("  ← A MEJOR" if diff < -0.01 else "")
            else:
                str_diff = "N/A"
                winner = ""
            
            print(f"  {label:<{w}} {str_a:>{COL}} {str_b:>{COL}}  {str_diff:>10}{winner}")
    
    print("\n" + "="*90)
    print("LEYENDA:")
    print("  Recall IoU≥0.50: métrica estándar (bbox tiene que solapar 50%+ con el GT)")
    print("  Recall IoU≥0.10: ¿detecta que 'hay algo' en esa zona? (caja puede bailar)")
    print("  Recall Centro-en-BBox: el centro del GT cae dentro de la predicción (extremadamente laxo)")
    print("  Tasa Detección: cuántas predicciones hace respecto a los GTs (>1 = over-detecting, <1 = miss)")
    print("="*90)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Clases del dataset GT con 3 clases: 0=ball, 1=player, 2=ref
    gt_classes = {0: "ball", 1: "player", 2: "ref"}
    
    print("\n>>> Inspeccionando clases de los modelos...")
    results_all = {}
    model_classes_dict = {}
    
    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"⚠️  Modelo no encontrado: {model_path}")
            continue
        m = YOLO(str(model_path))
        model_cls = m.names  # dict {id: name}
        model_classes_dict[model_name] = model_cls
        print(f"  {model_name}: {model_cls}")
    
    print()
    
    # ── Evaluamos best_actual ──────────────────────────────────────────────────
    m_name = "best_actual"
    m_path = MODELS[m_name]
    if m_path.exists():
        m_cls = model_classes_dict[m_name]
        # Primero construimos el mapeo: si el modelo tiene "ball", "player", "ref"
        # coincide 1:1 con el GT. Si tiene solo 2 clases necesitamos ver.
        # Haremos el mapeo por nombre de clase.
        gt_name_to_id = {v: k for k, v in gt_classes.items()}
        
        class_map_a = {}
        for pred_id, pred_name in m_cls.items():
            clean_name = pred_name.lower().strip()
            # Intentamos mapear: referee/ref/arbitro -> 2 en GT
            if clean_name in ("ref", "referee", "arbitro", "árbitro"):
                clean_name = "ref"
            if clean_name in gt_name_to_id:
                class_map_a[pred_id] = gt_name_to_id[clean_name]
        
        print(f"Mapeo de clases para {m_name}: {class_map_a}")
        results_all[m_name] = evaluate_model(
            m_name, m_path, DATASET_3CL, gt_classes, m_cls, class_map=class_map_a
        )
    
    # ── Evaluamos dfl_bundesliga ───────────────────────────────────────────────
    m_name = "dfl_bundesliga"
    m_path = MODELS[m_name]
    if m_path.exists():
        m_cls = model_classes_dict[m_name]
        gt_name_to_id = {v: k for k, v in gt_classes.items()}
        
        class_map_b = {}
        for pred_id, pred_name in m_cls.items():
            clean_name = pred_name.lower().strip()
            if clean_name in ("ref", "referee", "arbitro", "árbitro"):
                clean_name = "ref"
            if clean_name in gt_name_to_id:
                class_map_b[pred_id] = gt_name_to_id[clean_name]
        
        print(f"Mapeo de clases para {m_name}: {class_map_b}")
        results_all[m_name] = evaluate_model(
            m_name, m_path, DATASET_3CL, gt_classes, m_cls, class_map=class_map_b
        )
    
    # ── Mostrar comparativa ────────────────────────────────────────────────────
    if "best_actual" in results_all and "dfl_bundesliga" in results_all:
        print_comparison(
            results_all["best_actual"],   "best_actual",
            results_all["dfl_bundesliga"], "dfl_bundesliga"
        )
    
    # ── Guardar JSON ───────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_classes": {k: dict(v) for k, v in model_classes_dict.items()},
            "results": {k: {c: dict(v) for c, v in r.items()} for k, r in results_all.items()}
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Resultados guardados en: {out_path}")
