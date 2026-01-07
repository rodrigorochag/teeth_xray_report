import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set

FDI_TEETH = set()
for quadrant in [1, 2, 3, 4]:
    for position in range(1, 9):
        FDI_TEETH.add(f"{quadrant}{position}")

def _to_set(xs: List) -> Set[str]:
    return {str(x) for x in xs}

def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def parse_json(json_path: Path, classes: List):
    try:
        elapsed = None
        with open(json_path, "r") as f:
            data = json.load(f)
        if "objects" in data:
            data = data["objects"]
        if "elapsed_ms" in data:
            elapsed = data["elapsed_ms"]
        if "pred" in data:
            data = data["pred"]
    except Exception:
        return {}, None
    
    if type(data) is dict:
        results = {tcls:[str(n) for n in data.get(tcls, [])] for tcls in classes}
    elif type(data) is list:
        results = {}
        for line in data:
            for tcls in classes:
                if line.startswith(f"{tcls}:"):
                    nums = re.findall(r"\d+", line)
                    results[tcls] = [str(n) for n in nums]
    else:
        results = {}
    
    if "Missing teeth" in classes and "Present teeth" in classes and not results.get("Present teeth"):
        missing = set(results.get("Missing teeth", []))
        results["Present teeth"] = sorted([t for t in FDI_TEETH if t not in missing])
        
    return results, elapsed

def evaluate_predictions(gt_path: Path, pred_path: Path, classes: List):
    gt_files = set(os.listdir(gt_path))
    pred_files = set(os.listdir(pred_path))
    imgs_names = sorted(gt_files & pred_files)

    print(f"We are going to evaluate results based on {len(imgs_names)} reports.") 
    
    # stats[class] = dict(tp=..., fp=..., fn=...)
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    # for img, gt_labels in gt.items():
    times_ms = []
    for img_name in imgs_names:
        gt_labels, _ = parse_json(gt_path / img_name, classes)
        pred_labels, elapsed_ms = parse_json(pred_path / img_name, classes)
        if elapsed_ms:
            times_ms.append(elapsed_ms)

        # accumulate TP, FP, FN
        for tcls in classes:
            gt_set = _to_set(gt_labels.get(tcls, []))
            pred_set = _to_set(pred_labels.get(tcls, []))

            stats[tcls]["tp"] += len(gt_set & pred_set)
            stats[tcls]["fp"] += len(pred_set - gt_set)
            stats[tcls]["fn"] += len(gt_set - pred_set)

    results = {}
    # per-class metrics
    for tcls, s in stats.items():
        p, r = prf(s["tp"], s["fp"], s["fn"])
        results[tcls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
        }

    # Missing + Present
    if "Missing teeth" in classes and "Present teeth" in classes:
        mp = {"Missing teeth", "Present teeth"}
        tp = sum(stats[c]["tp"] for c in mp)
        fp = sum(stats[c]["fp"] for c in mp)
        fn = sum(stats[c]["fn"] for c in mp)

        p_mp, r_mp = prf(tp, fp, fn)

        results["Both MP"] = {
            "precision": round(p_mp, 4),
            "recall": round(r_mp, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    results["elapsed_mean_ms"] = round(sum(times_ms) / len(times_ms), 2) if times_ms else None

    return results