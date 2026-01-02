#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import re
import sys

# ================== CONFIG DO USUÁRIO ==================
name_exp = "llm_guided_yoloSAM"
PRED_DIR   = Path("/home/rodrigo/Documents/Mestrado/LLM_INFERENCE/input_llm/output_llm/input_llm_guided_yoloSAM_formatados")   # pasta com .json de predição
GT_DIR     = Path("ground_truth")     # pasta com .json de ground truth
OUTPUT_DIR = Path("saida_metricas")
RECURSIVE  = True  # procurar recursivamente por *.json
# =======================================================

# --------- Rótulos canônicos suportados ---------
CANON_LABELS = {
    "Missing teeth",
    "Present teeth",
    "Type of dentition",          # ESCALAR (0 = permanent, 1 = mixed)
    "Endodontic treatment",
    "Crown lesions",
    "Tooth with mesial inclination",
    "Presence of implants",
}

# --------- Aliases (variações com/sem “:”, PT/EN, acentos) ---------
def _mk_aliases(*variants):
    out = {}
    for v in variants:
        k = v.lower().strip().rstrip(":")
        k = re.sub(r"\s+", " ", k)
        out[k] = variants[0]
    return out

LABEL_ALIASES = {}
LABEL_ALIASES |= _mk_aliases("Missing teeth", "missing teeth", "dentes ausentes")
LABEL_ALIASES |= _mk_aliases("Present teeth", "present teeth", "dentes presentes")
LABEL_ALIASES |= _mk_aliases("Type of dentition", "type of dentition",
                             "tipo de dentição", "tipo de denticao")
LABEL_ALIASES |= _mk_aliases("Endodontic treatment", "endodontic treatment",
                             "tratamento endodôntico", "tratamento endodontico")
LABEL_ALIASES |= _mk_aliases("Crown lesions", "crown lesions",
                             "lesões de coroa", "lesoes de coroa")
LABEL_ALIASES |= _mk_aliases("Tooth with mesial inclination",
                             "tooth with mesial inclination",
                             "dente com inclinação mesial", "dente com inclinacao mesial")
LABEL_ALIASES |= _mk_aliases("Presence of implants", "presence of implants",
                             "presença de implantes", "presenca de implantes")

# --------- Tipagem dos rótulos ---------
LIST_LABELS = {
    "Missing teeth",
    "Present teeth",
    "Endodontic treatment",
    "Crown lesions",
    "Tooth with mesial inclination",
    "Presence of implants",
}
SCALAR_LABELS = {"Type of dentition"}

# --------- Normalizações ---------
# IMPORTANTE: 0 = permanent, 1 = mixed
DENTITION_MAP = {
    "0": 0, "1": 1,
    "permanent": 0, "permanente": 0,
    "mixed": 1, "mista": 1,
}

NUM_TOKEN_RE = re.compile(r"\d+")

def canon_label(raw_label: str) -> str:
    """Normaliza rótulo: remove ':', compacta espaços e aplica aliases."""
    lbl = raw_label.strip().rstrip(":")
    low = re.sub(r"\s+", " ", lbl.lower())
    return LABEL_ALIASES.get(low, lbl)

def split_label_and_value(obj_str: str):
    """Divide 'Label: values' em (label, values). Aceita sem ':'."""
    if ":" in obj_str:
        left, right = obj_str.split(":", 1)
        return left.strip(), right.strip()
    s = obj_str.strip()
    return s, ""

def parse_list_of_teeth(values_str: str) -> set[int]:
    """Extrai números de dentes (inteiros), tolera vírgulas e espaços."""
    nums = NUM_TOKEN_RE.findall(values_str.replace(",", " "))
    return {int(n) for n in nums}

def parse_scalar(label: str, values_str: str):
    """Converte valores escalares para formato esperado."""
    if label == "Type of dentition":
        token = values_str.strip().lower().replace(",", " ")
        token = re.sub(r"\s+", " ", token).strip()
        if token in DENTITION_MAP:
            return DENTITION_MAP[token]
        m = NUM_TOKEN_RE.search(values_str)
        if m:
            v = int(m.group())
            if v == 0: return 0
            if v == 1: return 1
            return v
        return None
    m = NUM_TOKEN_RE.search(values_str)
    return int(m.group()) if m else None

def parse_objects(objects_list):
    """
    Retorna dict[label] = set[int] (listas) ou int/None (escalares).
    Ignora rótulos fora da lista CANON_LABELS.
    """
    data = {}
    for item in objects_list:
        s = str(item)
        raw_label, values = split_label_and_value(s)
        label = canon_label(raw_label)
        if label not in CANON_LABELS:
            continue
        if label in SCALAR_LABELS:
            data[label] = parse_scalar(label, values)
        else:
            data[label] = parse_list_of_teeth(values)
    return data

def load_jsons(dir_path: Path, recursive: bool):
    """Lê JSONs e indexa por img_name (ou nome do arquivo)."""
    pattern = "**/*.json" if recursive else "*.json"
    idx = {}
    for jf in dir_path.glob(pattern):
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
            img_key = data.get("img_name") or jf.stem
            parsed = parse_objects(data.get("objects", []))
            idx[img_key] = parsed
        except Exception as e:
            print(f"[AVISO] Falha ao ler {jf}: {e}", file=sys.stderr)
    return idx

def compare_sets(pred_set: set[int], gt_set: set[int]):
    """Retorna métricas de conjunto + acurácia (Jaccard)."""
    inter = pred_set & gt_set
    union = pred_set | gt_set
    tp_items = sorted(inter)
    fp_items = sorted(pred_set - gt_set)
    fn_items = sorted(gt_set - pred_set)
    tp, fp, fn = len(tp_items), len(fp_items), len(fn_items)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    # Acurácia de conjunto (Jaccard/IoU). Se as duas listas forem vazias, define 1.0
    acc  = (len(inter) / len(union)) if len(union) > 0 else 1.0
    return tp, fp, fn, prec, rec, f1, acc, tp_items, fp_items, fn_items, len(inter), len(union)

def main():
    if not PRED_DIR.is_dir():
        print(f"Erro: PRED_DIR não encontrado: {PRED_DIR}", file=sys.stderr); sys.exit(1)
    if not GT_DIR.is_dir():
        print(f"Erro: GT_DIR não encontrado: {GT_DIR}", file=sys.stderr); sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_index = load_jsons(PRED_DIR, RECURSIVE)
    gt_index   = load_jsons(GT_DIR,   RECURSIVE)

    imgs = sorted(set(pred_index) & set(gt_index))
    if not imgs:
        print("[AVISO] Nenhuma imagem em comum por 'img_name'.", file=sys.stderr)

    per_image_rows = []
    # Para listas: acumulamos TP/FP/FN + soma_inter e soma_uniao para Accuracy micro (Jaccard micro)
    # Para escalar: TP = acertos; GT_SUPPORT = total avaliados (para Accuracy)
    label_counters = defaultdict(lambda: Counter(TP=0, FP=0, FN=0, GT_SUPPORT=0, INTER=0, UNION=0))
    diffs_lines = []

    for img in imgs:
        p = pred_index.get(img, {})
        g = gt_index.get(img, {})

        for label in CANON_LABELS:
            pred_val = p.get(label, None)
            gt_val   = g.get(label, None)

            if label in SCALAR_LABELS:
                acc = 1.0 if (pred_val is not None and gt_val is not None and pred_val == gt_val) else 0.0
                per_image_rows.append({
                    "img_name": img, "label": label, "type": "scalar",
                    "tp": "", "fp": "", "fn": "",
                    "precision": "", "recall": "", "f1": "",
                    "accuracy": acc,
                    "scalar_pred": pred_val, "scalar_gt": gt_val
                })
                if gt_val is not None:
                    label_counters[label]["GT_SUPPORT"] += 1
                    if acc == 1.0:
                        label_counters[label]["TP"] += 1
            else:
                pred_set = set(pred_val) if isinstance(pred_val, (set, list)) else set()
                gt_set   = set(gt_val) if isinstance(gt_val, (set, list)) else set()
                tp, fp, fn, prec, rec, f1, acc, tp_items, fp_items, fn_items, inter_n, union_n = compare_sets(pred_set, gt_set)

                per_image_rows.append({
                    "img_name": img, "label": label, "type": "list",
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
                    "accuracy": round(acc, 6),
                    "scalar_pred": "", "scalar_gt": ""
                })

                label_counters[label]["TP"] += tp
                label_counters[label]["FP"] += fp
                label_counters[label]["FN"] += fn
                label_counters[label]["GT_SUPPORT"] += len(gt_set)
                label_counters[label]["INTER"] += inter_n
                label_counters[label]["UNION"] += union_n

                if fp_items or fn_items:
                    diffs_lines.append(f"{img} | {label} | FP={fp_items} | FN={fn_items}")

    # --- per_image_label_metrics.csv ---
    per_image_csv = OUTPUT_DIR / f"per_image_label_metrics_{name_exp}.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "img_name","label","type","tp","fp","fn","precision","recall","f1","accuracy",
            "scalar_pred","scalar_gt"
        ])
        writer.writeheader()
        writer.writerows(per_image_rows)

    # --- summary_by_label.csv ---
    summary_csv = OUTPUT_DIR / f"summary_by_label_{name_exp}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label","type","TP","FP","FN","Precision","Recall","F1","Accuracy","GT_SUPPORT","Extra"
        ])
        writer.writeheader()
        for label in sorted(CANON_LABELS):
            cnt = label_counters[label]
            if label in SCALAR_LABELS:
                total = cnt["GT_SUPPORT"]
                acc   = (cnt["TP"]/total) if total > 0 else 0.0
                writer.writerow({
                    "label": label, "type": "scalar",
                    "TP": cnt["TP"], "FP": "", "FN": "",
                    "Precision": "", "Recall": "", "F1": "",
                    "Accuracy": f"{acc:.6f}",
                    "GT_SUPPORT": total, "Extra": "0=permanent, 1=mixed"
                })
            else:
                TP, FP, FN = cnt["TP"], cnt["FP"], cnt["FN"]
                prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
                rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
                f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
                # Accuracy (micro Jaccard): soma_inter / soma_uniao (se união=0, define 1.0)
                inter_sum, union_sum = cnt["INTER"], cnt["UNION"]
                acc = (inter_sum/union_sum) if union_sum > 0 else 1.0
                writer.writerow({
                    "label": label, "type": "list",
                    "TP": TP, "FP": FP, "FN": FN,
                    "Precision": f"{prec:.6f}", "Recall": f"{rec:.6f}", "F1": f"{f1:.6f}",
                    "Accuracy": f"{acc:.6f}",
                    "GT_SUPPORT": cnt["GT_SUPPORT"], "Extra": ""
                })

    # --- diffs_examples.txt ---
    diffs_txt = OUTPUT_DIR / "diffs_examples.txt"
    with diffs_txt.open("w", encoding="utf-8") as f:
        f.write("# Discrepâncias por imagem/label (FP=previstos a mais; FN=itens faltantes)\n")
        for line in diffs_lines[:5000]:
            f.write(line + "\n")

    print(f"[OK] Imagens comparadas: {len(imgs)}")
    print(f"[OK] Salvo: {per_image_csv}")
    print(f"[OK] Salvo: {summary_csv}")
    print(f"[OK] Salvo: {diffs_txt}")

if __name__ == "__main__":
    main()