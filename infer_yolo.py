import sys
sys.path.append("src")

import argparse
import json
import yaml
from time import time
from pathlib import Path
from typing import Dict, Any, List

from ultralytics import YOLO

# If your YOLO class ids are 0..31 corresponding to FDI 11..48:
IDX_TO_FDI = {
    0:"11", 1:"12", 2:"13", 3:"14", 4:"15", 5:"16", 6:"17", 7:"18",
    8:"21", 9:"22", 10:"23", 11:"24", 12:"25", 13:"26", 14:"27", 15:"28",
    16:"31", 17:"32", 18:"33", 19:"34", 20:"35", 21:"36", 22:"37", 23:"38",
    24:"41", 25:"42", 26:"43", 27:"44", 28:"45", 29:"46", 30:"47", 31:"48",
}
FDI_ORDER = [IDX_TO_FDI[i] for i in sorted(IDX_TO_FDI.keys())]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def yolo_predict_one(model: YOLO, img_path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    conf = float(cfg.get("conf", 0.25))
    iou  = float(cfg.get("iou", 0.70))
    imgsz = int(cfg.get("imgsz", 1024))
    device = cfg.get("device", None)  # "cpu", "0", None

    t0 = time()
    r = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    elapsed_ms = int((time() - t0) * 1000)

    detections: List[dict] = []
    present = set()

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, confs):
            label = IDX_TO_FDI.get(int(c), str(int(c)))
            if label in IDX_TO_FDI.values():
                present.add(label)

            detections.append({
                "label": label,
                "class_id": int(c),
                "conf": float(cf),
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
            })

    teeth_present = [t for t in FDI_ORDER if t in present]
    teeth_missing  = [t for t in FDI_ORDER if t not in present]

    return {
        "pred": {
            "Present teeth": teeth_present,
            "Missing teeth": teeth_missing,
            "Detections": detections,
        },
        "elapsed_ms": elapsed_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True, help="YAML config name")
    ap.add_argument("--imgs-path", default="data/imgs/raw", help="Input images folder")
    ap.add_argument("--save-root", default="results", help="Output root folder")
    args = ap.parse_args()

    config_path = Path("config/yolo") / args.config_name
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    weights = cfg["weights"]  # path to .pt
    imgs_path = Path(args.imgs_path)

    out_dir = Path(args.save_root) / Path(args.config_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)

    imgs = sorted([p for p in imgs_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    for img_path in imgs:
        out = yolo_predict_one(model, img_path, cfg)
        out_path = out_dir / f"{img_path.stem}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {img_path.name} -> {out_dir}")


if __name__ == "__main__":
    main()