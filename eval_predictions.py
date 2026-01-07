import sys
sys.path.append("src")

import argparse
import json
from pathlib import Path

from teeth.evals import evaluate_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default = "data/ground_truth")
    ap.add_argument(
        "--pred-dirs",
        nargs="+",
        required=True,
        help="One or more prediction folders"
    )
    ap.add_argument(
        "--classes",
        nargs="+",
        default=["Missing teeth", "Present teeth"]
    )
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)

    for pred_dir in args.pred_dirs:
        pred_dir = Path(pred_dir)

        results = evaluate_predictions(
            gt_path=gt_dir,
            pred_path=pred_dir,
            classes=args.classes,
        )

        out_path = pred_dir / "stats.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] {pred_dir} -> stats.json")


if __name__ == "__main__":
    main()