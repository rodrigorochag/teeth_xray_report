import sys
sys.path.append("src")

import argparse
import json
import yaml
from pathlib import Path
from time import time
from typing import Any, Tuple

import torch
from PIL import Image

from teeth.utils import get_prompt, extract_json, get_qwen_infer_model


def qwen_predict_one(
    processor: Any,
    model: Any,
    prompt_text: str,
    img_path: Path,
    img_res: Tuple[int, int],
    max_new_tokens: int,
):
    # Match image preprocessing used during finetuning
    img = Image.open(img_path).convert("RGB")
    img.thumbnail(img_res, Image.Resampling.BILINEAR)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image", "image": img},
        ]},
    ]

    # Build chat-style prompt ending with assistant token
    prompt_chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=prompt_chat,
        images=img,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed_ms = int((time() - t0) * 1000)

    # Decode only generated tokens (exclude prompt)
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    pred = extract_json(raw_text)
    return pred, elapsed_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True, help="YAML config name")
    ap.add_argument("--imgs-path", default="data/imgs/raw", help="Input images folder")
    ap.add_argument("--save-root", default="results", help="Output root folder")
    args = ap.parse_args()

    config_path = Path("config/local_vlm") / args.config_name
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]
    prompt_text =  get_prompt(cfg["prompt_name"], classes = classes)
    imgs_path = Path(args.imgs_path)
    img_res = (int(cfg.get("img_res1", 1380)), int(cfg.get("img_res2", 1380)))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))

    # Create run-specific output directory (one folder per config)
    out_dir = Path(args.save_root) / Path(args.config_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base Qwen model + LoRA adapter for inference
    processor, model = get_qwen_infer_model(cfg, args.config_name)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    imgs = sorted([p for p in imgs_path.iterdir() if p.is_file() and p.suffix.lower() in exts])

    for img_path in imgs:
        pred, elapsed_ms = qwen_predict_one(
            processor=processor,
            model=model,
            prompt_text=prompt_text,
            img_path=img_path,
            img_res=img_res,
            max_new_tokens=max_new_tokens,
        )

        out_path = out_dir / f"{img_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"pred": pred, "elapsed_ms": elapsed_ms}, f, ensure_ascii=False, indent=2) # TODO: add elapsed time
        print(f"[ok] {img_path.name} -> {out_dir}")

if __name__ == "__main__":
    main()