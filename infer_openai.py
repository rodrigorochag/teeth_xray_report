import sys
sys.path.append("src")

import os
import argparse
import base64
import json
import yaml
from time import time
from pathlib import Path

from openai import OpenAI

from teeth import prompts as P

def get_img_url(p: Path) -> str:
    # Encode image as data URL for OpenAI vision input
    ext = p.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in {"jpg","jpeg"} else f"image/{ext or 'png'}"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"

def extract_json(t: str) -> dict:
    # Parse model output; return raw text on failure for debugging
    try:
        return json.loads(t)
    except Exception as e:
        return {"_error": str(e), "_raw": t}

def get_response(img_path, client, model, prompt, max_output_tokens):
    # Single image -> single JSON prediction via GPT model vision
    img_url = get_img_url(img_path)
    
    start = time()
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]
            }
        ],
        temperature=0.0,
        max_completion_tokens=max_output_tokens,
        response_format={"type": "json_object"}
    )
    elapsed_ms = int((time() - start) * 1000)

    text = response.choices[0].message.content
    pred = extract_json(text)
    return pred, elapsed_ms  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required = True)
    ap.add_argument("--imgs-path", default="data/imgs/raw", help="Input images folder")
    ap.add_argument("--save-root", default="results", help="Output root folder")
    args = ap.parse_args()

    # Load OpenAI inference config
    config_path = Path("config/openai") / args.config_name
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = cfg["model"]
    max_output_tokens = cfg.get("max_output_tokens", 256)
    prompt = getattr(P, cfg["prompt_name"])
    imgs_path = Path(args.imgs_path)
    
    # Create run-specific output directory (one folder per config)
    out_dir = Path(args.save_root) / Path(args.config_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # API key must be provided via environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key = api_key)
    
    # Process all supported image files
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    imgs = sorted([p for p in imgs_path.iterdir() if p.is_file() and p.suffix.lower() in exts])

    for i, img_path in enumerate(imgs):
        pred, elapsed_ms = get_response(img_path, client, model, prompt, max_output_tokens)
        to_save = {"pred": pred, "elapsed_ms": elapsed_ms}
        
        # Save per-image JSON result
        out_path = out_dir / f"{img_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()