import sys
sys.path.append("src")

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
import yaml
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Trainer

from teeth.utils import (
    get_prompt,
    target_to_json_str,
    get_train_args,
    get_model,
    load_teeth_dataset,
)


@dataclass
class DataCollatorQwenVL:
    """
    Builds multimodal (text + image) training batches for Qwen-VL.

    Key idea:
    - Prompt + image is given as input
    - Target JSON is appended as assistant output
    - Loss is optionally masked so it is computed ONLY on the JSON
    """
    processor: Any
    classes: List[str]
    prompt_text: str
    img_res: Tuple[int, int]
    loss_masking: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, images, prompt_lens = [], [], []

        for ex in features:
            # Load and downscale panoramic X-ray (speed / memory win)
            img = Image.open(ex["image"]).convert("RGB")
            img.thumbnail(self.img_res, Image.Resampling.BILINEAR)

            # Canonical JSON target (sorted, strings, required keys only)
            tgt_json = target_to_json_str(target = ex["target"], classes = self.classes)

            # Single-turn chat prompt (user + image)
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": self.prompt_text},
                    {"type": "image", "image": img},
                ]},
            ]

            # Build raw text prompt (ends with assistant token)
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # Model sees: [prompt][target_json]
            texts.append(prompt + tgt_json)
            images.append(img)

            # For loss masking: compute how many tokens belong to the prompt
            if self.loss_masking:
                p_ids = self.processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )["input_ids"][0]
                prompt_lens.append(p_ids.shape[0])

        # Final batch tokenization (no truncation)
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation = False,
        )
        labels = batch["input_ids"].clone()

        # Ignore padding in loss
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        # Ignore prompt tokens in loss (train only on JSON)
        if self.loss_masking:
            for i, start in enumerate(prompt_lens):
                labels[i, :start] = -100

        batch["labels"] = labels
        return batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required = True)
    args = ap.parse_args()

    # Load experiment config
    config_path = Path("config/local_vlm") / args.config_name
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Prompt + task definition
    classes = cfg["classes"]
    prompt_text = get_prompt(cfg["prompt_name"], classes = classes)
    
    # Image preprocessing
    img_res = (cfg["training"]["img_res1"], cfg["training"]["img_res2"])
    loss_masking = cfg["training"]["loss_masking"]
    
    # QLoRA + LoRA model
    model = get_model(cfg)

    # Dataset loader (keeps only image + selected target classes)
    ds = load_teeth_dataset(cfg)

    # HuggingFace Trainer arguments
    train_args = get_train_args(cfg, args.config_name)

    # Processor handles text + vision tokenization
    processor = AutoProcessor.from_pretrained(cfg["model_name"], trust_remote_code=True, use_fast=False)

    # Custom collator: builds (prompt + image -> JSON) samples and optionally masks loss to JSON only
    collator = DataCollatorQwenVL(
        processor=processor,
        classes=classes,
        prompt_text=prompt_text,
        img_res=img_res,
        loss_masking=loss_masking,
    )

    # Create Trainer and run QLoRA fine-tuning
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    trainer.train()

    # Save final model + processor
    trainer.save_model(train_args.output_dir)
    processor.save_pretrained(train_args.output_dir)

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()