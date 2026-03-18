
from typing import Any, Dict, List
import json
from pathlib import Path
from random import random, uniform
from PIL import ImageEnhance, ImageFilter, Image

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Features, Value, Sequence, Dataset, DatasetDict

from . import prompts as P

def get_prompt(prompt_name: str, classes: List) -> str:
    prompt_base = getattr(P, prompt_name)

    CLASSES = "\n".join(f"- {c}" for c in classes)
    JSON_SCHEMA = ",\n".join(f'  "{c}": []' for c in classes)
    prompt = prompt_base.format(CLASSES=CLASSES, JSON_SCHEMA=JSON_SCHEMA)
    return prompt

def canonicalize_target(t: Dict[str, Any], classes: List[str]) -> Dict[str, Any]:
    out = {}
    for k in classes:
        xs = t.get(k, [])
        if not isinstance(xs, list):
            xs = []
        out[k] = sorted({str(x) for x in xs}, key=lambda s: int(s) if s.isdigit() else 10**9)
    return out

def target_to_json_str(target: Dict[str, Any], classes: List) -> str:
    target = canonicalize_target(target, classes = classes)
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))

def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception as e:
        return {"_error": str(e), "_raw": text}

def augment_pano(img: Image.Image) -> Image.Image:
    # Mild photometric jitter
    if random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(uniform(0.85, 1.15))
    if random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(uniform(0.85, 1.25))

    # Very small blur sometimes (simulates acquisition noise)
    if random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=uniform(0.2, 0.8)))

    return img


def get_train_args(cfg: Dict, config_name: str):
    # One output folder per config (keeps runs isolated)
    config_stem = Path(config_name).stem
    output_dir = Path(cfg["output_dir"]) / config_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    per_device_train_batch_size = cfg["training"]["per_device_train_batch_size"]
    gradient_accumulation_steps = cfg["training"]["gradient_accumulation_steps"]
    lr = float(cfg["training"]["learning_rate"])
    num_train_epochs = cfg["training"]["num_train_epochs"]
    logging_steps = cfg["training"]["logging_steps"]
    eval_steps = cfg["training"]["eval_steps"]
    save_ckpt_steps = cfg["training"]["save_ckpt_steps"]
    save_total_limit = cfg["training"]["save_total_limit"]

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        logging_strategy = "steps",
        
        # Evaluate and checkpoint periodically (needed for "best model" selection)
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps", 
        save_steps=save_ckpt_steps,
        save_total_limit=save_total_limit,

        # Keep the checkpoint with lowest eval_loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        remove_unused_columns=False, # required for multimodal batch dicts
        report_to="tensorboard",
    )

def get_model(cfg: Dict):
    model_name = cfg["model_name"]
    r = cfg["training"]["lora_rank"]
    lora_alpha = cfg["training"]["lora_scale"] * r
    lora_dropout = cfg["training"].get("lora_dropout", 0.05)
    embeddings_grad = cfg["training"]["embeddings_grad"]

    # QLoRA: load base in 4-bit, train only LoRA adapters
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
        dtype=torch.bfloat16,
    )

    # KV cache conflicts with gradient checkpointing (training)
    model.config.use_cache = False

    # Prep quantized model for k-bit training + checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA config (alpha is scaling; typically proportional to r)
    lora = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora)


    # Ensure grads flow correctly with PEFT + checkpointing stacks
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Force gradients through input embeddings
    if embeddings_grad:
        model.get_input_embeddings().weight.requires_grad_(True)

    model.print_trainable_parameters()
    return model

def load_teeth_dataset(cfg):
    path_train = cfg["train_jsonl"]
    path_val = cfg["val_jsonl"]
    classes = cfg["classes"]

    TARGET_KEYS = [
        "Teeth with mesial inclination",
        "Missing teeth",
        "Present teeth",
        "Crown inclination",
        "Type of dentition",
        "Endodontic treatment",
    ]
    
    features = Features({
        "image": Value("string"),
        "target": {
            **{k: Sequence(Value("string")) for k in TARGET_KEYS if k != "Type of dentition"},
            "Type of dentition": Value("string"),
        },
    })

    ds = load_dataset(
        "json",
        data_files={"train": path_train, "validation": path_val},
        features=features,
    )

    # Keep only required keys in target (allows changing "classes" via config)
    def keep_required(example):
        t = example.get("target", {}) or {}
        target = {k: t.get(k, []) for k in classes}
        return {"image": example["image"], "target": target}

    ds_train = list(map(keep_required, ds["train"]))
    ds_valid = list(map(keep_required, ds["validation"]))
    ds = DatasetDict({
        "train": Dataset.from_list(ds_train),
        "validation": Dataset.from_list(ds_valid),
    })

    return ds


# INFERENCE

def get_ckpt_path(ckpt_root: Path):
    # Pick latest checkpoint by global step (checkpoint-<step>)
    ckpt_dirs = [
        p for p in ckpt_root.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not ckpt_dirs:
        raise RuntimeError(f"No checkpoints found in {ckpt_root}")

    return max(ckpt_dirs, key=lambda p: int(p.name.split("-")[-1]))
    

def get_qwen_infer_model(cfg: Dict[str, Any], cfg_name: str, ckpt_name: str | None = None):
    """Load base Qwen (4-bit) + attach latest LoRA checkpoint for inference."""
    model_name = cfg["model_name"]
    
    config_stem = Path(cfg_name).stem
    if ckpt_name is None:
        ckpt_path = get_ckpt_path(Path(cfg["output_dir"]) / config_stem)
    else:
        ckpt_path = Path(cfg["output_dir"]) / config_stem / ckpt_name

    # Must match training quantization settings
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    base = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb,
        dtype=torch.bfloat16,
    )

    # Cache ON for faster autoregressive generation
    base.config.use_cache = True

    model = PeftModel.from_pretrained(base, ckpt_path)
    model.eval()

    return processor, model