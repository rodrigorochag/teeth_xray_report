# Teeth X-ray Report

## 1. Finetuning (QWEN, QLoRA)

**Config:** `config/local_vlm/*.yaml`

**Run:**
```
uv run python finetune_vlm_qlora.py --config-name qwen_ft.yaml
```

**Output:** `data/checkpoints/<config_name_stem>/checkpoint-*`

Latest checkpoint (largest number) is used for inference.

---

## 2. Inference

### OpenAI (GPT Vision)

**Config:** `config/openai/*.yaml`

**Run:** 
```
uv run python infer_openai.py --config-name gpt_missing.yaml --imgs-path data/imgs/raw --save-root results
```

**Output:** `results/<config_name_stem>/<image>.json`

---

### Local QWEN (fine-tuned)

**Config:** `config/local_vlm/*.yaml`

**Run:**
```
uv run python infer_qwen.py --config-name qwen_inf.yaml --imgs-path data/imgs/raw --save-root results
```

**Output:** `results/<config_name_stem>/<image>.json`

---

### 3. YOLO

**Config:** `config/yolo/*.yaml`

**Run:**
```
uv run python infer_yolo.py --config-name yolo.yaml --imgs-path data/imgs/raw --save-root results
```

**Output:** `results/<config_name_stem>/<image>.json`

---

## Evaluation

**Inputs:**
- GT: `data/ground_truth/<image>.json`
- Predictions: `results/<run_name>/<image>.json`

**Run:**
```
uv run python eval_predictions.py --gt-dir gt --pred-dirs results/<run_name_1> results/<run_name_2> results/<run_name_3> --classes "Missing teeth" "Present teeth"
```

**Output:** `results/<run_name_1>/stats.json` `results/<run_name_2>/stats.json` `results/<run_name_3>/stats.json`

**Metrics:**
- Precision / Recall per class
- Overall (Precision + Recall)
- Mean latency (if available)

---

**Notes:**
- class names must match everywhere (prompt, training, inference, eval)
- QWEN training (optionally) uses loss masking (only JSON contributes to loss)
- GPT uses max_completion_tokens
- YOLO derives missing teeth via FDI set difference
