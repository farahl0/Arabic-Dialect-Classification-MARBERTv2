# Arabic Dialect Classification with MARBERTv2

> Fine-tuned Arabic dialect identification across 5 regional varieties using a frozen **MARBERTv2** backbone with a custom Multi-Head Bahdanau Attention classifier head — trained on the **IADD dataset** with LLM-augmented synthetic data.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Requirements](#requirements)

---

## Overview

This project tackles the task of **Arabic dialect identification (ADI)**, classifying text into one of five regional categories:

| Label | Region |
|---|---|
| `EGY` | Egyptian Arabic |
| `GLF` | Gulf Arabic (includes Iraqi, merged due to linguistic similarity) |
| `LEV` | Levantine Arabic |
| `MGH` | Maghrebi Arabic |
| `general` | Modern Standard Arabic (MSA / Fusha) |

The core challenge is that many dialects share substantial vocabulary with MSA, making fine-grained classification — especially for `general` and `GLF` — particularly difficult. This project addresses that with:

- A **frozen MARBERTv2** backbone (no fine-tuning of transformer weights)
- A custom **Multi-Head Bahdanau Attention** classification head trained from scratch
- **LLM-based synthetic data augmentation** using Qwen2.5-3B-Instruct for underrepresented classes
- **Focal Loss** with class weighting to handle severe class imbalance

---

## Dataset

**IADD (Integrated Arabic Dialect Dataset)** — a large-scale multi-dialect Arabic dataset loaded from a JSON file containing `Sentence`, `Region`, `DataSource`, and `Country` fields.

### Preprocessing Steps

1. **Iraqi dialect merging** — `IRQ` merged into `GLF` (too few samples; linguistically close)
2. **Conflicting label removal** — sentences with 2+ different region labels across the dataset are dropped
3. **Deduplication** — exact duplicate sentences removed
4. **Arabic text normalization**:
   - Remove URLs, `@mentions`, `#hashtags`
   - Strip diacritics (tashkeel) and tatweel
   - Normalize Alef variants (`أ إ آ` → `ا`)
   - Normalize Yeh and Teh Marbuta variants
   - Remove all non-Arabic characters

### Data Splits

| Split | Source | Notes |
|---|---|---|
| Train (75%) | Original + Synthetic | Synthetic injected **only** into train |
| Validation (12.5%) | Original only | No synthetic contamination |
| Test (12.5%) | Original only | Clean evaluation |

### Synthetic Data Augmentation

To address class imbalance (particularly for `EGY`, `GLF`, and `general`), **170 synthetic sentences per class** are generated using **Qwen2.5-3B-Instruct** with:

- Few-shot prompting with 8 seed examples per dialect
- Small batch generation (20 per call) to prevent repetition collapse
- **Fuzzy near-duplicate filtering** via `rapidfuzz` (threshold = 88) to ensure diversity
- Fallback to heuristic word-swap augmentation if LLM generation fails

---

## Model Architecture

```
Input Text
    │
    ▼
[MARBERTv2 Backbone] ── FROZEN (no gradient updates)
    │
    ├── Stream 1: CLS token embedding        [B, 768]
    ├── Stream 2: Token max-pooling          [B, 768]  ← captures peak dialect markers
    └── Stream 3: Multi-Head Bahdanau Attn  [B, 768]  ← 4 independent attention heads
            │
            ▼
    Concatenate [CLS | max-pool | context]  [B, 2304]
            │
    LayerNorm
            │
    Linear(2304 → 512) → GELU → Dropout(0.15)
            │
    Linear(512 → 5)
            │
    Logits (5 classes)
```

### Multi-Head Bahdanau Attention

A custom additive attention mechanism built from scratch with `H` independent heads, each learning its own projection `(W_h, v_h)`:

```
energy_h = tanh(W_h · H)          [B, T, attn_dim]
score_h  = v_h^T · energy_h       [B, T]
alpha_h  = softmax(score_h)        [B, T]  (masked for padding)
ctx_h    = Σ alpha_h_t · H_t      [B, D]

context  = out_proj(cat(ctx_1, ..., ctx_H))   [B, D]
```

The averaged attention weights across heads are also returned for **interpretability visualizations**.

### Key Design Choices

- **Frozen backbone** — MARBERTv2's 163M parameters are not updated; only the ~2M classifier head is trained. This prevents catastrophic forgetting and is computationally efficient.
- **Three-stream pooling** — CLS (global summary) + max-pool (peak features) + attention context (weighted summary) gives the classifier richer representations than CLS alone.
- **GELU activation** — smoother than ReLU for transformer-adjacent layers.
- **LayerNorm** before the classifier stabilizes training.

---

## Training Pipeline

| Hyperparameter | Value |
|---|---|
| Backbone | `UBC-NLP/MARBERTv2` |
| Max sequence length | 96 tokens |
| Batch size | 64 |
| Epochs | 15 (early stopping, patience=3) |
| Optimizer | AdamW (`weight_decay=0.01`) |
| Scheduler | OneCycleLR (`max_lr=5e-4`, 10% warmup, cosine decay) |
| Loss function | Focal Loss (`γ=1.5`) + class weights |
| Attention heads | 4 |
| Attention dim | 128 |
| Dropout | 0.15 |
| Seed | 42 |

### Focal Loss

Standard cross-entropy is replaced with Focal Loss to prioritize hard, misclassified examples over easy ones — crucial for the heavily imbalanced `general` class:

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```

The modulating factor `(1 - p_t)^γ` down-weights easy examples (high `p_t`) and focuses training on hard ones (low `p_t`).

---

## Results

### Test Set Performance

| Metric | Score |
|---|---|
| Macro-F1 | 0.6292 |
| Accuracy | 0.8126 |
| Loss | 0.5279 |

### Per-Class Results (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| EGY | 0.44 | 0.82 | 0.58 | 568 |
| GLF | 0.50 | 0.85 | 0.63 | 854 |
| LEV | 0.99 | 0.80 | 0.88 | 10,537 |
| MGH | 0.85 | 0.89 | 0.87 | 2,768 |
| general | 0.12 | 0.43 | 0.19 | 302 |
| **macro avg** | **0.58** | **0.76** | **0.63** | **15,029** |
| **weighted avg** | **0.90** | **0.81** | **0.84** | **15,029** |

> **Note:** The `general` (MSA) class remains the hardest to classify due to heavy lexical overlap with all dialects. High recall on `EGY` and `GLF` (0.82 / 0.85) shows the model correctly identifies most dialect instances, but precision suffers from MSA sentences being misclassified as dialectal.

### Attention Visualization

The notebook includes attention weight bar plots for:
- **Misclassified `general`** examples (showing which tokens confused the model)
- **Correctly classified** `EGY`, `GLF`, and `LEV` examples

---

## Project Structure

```
Arabic-Dialect-Classification-MARBERTv2/
│
├── marbert.ipynb          # Main notebook (EDA → augmentation → training → evaluation)
├── IADD.json              # Source dataset
├── synthetic_data.csv     # LLM-generated synthetic training data (Qwen2.5-3B-Instruct)
└── README.md
```

> **Note:** `best_model_v2.pt` (the saved model checkpoint) is not included in the repo due to file size. Run the notebook to reproduce it.

---

## Setup & Usage

### Environment

This project is designed to run on **Kaggle** with GPU acceleration. To adapt for local use:

```python
# Update the dataset path
PATH = 'path/to/your/IADD.json'
```

### Installation

```bash
pip install pyarabic transformers accelerate rapidfuzz
```

### Running the Notebook

1. Upload `marbert.ipynb` to Kaggle (or clone this repo)
2. Add the IADD dataset as a Kaggle input dataset
3. Enable GPU accelerator (P100 or T4 recommended)
4. Run all cells in order

The notebook will:
1. Load and preprocess the IADD dataset
2. Generate or load cached synthetic data (`synthetic_data.csv`)
3. Build and train the `DialectClassifierV2` model
4. Save the best checkpoint to `best_model_v2.pt`
5. Evaluate on the test set with full classification report, confusion matrix, and attention visualizations

### Inference (Single Sentence)

```python
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERTv2')
model.eval()

text = "يعني انت شايف ان اللي حصل ده عادي"  # Egyptian example
enc = tokenizer(text, max_length=96, padding='max_length',
                truncation=True, return_tensors='pt')

with torch.no_grad():
    logits, alpha = model(enc['input_ids'].to(device),
                          enc['attention_mask'].to(device))

pred = le.inverse_transform([logits.argmax().item()])[0]
print(f"Predicted dialect: {pred}")
```

---

## Requirements

```
torch
transformers
accelerate
pyarabic
rapidfuzz
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## References

- **MARBERTv2**: [UBC-NLP/MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) — a BERT model pre-trained exclusively on Arabic dialects and MSA
- **IADD Dataset**: Integrated Arabic Dialect Dataset
- **Qwen2.5-3B-Instruct**: Used for synthetic dialect sentence generation
- Bahdanau et al. (2015) — *Neural Machine Translation by Jointly Learning to Align and Translate*
- Lin et al. (2017) — *Focal Loss for Dense Object Detection*
