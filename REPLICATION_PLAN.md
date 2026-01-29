# Replication Plan: Semantic IDs for Joint Generative Search and Recommendation

**Paper**: [Penha et al. 2025](https://arxiv.org/abs/2508.10478) - RecSys 2025

## Paper Summary

This paper investigates how to construct Semantic IDs that work well for **both search and recommendation** in a unified generative model. The key finding is that a **multi-task bi-encoder** fine-tuned on both tasks, followed by **RQ-KMeans** discretization, provides the best trade-off.

### Key Contributions
1. Compares task-specific vs cross-task Semantic ID construction strategies
2. Shows RQ-KMeans outperforms RQ-VAE for Semantic ID construction
3. Demonstrates multi-task bi-encoder embeddings enable effective joint search & recommendation

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE OVERVIEW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA PREPARATION                                            │
│     └── MovieLens25M + Synthetic Queries (Gemini-2.0-flash)     │
│                                                                 │
│  2. EMBEDDING GENERATION (Bi-encoder: all-mpnet-base-v2)        │
│     ├── Search-only bi-encoder (query-item pairs)               │
│     ├── Rec-only bi-encoder (item-item co-occurrence)           │
│     ├── Multi-task bi-encoder (joint training) ← BEST          │
│     └── Fused (concatenate search + rec embeddings)             │
│                                                                 │
│  3. SEMANTIC ID CONSTRUCTION                                    │
│     ├── RQ-KMeans (FAISS residual quantizer) ← BEST            │
│     └── RQ-VAE (baseline)                                       │
│                                                                 │
│  4. GENERATIVE MODEL TRAINING (flan-t5-base)                    │
│     └── T5 fine-tuned on Semantic ID sequences                  │
│                                                                 │
│  5. EVALUATION                                                  │
│     ├── Search: NDCG, Recall                                    │
│     └── Recommendation: NDCG, Recall                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Specifications

| Component | Dev Run | Full Run |
|-----------|---------|----------|
| **Embedding Model** | `all-MiniLM-L6-v2` (22M params) | `all-mpnet-base-v2` (110M params) |
| **Generative LLM** | `google/flan-t5-small` (60M params) | `google/flan-t5-base` (250M params) |
| **Semantic ID (L, W)** | (2, 64) | (3, 256) |

### Why These Models?

- **all-mpnet-base-v2**: Paper's choice for embedding generation. Produces 768-dim embeddings.
- **flan-t5-base**: Encoder-decoder architecture well-suited for sequence generation. Fine-tuned on 1.8K tasks.
- **flan-t5-small**: 4x smaller than base, suitable for rapid iteration during development.

---

## Dataset

| Component | Details |
|-----------|---------|
| Base Dataset | MovieLens25M |
| Items | 62,138 movies |
| Interactions | 1.24M user-item pairs |
| Train/Test Split | Chronological (last interaction per user for test) |
| Queries | 20 per item (10 train / 10 test), generated with Gemini-2.0-flash |

---

## Cloud Platform Recommendation: RunPod

### Why RunPod over Lightning AI?

| Factor | RunPod | Lightning AI |
|--------|--------|--------------|
| **Pricing** | $0.22-0.44/hr (RTX 3090/4090) | Higher, enterprise-focused |
| **Flexibility** | Raw GPU access, any framework | Opinionated workflow |
| **Cold starts** | Sub-200ms | Slower |
| **Data egress** | Free | Standard cloud fees |
| **Best for** | ML research, experimentation | Production ML pipelines |

### Recommended GPU Configurations

| Run Type | GPU | VRAM | Cost/hr | Notes |
|----------|-----|------|---------|-------|
| **Dev** | RTX 3090 | 24GB | ~$0.22 (community) | Sufficient for small models |
| **Full** | RTX A6000 | 48GB | ~$0.79 | Or 2x RTX 3090 |
| **Full (fast)** | A100 80GB | 80GB | ~$1.64 | Single GPU, no sharding needed |

### RunPod Setup

```bash
# 1. Create pod with PyTorch template
#    - Select GPU: RTX 3090 (dev) or A6000/A100 (full)
#    - Disk: 50GB (dev) or 200GB (full)
#    - Template: RunPod PyTorch 2.1

# 2. SSH into pod
ssh root@<pod-ip> -i ~/.ssh/runpod

# 3. Clone and setup
git clone https://github.com/snap-research/GRID.git
cd GRID
pip install -r requirements.txt

# 4. Install additional dependencies for this replication
pip install sentence-transformers wandb
```

---

## GRID Framework Setup

### Step 1: Clone and Install

```bash
git clone https://github.com/snap-research/GRID.git
cd GRID
pip install -r requirements.txt

# Additional dependencies
pip install sentence-transformers==2.2.2
pip install wandb  # For experiment tracking
```

### Step 2: GRID Directory Structure

```
GRID/
├── src/
│   ├── train.py           # Main training entry point
│   ├── inference.py       # Inference entry point
│   ├── models/
│   │   ├── rkmeans.py     # RQ-KMeans implementation
│   │   ├── rqvae.py       # RQ-VAE implementation
│   │   └── tiger.py       # TIGER recommendation model
│   └── data/
│       └── dataset.py     # Data loading
├── configs/
│   └── experiment/        # Hydra configs
└── data/
    └── amazon_data/       # Example data format
```

### Step 3: Modifications Needed for Penha Replication

GRID is designed for recommendation only. We need to add:

1. **Bi-encoder training** for embedding generation
2. **Search task handling** (query-item pairs)
3. **Multi-task training** (joint search + rec loss)
4. **Search evaluation metrics**

```
# Files to add/modify in GRID:
src/
├── models/
│   └── biencoder.py       # NEW: Bi-encoder for embeddings
├── training/
│   ├── contrastive.py     # NEW: Contrastive learning
│   └── multitask.py       # NEW: Multi-task training
├── data/
│   ├── movielens.py       # NEW: MovieLens data loader
│   └── query_pairs.py     # NEW: Query-item pairs
└── evaluation/
    └── search_metrics.py  # NEW: Search NDCG, Recall
```

---

## Development vs Production Runs

### Dev Run Configuration

```yaml
# config/dev.yaml
dataset:
  name: movielens25m
  subset_fraction: 0.01      # 1% of data (~12K interactions)
  num_items: 1000            # Subset of items
  queries_per_item: 2        # 1 train, 1 test

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # 22M params
  embedding_dim: 384
  batch_size: 64

semantic_id:
  method: rq_kmeans
  num_layers: 2              # L=2
  codebook_size: 64          # W=64

generative:
  model: "google/flan-t5-small"  # 60M params
  max_length: 32

training:
  epochs: 5
  batch_size: 32
  learning_rate: 1e-4
  gradient_accumulation: 1
  early_stopping_patience: 2

evaluation:
  metrics: ["ndcg@10", "recall@10"]

resources:
  gpu: "rtx3090"
  num_workers: 4
```

**Dev run specs:**
- GPU: RTX 3090 (24GB) - ~$0.22/hr on RunPod community
- Time: ~30-60 minutes
- Cost: ~$0.20-0.40

### Full Run Configuration

```yaml
# config/full.yaml
dataset:
  name: movielens25m
  subset_fraction: 1.0       # Full dataset
  num_items: 62138           # All items
  queries_per_item: 20       # 10 train, 10 test

embedding:
  model: "sentence-transformers/all-mpnet-base-v2"  # 110M params (paper)
  embedding_dim: 768
  batch_size: 256

semantic_id:
  method: rq_kmeans
  num_layers: 3              # L=3 (paper)
  codebook_size: 256         # W=256 (paper)

generative:
  model: "google/flan-t5-base"  # 250M params (paper)
  max_length: 64

training:
  epochs: 50
  batch_size: 128
  learning_rate: 5e-5
  gradient_accumulation: 4
  warmup_steps: 1000
  early_stopping_patience: 5

evaluation:
  metrics: ["ndcg@5", "ndcg@10", "recall@5", "recall@10"]

resources:
  gpu: "a100_80gb"           # Or 2x RTX 3090
  num_workers: 16
```

**Full run specs:**
- GPU: A100 80GB (~$1.64/hr) or 2x A6000 48GB (~$1.58/hr)
- Time: ~8-18 hours
- Cost: ~$15-30

---

## Working with Claude in Dev Environment

### Recommended Workflow for Auto-Correction

#### Option 1: Claude Code CLI (Recommended)

Run Claude Code directly in your RunPod terminal:

```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Run in your project directory
cd /workspace/GRID
claude

# Claude can now:
# - Read error messages and fix code
# - Run experiments and analyze results
# - Iterate on failing tests
```

**Best practices:**
```bash
# 1. Keep a terminal with training running
python src/train.py experiment=dev 2>&1 | tee logs/train.log

# 2. In another terminal, run Claude Code
claude

# 3. Ask Claude to monitor and fix issues:
#    "Watch logs/train.log and fix any errors"
#    "The training crashed with OOM, reduce batch size"
#    "Analyze the results in outputs/ and suggest improvements"
```

#### Option 2: VSCode + Claude Extension

1. Install VSCode Remote SSH extension
2. Connect to RunPod via SSH
3. Install Claude extension in VSCode
4. Claude can see your code and terminal output

#### Option 3: Structured Error Feedback Loop

Create a script that captures errors for Claude:

```bash
#!/bin/bash
# scripts/run_with_feedback.sh

set -o pipefail

python src/train.py experiment=$1 2>&1 | tee logs/run.log

if [ $? -ne 0 ]; then
    echo "=== RUN FAILED ===" >> logs/run.log
    echo "Error occurred. Check logs/run.log"
    # Optionally: call Claude API to analyze error
fi
```

### Claude-Friendly Project Structure

```
experiments-penha-2025/
├── .claude/
│   └── settings.json        # Claude Code settings
├── logs/
│   ├── train.log            # Training logs (Claude can read)
│   ├── errors.log           # Error logs for debugging
│   └── results.json         # Metrics (Claude can analyze)
├── scripts/
│   ├── run_dev.sh           # Simple entry points
│   ├── run_full.sh
│   └── validate.sh          # Quick validation script
└── tests/
    └── test_pipeline.py     # Tests Claude can run
```

### Auto-Correction Patterns

**Pattern 1: Test-Driven Development**
```bash
# Ask Claude to run tests after each change
pytest tests/ -v 2>&1 | tee logs/test.log

# If tests fail, Claude sees the output and can fix
```

**Pattern 2: Incremental Validation**
```bash
# Create small validation checkpoints
python scripts/validate.py --step data_loading
python scripts/validate.py --step embedding_generation
python scripts/validate.py --step semantic_id
python scripts/validate.py --step training
```

**Pattern 3: Error-Specific Prompts**
```
# When you see an error, paste it to Claude with context:
"I got this error when running the embedding generation step:
[paste error]
The relevant code is in src/models/biencoder.py. Please fix it."
```

### Recommended Claude Code Settings

```json
// .claude/settings.json
{
  "context": {
    "include": [
      "src/**/*.py",
      "configs/**/*.yaml",
      "logs/*.log"
    ],
    "exclude": [
      "data/**",
      "outputs/**",
      "*.pt"
    ]
  }
}
```

---

## Experimental Conditions to Replicate

### Embedding Strategies (Table 1 in paper)

| Strategy | Description | Token Budget |
|----------|-------------|--------------|
| **Search** | Bi-encoder trained on query-item pairs only | 1x |
| **Rec** | Bi-encoder trained on item-item co-occurrence only | 1x |
| **Multi-task** | Bi-encoder jointly trained on both tasks | 1x |
| **Fused** | Concatenate Search + Rec embeddings | 1x |
| **Separate** | Task-specific Semantic IDs (separate token spaces) | 2x |

### Discretization Methods (Ablation)

| Method | Description |
|--------|-------------|
| **RQ-KMeans** | FAISS residual quantizer (recommended) |
| **RQ-VAE** | Variational autoencoder approach |

---

## Step-by-Step Replication Guide

### Phase 1: Environment Setup

```bash
# On RunPod or local machine
git clone https://github.com/snap-research/GRID.git
cd GRID

# Create conda environment
conda create -n penha python=3.10
conda activate penha

# Install dependencies
pip install -r requirements.txt
pip install sentence-transformers==2.2.2 wandb

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Phase 2: Data Preparation

```bash
# Download MovieLens25M
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/

# Generate synthetic queries (requires Gemini API key)
export GOOGLE_API_KEY="your-key"
python src/data/query_gen.py \
  --input data/ml-25m/movies.csv \
  --output data/queries/ \
  --queries_per_item 20 \
  --model gemini-2.0-flash

# Create train/test splits
python src/data/prepare_splits.py \
  --interactions data/ml-25m/ratings.csv \
  --output data/splits/
```

### Phase 3: Run Dev Experiment (Validate Pipeline)

```bash
# Quick validation with small data
python src/train.py experiment=dev

# Check outputs
ls outputs/dev/
# Should see: embeddings/, semantic_ids/, checkpoints/, results/
```

### Phase 4: Run Full Experiments

```bash
# Run all embedding strategies
for strategy in search rec multitask fused separate; do
  python src/train.py \
    experiment=full \
    embedding.strategy=$strategy \
    wandb.name="penha_${strategy}"
done

# Run discretization ablation
for method in rq_kmeans rq_vae; do
  python src/train.py \
    experiment=full \
    embedding.strategy=multitask \
    semantic_id.method=$method \
    wandb.name="penha_multitask_${method}"
done
```

### Phase 5: Analyze Results

```bash
# Compare results
python scripts/analyze_results.py --output outputs/comparison.md

# Generate plots
python scripts/plot_results.py --output outputs/figures/
```

---

## Key Hyperparameters

| Parameter | Dev | Full | Notes |
|-----------|-----|------|-------|
| Embedding model | all-MiniLM-L6-v2 | all-mpnet-base-v2 | Paper uses mpnet |
| Embedding dim | 384 | 768 | Matches model output |
| Generative model | flan-t5-small | flan-t5-base | Paper uses base |
| Semantic ID layers (L) | 2 | 3 | Paper uses L=3 |
| Codebook size (W) | 64 | 256 | Paper uses W=256 |
| Contrastive batch size | 64 | 256 | In-batch negatives |
| Generative batch size | 32 | 128 | With grad accumulation |
| Learning rate | 1e-4 | 5e-5 | With warmup |
| Epochs | 5 | 50 | Until convergence |

---

## Expected Results (from paper)

| Strategy | Search NDCG@10 | Rec NDCG@10 |
|----------|----------------|-------------|
| Search-only | **Best** | Poor |
| Rec-only | Poor | **Best** |
| Multi-task | Good | Good |
| Fused | Moderate | Moderate |
| Separate | Good | Good |

**Key finding**: Multi-task provides best trade-off without doubling token budget.

---

## Cost Estimation

| Phase | Dev (RTX 3090) | Full (A100) |
|-------|----------------|-------------|
| Data prep | Free | Free |
| Embedding training | ~$0.10 | ~$3.00 |
| Semantic ID | ~$0.02 | ~$0.50 |
| Generative training | ~$0.15 | ~$10.00 |
| Evaluation | ~$0.02 | ~$1.00 |
| **Total per run** | **~$0.30** | **~$15.00** |
| **Full ablation (7 runs)** | **~$2.00** | **~$100.00** |

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM during bi-encoder training | Reduce batch size, use gradient accumulation |
| OOM with flan-t5-base | Use gradient checkpointing, reduce seq length |
| RQ-KMeans slow | Use FAISS GPU: `faiss-gpu` package |
| Query generation expensive | Cache API responses, batch calls |
| RunPod pod stops | Use persistent volume, checkpoint frequently |

---

## Dependencies

```txt
# requirements.txt
torch>=2.0
transformers>=4.35
sentence-transformers>=2.2.2
faiss-gpu>=1.7.4

# Data
pandas>=2.0
numpy>=1.24
scipy>=1.10

# Config & Logging
hydra-core>=1.3
wandb>=0.15

# Training
accelerate>=0.24
bitsandbytes>=0.41  # Optional: quantization

# Evaluation
scikit-learn>=1.3
```

---

## Timeline Estimate

| Phase | Dev | Full |
|-------|-----|------|
| Environment setup | 10 min | 10 min |
| Data prep | 10 min | 30 min |
| Embedding training | 15 min | 2-4 hrs |
| Semantic ID construction | 5 min | 30 min |
| Generative training | 30 min | 4-12 hrs |
| Evaluation | 5 min | 1 hr |
| **Total per run** | **~1 hr** | **~8-18 hrs** |

---

## Next Steps

1. [ ] Set up RunPod account and create pod
2. [ ] Clone GRID framework and install dependencies
3. [ ] Download MovieLens25M dataset
4. [ ] Add bi-encoder training code to GRID
5. [ ] Generate synthetic queries (or use subset for dev)
6. [ ] Run dev experiment to validate pipeline
7. [ ] Run full ablation study
8. [ ] Analyze and compare results with paper

---

## References

- [Paper: Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478)
- [GRID Framework (Snap Research)](https://github.com/snap-research/GRID)
- [Practitioner's Handbook for Semantic IDs](https://arxiv.org/abs/2507.22224)
- [RunPod GPU Cloud](https://www.runpod.io/)
- [Flan-T5 Models on HuggingFace](https://huggingface.co/google/flan-t5-base)
- [all-mpnet-base-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
