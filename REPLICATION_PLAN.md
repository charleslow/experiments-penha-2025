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
│  2. EMBEDDING GENERATION                                        │
│     ├── Search-only bi-encoder (query-item pairs)               │
│     ├── Rec-only bi-encoder (item-item co-occurrence)           │
│     ├── Multi-task bi-encoder (joint training) ← BEST          │
│     └── Fused (concatenate search + rec embeddings)             │
│                                                                 │
│  3. SEMANTIC ID CONSTRUCTION                                    │
│     ├── RQ-KMeans (FAISS residual quantizer) ← BEST            │
│     └── RQ-VAE (baseline)                                       │
│                                                                 │
│  4. GENERATIVE MODEL TRAINING                                   │
│     └── Transformer trained on Semantic ID sequences            │
│                                                                 │
│  5. EVALUATION                                                  │
│     ├── Search: NDCG, Recall                                    │
│     └── Recommendation: NDCG, Recall                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

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

## Implementation Strategy

### Option A: Build on GRID Framework (Recommended)

The [GRID framework](https://github.com/snap-research/GRID) from Snap Research provides:
- Embedding generation with LLMs
- RQ-KMeans, RQ-VAE, RVQ implementations
- Transformer-based generative model (TIGER)

**Modifications needed**:
1. Add bi-encoder training code (contrastive learning)
2. Add search query handling
3. Implement multi-task training loss
4. Add search evaluation metrics

### Option B: Build from Scratch

Use these libraries:
- `sentence-transformers` for bi-encoder
- `faiss` for RQ-KMeans
- `transformers` for generative model

---

## Development vs Production Runs

### Dev Run Configuration

```yaml
# config/dev.yaml
dataset:
  subset_fraction: 0.01  # 1% of data (~12K interactions)
  num_items: 1000        # Subset of items
  queries_per_item: 2    # 1 train, 1 test

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model
  batch_size: 64

semantic_id:
  num_layers: 2          # L=2 (paper uses L=3)
  codebook_size: 64      # W=64 (paper uses W=256)

training:
  epochs: 5
  batch_size: 32
  learning_rate: 1e-4
  early_stopping_patience: 2

evaluation:
  metrics: ["ndcg@10", "recall@10"]

resources:
  device: "cuda"         # Single GPU
  num_workers: 4
```

**Expected dev run time**: ~30 minutes - 1 hour

### Full Run Configuration

```yaml
# config/full.yaml
dataset:
  subset_fraction: 1.0   # Full dataset
  num_items: 62138       # All items
  queries_per_item: 20   # 10 train, 10 test

embedding:
  model: "BAAI/bge-large-en-v1.5"  # Or comparable model
  batch_size: 256

semantic_id:
  num_layers: 3          # L=3 (paper default)
  codebook_size: 256     # W=256 (paper default)

training:
  epochs: 50
  batch_size: 128
  learning_rate: 5e-5
  early_stopping_patience: 5
  warmup_steps: 1000

evaluation:
  metrics: ["ndcg@5", "ndcg@10", "recall@5", "recall@10"]

resources:
  device: "cuda"
  num_gpus: 4            # Multi-GPU recommended
  num_workers: 16
```

**Expected full run time**: ~6-24 hours (depending on GPU)

---

## Recommended Directory Structure

```
experiments-penha-2025/
├── configs/
│   ├── dev.yaml
│   ├── full.yaml
│   └── experiments/
│       ├── search_only.yaml
│       ├── rec_only.yaml
│       ├── multitask.yaml
│       ├── fused.yaml
│       └── separate.yaml
├── src/
│   ├── data/
│   │   ├── movielens.py       # Data loading
│   │   └── query_gen.py       # Query generation
│   ├── models/
│   │   ├── biencoder.py       # Bi-encoder for embeddings
│   │   ├── semantic_id.py     # RQ-KMeans / RQ-VAE
│   │   └── generative.py      # Transformer model
│   ├── training/
│   │   ├── contrastive.py     # Bi-encoder training
│   │   └── generative.py      # Generative model training
│   └── evaluation/
│       ├── search.py          # Search metrics
│       └── recommendation.py  # Rec metrics
├── scripts/
│   ├── run_dev.sh
│   ├── run_full.sh
│   └── run_ablation.sh
├── notebooks/
│   └── analysis.ipynb
└── outputs/
    ├── embeddings/
    ├── semantic_ids/
    ├── checkpoints/
    └── results/
```

---

## Step-by-Step Replication Guide

### Phase 1: Data Preparation

```bash
# Step 1.1: Download MovieLens25M
./scripts/download_movielens.sh

# Step 1.2: Generate synthetic queries (requires Gemini API)
python src/data/query_gen.py \
  --input data/movielens/movies.csv \
  --output data/queries/ \
  --queries_per_item 20 \
  --model gemini-2.0-flash

# Step 1.3: Create train/test splits
python src/data/prepare_splits.py \
  --interactions data/movielens/ratings.csv \
  --output data/splits/
```

### Phase 2: Embedding Generation (Run for each strategy)

```bash
# Multi-task bi-encoder (recommended)
python src/training/contrastive.py \
  --config configs/experiments/multitask.yaml \
  --mode train

# Generate embeddings
python src/models/biencoder.py \
  --config configs/experiments/multitask.yaml \
  --mode inference \
  --output outputs/embeddings/multitask/
```

### Phase 3: Semantic ID Construction

```bash
# RQ-KMeans tokenization
python src/models/semantic_id.py \
  --method rq_kmeans \
  --embeddings outputs/embeddings/multitask/ \
  --num_layers 3 \
  --codebook_size 256 \
  --output outputs/semantic_ids/multitask_rqkmeans/
```

### Phase 4: Generative Model Training

```bash
# Train generative model
python src/training/generative.py \
  --config configs/full.yaml \
  --semantic_ids outputs/semantic_ids/multitask_rqkmeans/ \
  --output outputs/checkpoints/
```

### Phase 5: Evaluation

```bash
# Evaluate on both tasks
python src/evaluation/evaluate.py \
  --checkpoint outputs/checkpoints/best.pt \
  --task both \
  --output outputs/results/
```

---

## Running Experiments

### Quick Dev Validation

```bash
# Single command to run full pipeline on small data
./scripts/run_dev.sh --strategy multitask

# Expected output:
# - Search NDCG@10: ~0.15-0.25 (noisy on small data)
# - Rec NDCG@10: ~0.05-0.10 (noisy on small data)
```

### Full Ablation Study

```bash
# Run all embedding strategies
for strategy in search rec multitask fused separate; do
  ./scripts/run_full.sh --strategy $strategy
done

# Run discretization ablation
for method in rq_kmeans rq_vae; do
  ./scripts/run_full.sh --strategy multitask --tokenizer $method
done
```

### Using SLURM (Cluster)

```bash
# Submit all experiments as array job
sbatch --array=0-4 scripts/slurm_ablation.sh
```

---

## Key Hyperparameters

| Parameter | Dev | Full | Notes |
|-----------|-----|------|-------|
| Semantic ID layers (L) | 2 | 3 | Paper uses L=3 |
| Codebook size (W) | 64 | 256 | Paper uses W=256 |
| Bi-encoder model | MiniLM | BGE-large | Paper likely uses larger |
| Contrastive batch size | 64 | 256+ | In-batch negatives |
| Generative epochs | 5 | 50 | Until convergence |
| Learning rate | 1e-4 | 5e-5 | With warmup |

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

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM during bi-encoder training | Reduce batch size, use gradient accumulation |
| RQ-KMeans slow | Use FAISS GPU implementation |
| Query generation expensive | Cache API responses, use batched calls |
| Evaluation slow | Pre-compute candidate embeddings |

---

## Dependencies

```txt
# Core
torch>=2.0
transformers>=4.30
sentence-transformers>=2.2
faiss-gpu>=1.7

# Data
pandas
numpy
scipy

# Config & Logging
hydra-core
wandb

# Optional
accelerate  # Multi-GPU
bitsandbytes  # Quantization
```

---

## Timeline Estimate

| Phase | Dev | Full |
|-------|-----|------|
| Data prep | 10 min | 30 min |
| Embedding training | 15 min | 2-4 hrs |
| Semantic ID construction | 5 min | 30 min |
| Generative training | 30 min | 4-12 hrs |
| Evaluation | 5 min | 1 hr |
| **Total** | **~1 hr** | **~8-18 hrs** |

---

## Next Steps

1. [ ] Set up environment and install dependencies
2. [ ] Download and prepare MovieLens25M dataset
3. [ ] Implement/adapt bi-encoder training code
4. [ ] Set up query generation pipeline (or use synthetic fallback)
5. [ ] Run dev experiments to validate pipeline
6. [ ] Run full ablation study
7. [ ] Analyze and compare results with paper

---

## References

- [Paper: Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478)
- [GRID Framework (Snap Research)](https://github.com/snap-research/GRID)
- [Practitioner's Handbook for Semantic IDs](https://arxiv.org/abs/2507.22224)
- [RQ-VAE Recommender Implementation](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
