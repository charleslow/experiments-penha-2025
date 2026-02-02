# Replication Plan: Semantic IDs for Joint Generative Search and Recommendation

**Paper**: [Penha et al. 2025](https://arxiv.org/abs/2508.10478) - RecSys 2025

## Summary

This paper investigates how to construct Semantic IDs that work well for **both search and recommendation** in a unified generative model. Key finding: **multi-task bi-encoder** + **RQ-KMeans** discretization provides the best trade-off.

**Key Contributions:**
1. Compares task-specific vs cross-task Semantic ID construction strategies
2. Shows RQ-KMeans outperforms RQ-VAE for Semantic ID construction
3. Demonstrates multi-task bi-encoder embeddings enable effective joint search & recommendation

---

## Our Experimental Goals

We focus on two ablations:

### Ablation 1: Embedding Strategy (Multi-task vs Single-task)
Fixed: RQ-KMeans discretization

| Condition | Description |
|-----------|-------------|
| Search-only | Bi-encoder trained on query-item pairs only |
| Rec-only | Bi-encoder trained on item-item co-occurrence only |
| **Multi-task** | Bi-encoder jointly trained on both tasks |

### Ablation 2: Discretization Method
Fixed: Multi-task embeddings

| Condition | Description |
|-----------|-------------|
| **RQ-KMeans** | FAISS residual quantizer |
| RQ-VAE | Variational autoencoder |
| LSH | Locality-sensitive hashing |
| PQ | Product quantization |

### Compute Budget

| Run Type | Target Time | GPU |
|----------|-------------|-----|
| Dev run | < 10 minutes | A4500 (20GB) |
| Full experiment | < 5 hours total | A4500 (20GB) |

Full experiments: 5 seeds × (3 embedding + 4 discretization) = 35 runs
- Ablation 1: 3 × 5 = 15 runs
- Ablation 2: 4 × 5 = 20 runs
- Target: ~8.5 min per run average

---

## Pipeline Overview

```
1. DATA PREPARATION
   └── MovieLens25M + Synthetic Queries (open-source LLM)

2. EMBEDDING GENERATION (Bi-encoder)
   ├── Search-only (query-item pairs)
   ├── Rec-only (item-item co-occurrence)
   └── Multi-task (joint training) ← MAIN FOCUS

3. SEMANTIC ID CONSTRUCTION
   ├── RQ-KMeans (FAISS) ← MAIN FOCUS
   ├── RQ-VAE (baseline)
   └── Others (LSH, PQ)

4. GENERATIVE MODEL TRAINING (flan-t5)
   └── T5 fine-tuned on Semantic ID sequences

5. EVALUATION
   └── NDCG@{5,10}, Recall@{5,10} for both Search and Rec
```

---

## Configuration

| Parameter | Dev | Full | Notes |
|-----------|-----|------|-------|
| **Embedding model** | all-MiniLM-L6-v2 (22M) | all-mpnet-base-v2 (110M) | Paper uses mpnet |
| **Embedding dim** | 384 | 768 | |
| **Generative model** | flan-t5-small (60M) | flan-t5-base (250M) | Paper uses base |
| **Semantic ID (L, W)** | (2, 64) | (3, 256) | Paper uses L=3, W=256 |
| **Dataset fraction** | 1% (~12K interactions) | 100% (1.24M) | |
| **Queries per item** | 2 | 10 | Open-source LLM generated |
| **Contrastive batch** | 64 | 128 | Reduced for A4500 VRAM |
| **Generative batch** | 32 | 64 | With grad accumulation |
| **Learning rate** | 1e-4 | 5e-5 | With warmup |
| **Epochs** | 5 | 30 | Early stopping |
| **GPU** | A4500 (20GB) | A4500 (20GB) | |

---

## Dataset

| Component | Details |
|-----------|---------|
| Base | MovieLens25M |
| Items | 62,138 movies |
| Interactions | 1.24M user-item pairs |
| Split | Chronological (last interaction per user for test) |
| Queries | 10 per item (5 train / 5 test), open-source LLM |

### Query Generation (Preprocessing - not counted in experiment time)

Use Qwen3-8B for synthetic query generation:
- **Model**: Qwen3-8B via vLLM or Ollama
- **Prompt**: Generate realistic search queries a user might type to find this movie
- **Caching**: All generated queries cached to `/app/data/queries/`
- **Note**: This is one-time preprocessing, cached and reused across all experiments

---

## Experimental Matrix

### Ablation 1: Embedding Strategy (3 conditions × 5 seeds = 15 runs)
Fixed: RQ-KMeans, Multi-task generative model

| Strategy | Training Data | Expected Outcome |
|----------|---------------|------------------|
| Search-only | Query-item pairs | Best search, poor rec |
| Rec-only | Item-item co-occurrence | Poor search, best rec |
| Multi-task | Both jointly | Good at both |

### Ablation 2: Discretization Method (4 conditions × 5 seeds = 20 runs)
Fixed: Multi-task embeddings

| Method | Description | Expected |
|--------|-------------|----------|
| RQ-KMeans | FAISS residual quantizer | Best |
| RQ-VAE | Variational autoencoder | Baseline |
| LSH | Locality-sensitive hashing | Fast but lower quality |
| PQ | Product quantization | Alternative baseline |

---

## GRID Assessment

**Status**: Need to evaluate GRID codebase to determine:
1. What components are reusable (RQ-VAE, T5 training, evaluation)?
2. What needs to be written from scratch (bi-encoder, multi-task training, search)?
3. Is it cleaner to extend GRID or write standalone code?

**Decision criteria**:
- If GRID has >50% reusable code → extend it
- If GRID is too complex or incompatible → write minimal standalone implementation

**Components needed** (regardless of approach):
```
src/
├── models/biencoder.py        # Bi-encoder for embeddings
├── training/contrastive.py    # Contrastive learning (search, rec, multi-task)
├── discretization/rq_kmeans.py # FAISS residual quantizer
├── discretization/rq_vae.py    # RQ-VAE (can reuse from GRID if good)
├── discretization/lsh.py       # Locality-sensitive hashing
├── discretization/pq.py        # Product quantization
├── data/movielens.py          # MovieLens data loader
├── data/query_generator.py    # Qwen3-8B query generation
└── evaluation/metrics.py      # NDCG, Recall for both tasks

tests/
├── test_data/                 # Data loading tests
├── test_models/               # Model tests
├── test_discretization/       # Discretization tests
├── test_evaluation/           # Metric tests
└── test_integration.py        # End-to-end Lightning tests
```

---

## Testing Strategy

### Unit Tests (pytest)
Test individual components in isolation:

```
tests/
├── test_data/
│   ├── test_movielens.py      # Data loading, splits, preprocessing
│   └── test_query_generator.py # Query generation, caching
├── test_models/
│   ├── test_biencoder.py      # Forward pass, embedding shapes
│   └── test_contrastive.py    # Loss computation, batch handling
├── test_discretization/
│   ├── test_rq_kmeans.py      # FAISS quantizer, ID generation
│   ├── test_rq_vae.py         # VAE encoding/decoding
│   ├── test_lsh.py            # LSH hashing
│   └── test_pq.py             # Product quantization
└── test_evaluation/
    └── test_metrics.py        # NDCG, Recall correctness
```

**Key unit test requirements:**
- Test with small synthetic data (no external dependencies)
- Verify tensor shapes and dtypes
- Check edge cases (empty batches, single items)
- Validate metric computation against known values

### Integration Tests (PyTorch Lightning)
Test end-to-end pipeline with minimal data:

```python
# tests/test_integration.py
def test_dev_run_completes():
    """Full pipeline on 1% data finishes without error."""

def test_training_reduces_loss():
    """Loss decreases over 5 epochs (sanity check)."""

def test_metrics_non_zero():
    """NDCG and Recall > 0 after training."""

def test_checkpoint_save_load():
    """Can save and restore model checkpoints."""

def test_reproducibility():
    """Same seed produces same results."""
```

### Smoke Tests (Quick validation)
Run before full experiments:

```bash
# Quick sanity checks (~2 min)
pytest tests/ -m "smoke" --tb=short

# Full test suite (~10 min)
pytest tests/ -v
```

### Lightning-specific Tests
```python
def test_lightning_trainer_fit():
    """Trainer.fit() completes on tiny dataset."""

def test_lightning_callbacks():
    """EarlyStopping, ModelCheckpoint work correctly."""

def test_lightning_logging():
    """Metrics logged to tensorboard/wandb."""

def test_multi_gpu_ddp():
    """DDP strategy works (if multi-GPU available)."""
```

---

## Expected Results

### Ablation 1: Embedding Strategy
| Strategy | Search NDCG@10 | Rec NDCG@10 | Notes |
|----------|----------------|-------------|-------|
| Search-only | **Best** | Poor | Task-specific |
| Rec-only | Poor | **Best** | Task-specific |
| Multi-task | Good | Good | Best trade-off |

### Ablation 2: Discretization
| Method | Search NDCG@10 | Rec NDCG@10 | Notes |
|--------|----------------|-------------|-------|
| RQ-KMeans | **Best** | **Best** | Paper's finding |
| RQ-VAE | Lower | Lower | More complex |
| LSH | Lower | Lower | Fast, less precise |
| PQ | Moderate | Moderate | Standard baseline |

**Key hypothesis to validate**: Multi-task + RQ-KMeans is optimal.

---

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM bi-encoder (A4500 20GB) | Batch size 64-128, gradient accumulation |
| OOM flan-t5-base | Gradient checkpointing, batch 32-64 |
| RQ-KMeans slow | Use `faiss-gpu` with A4500 |
| Qwen3-8B slow | Batch with vLLM, cache to `/app/data/queries/` |
| Full run >5h | Reduce epochs, early stopping, fewer seeds |
| Tests failing | Run `pytest -x` to stop on first failure |

---

## Next Steps

### Phase 1: Setup & Assessment
1. [ ] Assess GRID codebase - identify reusable components
2. [ ] Download MovieLens25M to `/app/data/raw/`
3. [ ] Set up Qwen3-8B for query generation (vLLM)

### Phase 2: Implementation & Unit Tests
4. [ ] Implement data loading + write `test_data/` tests
5. [ ] Implement bi-encoder + write `test_models/` tests
6. [ ] Implement discretization (RQ-KMeans, RQ-VAE, LSH, PQ) + tests
7. [ ] Implement evaluation metrics + tests
8. [ ] Run `pytest tests/` - all unit tests pass

### Phase 3: Dev Run (target: <10 min)
9. [ ] Generate dev queries (~1K items × 2 queries) - cached
10. [ ] Run integration tests (`test_integration.py`)
11. [ ] Train bi-encoder (multi-task only for dev)
12. [ ] Run RQ-KMeans discretization
13. [ ] Train T5 generative model
14. [ ] Evaluate - verify non-zero metrics (sanity check)

### Phase 4: Full Experiments (target: <5 hours)
15. [ ] Generate full queries (62K items × 10 queries) - cached, preprocessing
16. [ ] Run Ablation 1: Search vs Rec vs Multi-task (5 seeds each)
17. [ ] Run Ablation 2: RQ-KMeans vs RQ-VAE vs LSH vs PQ (5 seeds each)
18. [ ] Aggregate results (mean ± std error)
19. [ ] Compare with paper findings

### Dev Run Success Criteria
- All unit tests pass (`pytest tests/`)
- Integration tests pass (`pytest tests/test_integration.py`)
- Pipeline runs end-to-end without errors
- Non-zero NDCG and Recall metrics
- Loss decreases during training

---

## References

- [Paper](https://arxiv.org/abs/2508.10478)
- [GRID Framework](https://github.com/snap-research/GRID)
- [Practitioner's Handbook for Semantic IDs](https://arxiv.org/abs/2507.22224)
