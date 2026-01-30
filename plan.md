# Replication Plan: Semantic IDs for Joint Generative Search and Recommendation

**Paper**: [Penha et al. 2025](https://arxiv.org/abs/2508.10478) - RecSys 2025

## Summary

This paper investigates how to construct Semantic IDs that work well for **both search and recommendation** in a unified generative model. Key finding: **multi-task bi-encoder** + **RQ-KMeans** discretization provides the best trade-off.

**Key Contributions:**
1. Compares task-specific vs cross-task Semantic ID construction strategies
2. Shows RQ-KMeans outperforms RQ-VAE for Semantic ID construction
3. Demonstrates multi-task bi-encoder embeddings enable effective joint search & recommendation

---

## Pipeline Overview

```
1. DATA PREPARATION
   └── MovieLens25M + Synthetic Queries (Gemini-2.0-flash)

2. EMBEDDING GENERATION (Bi-encoder)
   ├── Search-only (query-item pairs)
   ├── Rec-only (item-item co-occurrence)
   ├── Multi-task (joint training) ← BEST
   └── Fused (concatenate search + rec)

3. SEMANTIC ID CONSTRUCTION
   ├── RQ-KMeans (FAISS) ← BEST
   └── RQ-VAE (baseline)

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
| **Queries per item** | 2 | 20 | |
| **Contrastive batch** | 64 | 256 | In-batch negatives |
| **Generative batch** | 32 | 128 | With grad accumulation |
| **Learning rate** | 1e-4 | 5e-5 | With warmup |
| **Epochs** | 5 | 50 | Early stopping |
| **GPU** | RTX 3090 | A100 80GB | |

---

## Dataset

| Component | Details |
|-----------|---------|
| Base | MovieLens25M |
| Items | 62,138 movies |
| Interactions | 1.24M user-item pairs |
| Split | Chronological (last interaction per user for test) |
| Queries | 20 per item (10 train / 10 test), Gemini-2.0-flash |

---

## Experimental Conditions

### Embedding Strategies (Table 1 in paper)

| Strategy | Description | Token Budget |
|----------|-------------|--------------|
| Search | Bi-encoder on query-item pairs only | 1x |
| Rec | Bi-encoder on item-item co-occurrence only | 1x |
| Multi-task | Bi-encoder jointly trained on both | 1x |
| Fused | Concatenate Search + Rec embeddings | 1x |
| Separate | Task-specific Semantic IDs | 2x |

### Discretization Methods

| Method | Description |
|--------|-------------|
| RQ-KMeans | FAISS residual quantizer (recommended) |
| RQ-VAE | Variational autoencoder approach |

---

## GRID Modifications

GRID is designed for recommendation only. Files to add:

```
GRID/src/
├── models/biencoder.py        # Bi-encoder for embeddings
├── training/contrastive.py    # Contrastive learning
├── training/multitask.py      # Multi-task training
├── data/movielens.py          # MovieLens data loader
├── data/query_pairs.py        # Query-item pairs
└── evaluation/search_metrics.py  # Search NDCG, Recall
```

---

## Expected Results

| Strategy | Search NDCG@10 | Rec NDCG@10 |
|----------|----------------|-------------|
| Search-only | **Best** | Poor |
| Rec-only | Poor | **Best** |
| Multi-task | Good | Good |
| Fused | Moderate | Moderate |
| Separate | Good | Good |

**Key finding**: Multi-task provides best trade-off without doubling token budget.

---

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM bi-encoder | Reduce batch size, gradient accumulation |
| OOM flan-t5 | Gradient checkpointing, reduce seq length |
| RQ-KMeans slow | Use `faiss-gpu` |
| Query gen expensive | Cache API responses, batch calls |

---

## Next Steps

1. [ ] Download MovieLens25M
2. [ ] Add bi-encoder training code to GRID
3. [ ] Generate synthetic queries (subset for dev)
4. [ ] Run dev experiment to validate pipeline
5. [ ] Run full ablation (5 strategies × 2 methods × 5 seeds)
6. [ ] Analyze and compare with paper

---

## References

- [Paper](https://arxiv.org/abs/2508.10478)
- [GRID Framework](https://github.com/snap-research/GRID)
- [Practitioner's Handbook for Semantic IDs](https://arxiv.org/abs/2507.22224)
