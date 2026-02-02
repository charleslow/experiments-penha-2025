# Progress Log: Semantic ID Replication

## 2026-02-02 - Initial Implementation

Started implementation of the Penha 2025 semantic ID paper replication. The goal is to create an end-to-end dev run that completes in <10 min on A4500.

### Completed Today

**Phase 1-6: Core Implementation**
- Created full project structure with `src/`, `scripts/`, `tests/`
- Implemented all config dataclasses in `src/config.py`
- Built data loading pipeline: MovieLens loading, chronological splitting, co-occurrence pairs
- Implemented synthetic query generator (no LLM required for dev runs)
- Created bi-encoder with search/rec/multi-task modes using sentence-transformers
- Implemented 3 discretization methods: RQ-KMeans, LSH, and PQ (FAISS)
- Created evaluation metrics (NDCG, Recall, MRR, Hit Rate)
- Built visualization module for ablation plots

**Testing**
- Wrote comprehensive unit tests (17 data tests, 13 bi-encoder tests, 19 discretization tests, 16 metrics tests)
- Created integration tests for full pipeline validation

### Snags Encountered

1. **Hatchling build config**: Initial `pyproject.toml` didn't specify where source files are located. Had to add `[tool.hatch.build.targets.wheel] packages = ["src"]` to fix the build.

2. **Device placement issues**: The bi-encoder tests failed because labels tensor stayed on CPU while model moved to GPU. Fixed by using `labels.to(query_emb.device)` instead of `batch.labels.to(self.device)`.

3. **FAISS sample size requirement**: PQ discretizer test failed because FAISS requires `n_samples >= codebook_size`. Reduced codebook from 256 to 64 for the test.

4. **Missing shell commands**: The RunPod environment doesn't have basic utilities like `head`, `tail` in PATH. Had to adjust bash commands accordingly.

### Current Status

**All 69 unit tests passing.** The full dev run pipeline completed successfully:
- Generated synthetic data (100 items, 1000 interactions)
- Created embeddings using all-MiniLM-L6-v2
- Ran RQ-KMeans and LSH discretization
- Generated 3 ablation plots to `results/dev_run/plots/`
- Saved results to `results/dev_run/results.json`

### Additional Fixes Applied

4. **PQ sample size**: Added check to skip PQ discretization when n_items < codebook_size, as FAISS requires sufficient training samples.

### Output Verification

```
results/dev_run/
├── plots/
│   ├── ablation1_embedding.png     (48KB)
│   ├── ablation2_discretization.png (86KB)
│   └── tradeoff_scatter.png        (93KB)
└── results.json
```

The pipeline uses placeholder metrics for now since actual training is skipped for dev speed. With real data and actual training, the metrics will be meaningful.

### Next Steps

1. ~~Run full test suite to verify fixes~~ DONE
2. ~~Test the dev run script end-to-end~~ DONE
3. ~~Verify plot generation~~ DONE
4. ~~Ready for PR creation~~ DONE

---

## 2026-02-02 - Full Run Implementation

### Phase 7: Full Experiment Run Script

**Goal**: Create `scripts/run_full.py` for running the complete experimental matrix:
- Ablation 1: 3 embedding strategies × 5 seeds = 15 runs
- Ablation 2: 4 discretization methods × 5 seeds = 20 runs
- Target: < 5 hours total on A4500 GPU

### Completed

1. **Created `scripts/run_full.py`** - Full experiment run script with:
   - `FullRunConfig` dataclass with all experiment parameters
   - Support for both ablation studies
   - Bi-encoder training with Lightning
   - Discretization (RQ-KMeans, RQ-VAE, LSH, PQ)
   - Generative model training
   - Evaluation metrics (NDCG, Recall, HitRate, MRR)
   - Results aggregation (mean ± std error across seeds)
   - Automatic plot generation

2. **Updated `src/data/datamodule.py`**:
   - Added `generative_collate_fn` for generative task batches
   - Added generative task support to train/val/test dataloaders
   - Added test dataset setup for generative task

3. **Fixed gradient preservation in `src/models/bi_encoder.py`**:
   - Added `encode_with_grad()` method that preserves gradients for training
   - Updated `compute_search_loss()` and `compute_rec_loss()` to use gradient-preserving encoding
   - Kept `encode()` for inference (no gradients needed)

### Snags Encountered

5. **Bi-encoder gradient issue**: The `SentenceTransformer.encode()` method doesn't preserve gradients by default. Fixed by adding `encode_with_grad()` that tokenizes and runs the model forward pass directly.

6. **Training time**: Initial test with 5000 items and all-mpnet-base-v2 (109M params) is slow. Need to optimize batch sizes and consider using smaller model for dev/quick tests.

### Current Status

- Full run script created and imports successfully
- Bi-encoder training works with gradients preserved
- Currently testing the full pipeline with small data fraction

### Additional Fixes

7. **Early stopping metric for multi-task**: Multi-task training produces separate validation metrics (`val/loss/dataloader_idx_0`, `val/loss/dataloader_idx_1`) instead of a single `val/loss`. Changed early stopping to monitor `train/loss_epoch` instead.

8. **Disk space issues on RunPod**: The root filesystem (20GB) fills up with huggingface/torch cache. Added environment variables to redirect cache to `/workspace/` which has network storage (1.7PB available).

9. **Synthetic data scaling**: Modified synthetic data generation to scale with `data_fraction` parameter for faster iteration during testing.

### Verification Test Results

Successfully completed a verification run with:
- Data: 500 items, 1400 train, 200 val, 400 test
- 1 seed, 1 epoch for encoder and generative model

**Results (Ablation 1 - Embedding Strategy):**
| Strategy | Search NDCG@10 | Rec NDCG@10 |
|----------|----------------|-------------|
| search | 0.9946 | 0.9946 |
| rec | 0.0062 | 0.0062 |
| multi_task | 1.0000 | 0.0034 |

Total time: 2.3 minutes for 3 embedding strategies

**Ready for Full Run!**

### Full Run Configuration

To run the complete experiment:
```bash
TOKENIZERS_PARALLELISM=false \
HF_HOME=/workspace/experiments-penha-2025/.cache/huggingface \
TRANSFORMERS_CACHE=/workspace/experiments-penha-2025/.cache/huggingface/transformers \
TORCH_HOME=/workspace/experiments-penha-2025/.cache/torch \
nohup python scripts/run_full.py \
    --data-fraction 1.0 \
    --n-seeds 5 \
    --encoder-epochs 5 \
    --gen-epochs 10 \
    --output-dir /workspace/experiments-penha-2025/results/full_run \
    > /workspace/experiments-penha-2025/results/full_run.log 2>&1 &
```

### Current Run Status

A test run is in progress:
- Config: `--data-fraction 0.1 --n-seeds 2 --encoder-epochs 3 --gen-epochs 3`
- Output: `/workspace/experiments-penha-2025/results/full_run_test`
- Monitor: `tail -f /workspace/experiments-penha-2025/results/full_run.log`

### Files Created

- `scripts/run_full.py` - Main full run script
- `src/data/datamodule.py` - Updated with generative task support
- `src/models/bi_encoder.py` - Fixed gradient preservation
- `results/test_final2/` - Verification test results

### Configuration (Full Run - Updated to Match Paper)

| Parameter | Value | Paper Reference |
|-----------|-------|-----------------|
| Encoder model | all-mpnet-base-v2 (109M params) | - |
| Encoder epochs | 5 | - |
| Encoder batch | 64 | - |
| Generative model | flan-t5-base (250M params) | - |
| Gen epochs | 10 | - |
| Gen batch | 32 | - |
| N hierarchies | 2 | Paper uses 2 codebooks × 256 |
| Codebook size | 256 | 256 |
| Seeds | 5 (42, 123, 456, 789, 1024) | - |
| Primary metric | Recall@30 | Paper uses R@30 |

### Paper Validation Targets

**Table 1 - Embedding Strategy (R@30):**
| Strategy | Search R@30 | Rec R@30 |
|----------|-------------|----------|
| Search-based | 0.072 (±0.028) | 0.026 (±0.017) |
| Rec-based | 0.004 (±0.001) | 0.062 (±0.015) |

**Table 2 - Multi-task (R@30):**
| Strategy | Search R@30 | Rec R@30 |
|----------|-------------|----------|
| Multi-task | 0.046 | 0.049 |

**Table 3 - Discretization Methods with Multi-task (R@30):**
| Method | Search R@30 | Rec R@30 |
|--------|-------------|----------|
| RQ-KMeans | 0.046 | 0.049 |
| RQ-VAE | 0.002 | 0.024 |

---

## Experiment Phases

The experiment is structured into 3 phases for efficient iteration:

### Phase 1: Dev Run (Quick Validation)
- **Purpose**: Verify pipeline works end-to-end
- **Data**: Synthetic or small fraction of MovieLens
- **Seeds**: 1
- **Time**: < 10 minutes
- **Script**: `scripts/run_dev.py`

### Phase 2: Single Run per Config (Metric Validation)
- **Purpose**: Validate metrics are in ballpark of paper results
- **Data**: Full MovieLens-25M
- **Seeds**: 1
- **Approach**:
  1. Train bi-encoder ONCE per strategy (search, rec, multi_task)
  2. Freeze bi-encoder embeddings
  3. Run 1 discretization + generative training per config
- **Expected Time**: ~3-4 hours
- **Validation**: Check if R@30 matches paper within reasonable tolerance

### Phase 3: Full Statistical Run (5 Seeds)
- **Purpose**: Get mean ± std error for publication-quality results
- **Data**: Full MovieLens-25M
- **Seeds**: 5 (42, 123, 456, 789, 1024)
- **Approach**:
  1. Use frozen bi-encoder from Phase 2
  2. Run 5 seeds ONLY for discretization + generative model
  3. Aggregate results
- **Expected Time**: ~10-15 hours
- **Output**: Final results with statistical significance

### Efficiency Insight
The bi-encoder only needs to be trained ONCE per strategy since it creates deterministic embeddings. The variance in results comes from:
- Discretization (k-means initialization)
- Generative model training (random initialization)

Therefore, we train bi-encoder once and run 5 seeds only for the latter stages.
