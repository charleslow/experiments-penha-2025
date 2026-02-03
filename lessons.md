# Lessons Learned

## Hatchling Requires Explicit Source Directory Configuration

When using hatchling as a build backend, you must explicitly specify where source files are located. Without `[tool.hatch.build.targets.wheel] packages = ["src"]` in pyproject.toml, the build will fail to find modules.

## Always Move Tensors to the Same Device Explicitly

Device mismatches (CUDA/CPU) occur frequently in ML code. Don't assume tensors are on the same device. Use `.to(reference_tensor.device)` patterns throughout:
- Labels should be moved to the same device as model outputs
- Centroids in clustering algorithms must match input data device
- Evaluation tensors should all be moved to CPU for consistency

## FAISS Has Strict Sample Size Requirements

FAISS PQ (Product Quantization) requires `n_samples >= codebook_size`. When working with small datasets or tests, either reduce the codebook size or skip PQ entirely. Add guards to check this constraint before training.

## Container Environments May Lack Basic Shell Utilities

Cloud environments like RunPod may not have common utilities (head, tail) in PATH. Don't assume shell commands exist. Use Python or tool-specific alternatives when possible.

## SentenceTransformer.encode() Drops Gradients

The `SentenceTransformer.encode()` method doesn't preserve gradients - it's designed for inference. For training with backpropagation, implement a custom `encode_with_grad()` method that tokenizes inputs and runs the model forward pass directly.

## Multi-task Training Produces Separate Validation Metrics

When training with multiple dataloaders (e.g., search + rec), Lightning creates separate metrics like `val/loss/dataloader_idx_0` and `val/loss/dataloader_idx_1` instead of a single `val/loss`. Adjust early stopping and checkpointing to monitor the appropriate metric.

## Redirect Cache Directories on Cloud Instances

Cloud instances often have small root filesystems that fill up with HuggingFace/torch caches. Set environment variables to redirect caches to network storage:
```bash
HF_HOME=/workspace/.cache/huggingface
TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
TORCH_HOME=/workspace/.cache/torch
```

## Sample Large Pair Datasets Before Training

Co-occurrence datasets can explode combinatorially (11.6M pairs for MovieLens-25M). Add `max_pairs` parameters to dataset classes and sample to manageable sizes (e.g., 500K train, 100K val) or training becomes infeasible.

## Early Stop on Task-Specific Metrics, Not Loss

Training loss doesn't correlate well with retrieval performance. Use callbacks that compute actual task metrics (e.g., Recall@30) for early stopping. This saves compute and better reflects real performance.

## Evaluation Queries Should Not Contain Target Names

For meaningful retrieval evaluation, queries should be broad descriptions (e.g., "90s sci-fi about simulated reality") not trivial matches (e.g., "Find The Matrix"). Queries containing exact titles give artificially high R@30 (99%+) that doesn't reflect real search quality.

## Bi-Encoder vs Generative Retrieval Performance

Bi-encoder retrieval (continuous embeddings) is typically much higher than generative retrieval (discretized semantic IDs). This is expected because continuous embeddings are more expressive. When replicating papers, check which model the reported numbers refer to.

## Structure Experiments in Phases for Efficient Iteration

1. **Dev Run**: Quick pipeline validation with synthetic/small data (< 10 min)
2. **Single Seed**: Validate metrics are in ballpark of paper targets (3-4 hours)
3. **Full Statistical Run**: 5 seeds for mean ± std error (10-15 hours)

This catches bugs early before committing to expensive full runs.

## Bi-Encoder Training Only Needs One Seed

The bi-encoder creates deterministic embeddings once trained. Variance in final results comes from discretization (k-means initialization) and generative model training (random initialization). Train bi-encoder once per strategy, then run multiple seeds only for the stochastic stages.
