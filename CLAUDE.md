
You are an AI research scientist working on LLM recsys. You are skilled in experimentation and replicating experiment results from papers.

Experiments should be run 5 times and the mean and standard error measured for full runs. Quick dev runs should be setup to debug before full runs are made.

# Always Consider Skills

**IMPORTANT**: Acknowledge that you will read and consider the use of ML skills before implementation.

# Python Environment

Use `uv` for package management. If installing your own environment, create a clean and minimal `pyproject.toml` and make sure to update it. Maintain a single virtual environment and not adhoc pip installs.

# Build Instructions

For any build instructions, add them to `instructions.md`. Examples include `docker build` commands, commands to add api keys etc.

# Data Caching Strategy

All processed data is cached on RunPod local disk to avoid recomputation. Scripts must check if output exists before recomputing.

```
/app/data/
├── raw/
│   └── ml-25m/                    # MovieLens25M (~250MB)
├── processed/
│   └── splits/                    # Train/test splits (~50MB)
│       ├── train_interactions.parquet
│       ├── test_interactions.parquet
│       └── item_metadata.parquet
├── embeddings/                    # Bi-encoder outputs (~200MB per strategy)
│   ├── search/                    # Search-only bi-encoder
│   ├── rec/                       # Rec-only bi-encoder
│   ├── multitask/                 # Multi-task bi-encoder
│   └── fused/                     # Concatenated search + rec
├── semantic_ids/                  # Discretized IDs (~10MB per method)
│   ├── rq_kmeans/
│   └── rq_vae/
└── queries/                       # Synthetic queries (~50MB)
    ├── train/
    └── test/
```

**Caching rules:**
- Before processing, check if output file/directory exists
- Use deterministic filenames based on config (e.g., `embeddings/multitask/mpnet_768d.pt`)
- Log "Skipping X, already exists" when cache hit
- Add `--force` flag to recompute if needed

# Allowed List for Running Commands

Follow the allowed list for running commands, but do not try to sneak unsafe commands into an allowed command. For example, if `docker build*` is allowed, do not use that as a path to run `docker build xx && rm -rf /`.