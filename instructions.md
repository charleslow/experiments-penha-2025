# RunPod Setup Instructions

## RunPod Template Settings

- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container Start Command**:

```bash
bash -lc '
cat >/post_start.sh <<'"'"'EOF'"'"'
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/workspace/experiments-penha-2025
REPO_URL=https://github.com/charleslow/experiments-penha-2025.git

mkdir -p /workspace

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

if ! id claude-user &>/dev/null; then
  useradd -m -s /bin/bash claude-user
fi

cd "$REPO_DIR"
git pull --ff-only
bash setup.sh
EOF

chmod +x /post_start.sh
exec /start.sh
'
```
- **Environment Variables** (optional):
  - `GIT_USER_NAME`: Your name for git commits
  - `GIT_USER_EMAIL`: Your email for git commits

The container will automatically:
1. Clone the repo (first time only)
2. Pull latest changes
3. Run setup.sh (installs uv + dependencies)

## After SSH-ing In

Just `cd /workspace/experiments-penha-2025` and start working.

## Git Credentials

Credentials are cached for 7 days. On first push, you'll be prompted:

```bash
git push  # Enter GitHub username and Personal Access Token as password
```

---

## AI Research Skills Installation

AI research skills provide expert guidance for common ML/AI tasks. Skills are installed as Claude Code plugins.

### Step 1: Add the Marketplace

```bash
/plugin marketplace add orchestra-research/AI-research-SKILLs
```

### Step 2: Install by Category

```bash
/plugin install distributed-training@ai-research-skills
/plugin install rag@ai-research-skills
/plugin install mlops@ai-research-skills
/plugin install fine-tuning@ai-research-skills
/plugin install optimization@ai-research-skills
/plugin install evaluation@ai-research-skills
```

### View All Available Skills

See `references.md` for the complete list of 82 available skills organized by 20 categories.

### Skills Source

- Local clone: `AI-research-SKILLs/`
- Documentation: https://www.orchestra-research.com/perspectives/ai-research-skills

---

## Semantic ID Replication Project

### Installation

```bash
cd /workspace/experiments-penha-2025

# Install dependencies using uv
uv sync

# Install with dev dependencies (for testing)
uv sync --extra dev
```

### Download Data

```bash
# Download MovieLens-25M dataset
python scripts/download_data.py --output-dir /app/data/raw
```

### Run Dev Experiment

```bash
# Run quick dev experiment (< 10 min)
python scripts/run_dev.py --output-dir results/dev_run

# With custom data fraction
python scripts/run_dev.py --data-fraction 0.01 --output-dir results/dev_run
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Generate Queries

```bash
# Generate synthetic queries (fast, no LLM)
python scripts/generate_queries.py --dev

# Generate queries with LLM (requires GPU)
python scripts/generate_queries.py --model Qwen/Qwen2.5-3B-Instruct
```

### Project Structure

```
src/
├── config.py                 # Configuration dataclasses
├── data/                     # Data loading and processing
│   ├── movielens.py          # MovieLens dataset
│   ├── query_generator.py    # Query generation
│   └── datamodule.py         # Lightning DataModule
├── models/                   # Model implementations
│   ├── bi_encoder.py         # Contrastive bi-encoder
│   └── generative.py         # T5 generative model
├── discretization/           # Embedding discretization
│   ├── base.py               # Abstract interface
│   ├── rq_kmeans.py          # RQ-KMeans
│   ├── lsh.py                # LSH
│   └── pq.py                 # Product Quantization
├── evaluation/               # Metrics
│   └── metrics.py            # NDCG, Recall, etc.
├── visualization/            # Plotting
│   └── plots.py              # Result visualization
└── utils/                    # Utilities
    ├── cache.py              # Artifact caching
    └── seed.py               # Reproducibility

scripts/
├── download_data.py          # Download MovieLens
├── generate_queries.py       # Query generation
└── run_dev.py                # Dev run entry point

tests/
├── unit/                     # Fast unit tests
└── integration/              # Full pipeline tests
```
