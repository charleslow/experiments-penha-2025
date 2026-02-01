# RunPod Setup Instructions

## RunPod Template Settings

- **Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container Start Command**:
  ```
bash -c 'if [ ! -d /workspace/experiments-penha-2025 ]; then git clone https://github.com/charleslow/experiments-penha-2025.git /workspace/experiments-penha-2025; fi && cd /workspace/experiments-penha-2025 && git pull && bash setup.sh && sleep infinity'
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
