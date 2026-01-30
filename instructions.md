# Docker Setup Instructions

## Build the Docker Image

Replace `yourusername` with your DockerHub username:

```bash
docker build -t yourusername/penha-2025:latest .
```

## Push to DockerHub

1. Login to DockerHub (one time):
   ```bash
   docker login
   ```

2. Push the image:
   ```bash
   docker push yourusername/penha-2025:latest
   ```

## Pull and Run (on RunPod or other cloud)

```bash
docker pull yourusername/penha-2025:latest
docker run --gpus all -it yourusername/penha-2025:latest
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
