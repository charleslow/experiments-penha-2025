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
