# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements (already pinned from pip-compile)
COPY GRID/requirements.txt ./requirements.txt

# Install dependencies with uv from the pinned requirements
RUN uv pip install --system -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command
CMD ["bash"]
