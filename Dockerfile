# syntax=docker/dockerfile:1
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy setup script
COPY setup.sh /setup.sh
RUN chmod +x /setup.sh

CMD ["sleep", "infinity"]
