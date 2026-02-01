#!/bin/bash
set -e

REPO_URL="https://github.com/charleslow/experiments-penha-2025.git"
REPO_DIR="/workspace/experiments-penha-2025"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Git config
git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
git config --global credential.helper 'cache --timeout=604800'

# Clone if not exists, otherwise pull
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repository exists, pulling latest..."
    cd "$REPO_DIR"
    git pull
fi

cd "$REPO_DIR"

# Clone GRID repo if not exists
if [ ! -d "$REPO_DIR/GRID" ]; then
    echo "Cloning GRID repository..."
    git clone https://github.com/snap-research/GRID.git "$REPO_DIR/GRID"
fi

# Install Node.js if not present (needed for Claude Code)
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

# Install Claude Code CLI
if ! command -v claude &> /dev/null; then
    echo "Installing Claude Code..."
    npm install -g @anthropic-ai/claude-code
fi

# Persist Claude credentials across pod restarts (symlink to workspace)
if [ ! -L ~/.claude ]; then
    mkdir -p /workspace/.claude
    rm -rf ~/.claude
    ln -s /workspace/.claude ~/.claude
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv pip install --system --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match -r requirements.txt

echo ""
echo "Setup complete!"
echo "On first git push, enter your GitHub username and Personal Access Token as password."
echo "Credentials will be cached for 1 week."
