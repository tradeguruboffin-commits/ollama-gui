#!/bin/bash

# 1. Get current system version
CURRENT_VERSION=$(ollama --version 2>&1 | grep -oP '\d+\.\d+\.\d+')

# 2. Get latest release version from GitHub
LATEST_VERSION=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | grep -oP '"tag_name":\s*"v\K\d+\.\d+\.\d+')

echo "Current Version: $CURRENT_VERSION"
echo "Latest Version: $LATEST_VERSION"

# 3. Compare versions
if [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
    echo "New update found! Starting update..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "You are already using the latest version."
fi
