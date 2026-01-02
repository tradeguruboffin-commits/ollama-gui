#!/bin/bash

set -e

# Check ollama
if ! command -v ollama >/dev/null; then
    echo "❌ ollama not installed"
    exit 1
fi

# Current version
CURRENT_VERSION=$(ollama --version 2>/dev/null | awk '{print $NF}')

# Latest version
LATEST_VERSION=$(curl -fsSL https://api.github.com/repos/ollama/ollama/releases/latest \
  | grep '"tag_name"' | head -1 | cut -d '"' -f4 | sed 's/^v//')

if [ -z "$CURRENT_VERSION" ] || [ -z "$LATEST_VERSION" ]; then
    echo "❌ Version detection failed"
    exit 1
fi

echo "Current Version: $CURRENT_VERSION"
echo "Latest Version : $LATEST_VERSION"

if [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
    echo "⬆️ Updating ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Already up to date"
fi
