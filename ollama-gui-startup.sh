#!/bin/bash

# ------------------------------------------------------------
# Ollama Server & GUI Startup Script
# ------------------------------------------------------------
# This script:
# 1. Navigates to the project directory
# 2. Activates the Python virtual environment
# 3. Checks whether the Ollama server is already running
# 4. Starts the Ollama server if it is not running
# 5. Launches the Ollama GUI application
# ------------------------------------------------------------

# Move to the home directory (kept for consistency with the original script)
cd ~ || exit 1

# Navigate to the project directory
cd /opt/ollama-gui || {
    echo "Error: Project directory not found."
    exit 1
}

# Activate the Python virtual environment
if [ -f "ollama-env/bin/activate" ]; then
    source ollama-env/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# Check if the Ollama server is already running
if pgrep -x "ollama" > /dev/null; then
    echo "Ollama server is already running. Skipping startup."
else
    echo "Starting Ollama server..."
    
    # Start Ollama server in the background
    ollama serve &
    
    # Wait for the server to initialize
    sleep 5
fi

# Start the Ollama GUI application
echo "Launching Ollama GUI..."
/opt/ollama-gui/ollama-gui-runner.sh

# ------------------------------------------------------------
# To stop the Ollama server manually, use:
# killall ollama
# ------------------------------------------------------------
