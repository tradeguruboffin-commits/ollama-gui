#!/bin/bash

# ------------------------------------------------------------
# Ollama GUI Update & Environment Setup Script
# ------------------------------------------------------------
# This script:
# 1. Navigates to the Ollama-GUI installation directory
# 2. Checks for an active internet connection
# 3. Updates the Ollama-GUI source code from GitHub (if online)
# 4. Runs the Ollama-GUI environment setup script
# ------------------------------------------------------------


# First of all start DataBase
echo "🐘 Checking PostgreSQL status..."

if /usr/lib/postgresql/17/bin/pg_isready &> /dev/null; then
    echo "✅ PostgreSQL is already running."
else
    echo "🚀 PostgreSQL is down. Starting now..."

    /usr/lib/postgresql/17/bin/pg_ctl -D /home/debian/projects/postgres/data -l /home/debian/projects/logfile start
    
    sleep 3
fi

# Navigate to the Ollama-GUI directory
cd /opt/ollama-gui || {
    echo "Error: Ollama-GUI directory not found."
    exit 1
}

# Check for active internet connectivity (using Google as a reliable endpoint)
if ping -c 1 google.com &> /dev/null; then
    echo "🌐 Internet connection detected. Updating Ollama-GUI..."

    # Pull the latest changes from the GitHub repository
    # Note: Local changes are uncommon in this directory, but consider stashing if needed.
    # Example:
    #   git stash push -m "Pre-update stash"

#    git pull

    # If stashing was used, it can be restored with:
    #   git stash pop

    # Run the environment startup script after update
    /opt/ollama-gui/ollama-gui-startup.sh
else
    echo "⚠️ No internet connection detected. Skipping Ollama-GUI update."

    # Even if the update is skipped, ensure the environment is initialized
    /opt/ollama-gui/ollama-gui-startup.sh
fi

exit 0
