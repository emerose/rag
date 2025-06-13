#!/bin/bash

# Set up strict error handling
set -o errexit  # Exit immediately if a command fails
set -o pipefail # Exit if any command in a pipeline fails
set -o nounset  # Treat unset variables as an error

# Make sure we're using the right Python
if [[ -d ".venv" ]]; then
  echo "📦 Using Python from .venv environment"
  source .venv/bin/activate
else
  echo "⚠️ No virtual environment found at .venv/"
  exit 1
fi

# Show Python version for debugging
python --version

# Delegate to the new check command
exec python scripts/check 
