#!/bin/bash

# Set up strict error handling
set -o errexit  # Exit immediately if a command fails
set -o pipefail # Exit if any command in a pipeline fails
set -o nounset  # Treat unset variables as an error

# Initialize overall status
OVERALL_STATUS=0

# Function to run a command and track its status
run_check() {
  local cmd="$1"
  local description="$2"
  
  echo "➡️  Running: $description"
  if eval "$cmd"; then
    echo "✅ Passed: $description"
    return 0
  else
    echo "❌ Failed: $description"
    OVERALL_STATUS=1
    return 1
  fi
}

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

# Run all checks
echo "🔍 Starting code quality checks..."

# Run linting first
run_check "python scripts/run_tests.py lint" "Running linting and formatting"

# Run tests in order: unit → integration → e2e
if [ $OVERALL_STATUS -eq 0 ]; then
  run_check "python scripts/run_tests.py unit" "Running unit tests"
fi

if [ $OVERALL_STATUS -eq 0 ]; then
  run_check "python scripts/run_tests.py integration" "Running integration tests"
fi

if [ $OVERALL_STATUS -eq 0 ]; then
  run_check "python scripts/run_tests.py e2e" "Running e2e tests"
fi

# Final report
if [ $OVERALL_STATUS -eq 0 ]; then
  echo "✨ All checks passed successfully! ✨"
else
  echo "⚠️  Some checks failed. Please fix the issues and try again. ⚠️"
fi

exit $OVERALL_STATUS 
