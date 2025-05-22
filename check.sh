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

# Run tests (excluding integration tests)
run_check "python tests/run_tests.py" "Running tests (excluding integration tests)"

# Format code (excluding tests)
run_check "ruff format src/ --line-length 88" "Formatting code"

# Run linter for main code
run_check "ruff check src/rag --fix --line-length 88" "Linting main code"

# Run format again to ensure any auto-fixes are properly formatted
run_check "ruff format src/ --line-length 88" "Re-formatting code after linting"

# Final report
if [ $OVERALL_STATUS -eq 0 ]; then
  echo "✨ All checks passed successfully! ✨"
else
  echo "⚠️  Some checks failed. Please fix the issues and try again. ⚠️"
fi

exit $OVERALL_STATUS 
