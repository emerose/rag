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

# Run all checks
echo "🔍 Starting code quality checks..."

# Run tests
run_check "uv run pytest" "Running tests"

# Format code
run_check "uv run ruff format . --line-length 88" "Formatting code"

# Run linter for main code
run_check "uv run ruff check src/rag --fix --line-length 88" "Linting main code"

# Run linter for test code
run_check "uv run ruff check tests --fix --line-length 88 --ignore S101,SLF001" "Linting test code"

# Run format again to ensure any auto-fixes are properly formatted
run_check "uv run ruff format . --line-length 88" "Re-formatting code after linting"

# Run mypy strict type checks on tests
run_check "uv run mypy --strict tests/" "Running type checks on tests"

# Final report
if [ $OVERALL_STATUS -eq 0 ]; then
  echo "✨ All checks passed successfully! ✨"
else
  echo "⚠️  Some checks failed. Please fix the issues and try again. ⚠️"
fi

exit $OVERALL_STATUS 
