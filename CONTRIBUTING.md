# Contributing to RAG

Thank you for your interest in improving this project! This guide explains how to set up your development environment, run tests, and follow the repository workflow.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag
   ```
2. **Create a virtual environment** using [uv](https://github.com/astral-sh/uv):
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```
4. **Set your OpenAI API key** in a `.env` file:
   ```bash
   echo "OPENAI_API_KEY=your-api-key" > .env
   ```

## Running Tests and Checks

Run the full quality assurance suite before committing changes:
```bash
./check.sh
```
This script formats the code, runs the linter, and executes the unit tests via `tests/run_tests.py`.

## Git Workflow

- Use feature branches for your work and keep them short-lived.
- Follow the commit message convention described in [`.cursorrules`](.cursorrules).
- Ensure all changes pass `./check.sh` before opening a pull request.

For more detailed development guidelines, refer to [`.cursorrules`](.cursorrules).
