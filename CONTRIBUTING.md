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
   This installs packages required for development, including
   `pytest-asyncio` for running asynchronous tests.
4. **Set your OpenAI API key** in a `.env` file:
   ```bash
   echo "OPENAI_API_KEY=your-api-key" > .env
   ```

## Running Tests and Checks

Run the full quality assurance suite before committing changes:
```bash
./check.sh
```
This script formats the code, runs the linter, and executes the unit tests via `scripts/check`.

### Network isolation

Unit tests run with network access disabled using `pytest-socket`. If a test
needs network access, mark it with `@pytest.mark.integration` and run it via
`scripts/check integration`.

## Development Guidelines

- Write tests for all code changes whenever feasible, especially when fixing bugs or regressions.
- Use Pydantic models for all API request and response schemas.
- Update documentation alongside code changes.
- Keep `.cursorrules`, `AGENTS.md`, and `CONTRIBUTING.md` synchronized when guidelines change.

### Working with TODO.md

The project uses TODO.md for task tracking with bidirectional GitHub issue synchronization:

- **Completed tasks**: Remove finished tasks from TODO.md entirely. The sync system will automatically close the corresponding GitHub issue.
- **New tasks**: Add new tasks without GitHub issue numbers. Use this format:
  ```
  - **Task title** – Task description
  ```
  Or with priority:
  ```
  - [P2] **Task title** – Task description
  ```
  The sync system will automatically create GitHub issues and add issue numbers to TODO.md.
- **Never manually add issue numbers** like `[#123]` to new tasks. Let the sync system handle GitHub integration.
- **Task organization**: Place new tasks in the appropriate category section and mark as "Next" if they should be prioritized.

## Git Workflow

- Use feature branches for your work and keep them short-lived.
- Follow the commit message convention described in [`.cursorrules`](.cursorrules).
- Ensure all changes pass `./check.sh` before opening a pull request.

For more detailed development guidelines, refer to [`.cursorrules`](.cursorrules).
