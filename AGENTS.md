# Instructions for AI Contributors

This repository welcomes contributions from AI-based tools. When acting as an agent, follow these rules:

1. **Read `.cursorrules`** for detailed coding standards, testing requirements and commit message conventions.
2. **Run `./check.sh`** after making changes. This script formats the code, lints it and runs the unit tests.
3. **Keep commits atomic** and use the conventional commit style described in `.cursorrules`.
4. **Do not introduce external dependencies** without updating `pyproject.toml`.
5. **Write tests for your changes** whenever feasible, especially for bug or regression fixes.
6. **Use Pydantic models** for all API request and response data.
7. **Update documentation** before completing a task.
8. **Never commit secrets or generated files.**
9. **Keep contributor docs aligned** – update `.cursorrules`, `AGENTS.md`, and `CONTRIBUTING.md` together when guidelines change.

For human contributors, see [CONTRIBUTING.md](CONTRIBUTING.md).
