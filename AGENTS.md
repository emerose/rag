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

## Working with TODO.md

The project uses TODO.md for task tracking with bidirectional GitHub issue synchronization:

10. **Completed tasks**: Remove finished tasks from TODO.md entirely. The sync system will automatically close the corresponding GitHub issue.
11. **New tasks**: Add new tasks without GitHub issue numbers. Use this format:
    ```
    - **Task title** – Task description
    ```
    Or with priority:
    ```
    - [P2] **Task title** – Task description
    ```
    The sync system will automatically create GitHub issues and add issue numbers to TODO.md.
12. **Never manually add issue numbers** like `[#123]` to new tasks. Let the sync system handle GitHub integration.
13. **Task organization**: Place new tasks in the appropriate category section and mark as "Next" if they should be prioritized.
14. **Link task to PR** by including the issue number in the commit. If a PR completely resolves the issue, include the "closes" keyword: "closes #123". If a commit is just related to an open task, say something like "see #123"

For human contributors, see [CONTRIBUTING.md](CONTRIBUTING.md).
