#!/usr/bin/env python3
"""API diff tool for comparing public APIs between git references.

Usage:
    python scripts/diff_api.py <package_name> <git_ref>
    python scripts/diff_api.py rag main
    python scripts/diff_api.py rag HEAD~5

This tool creates a diff showing the changes to the public API of a Python package
between the current working tree and any git reference.
"""

import difflib
import importlib
import inspect
import pkgutil
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

# ANSI color codes
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[94m"


class APIDumper:
    """Extracts and dumps the public API of a Python package."""

    def __init__(self, package_name: str, root_path: Path):
        """Initialize the API dumper.

        Args:
            package_name: Name of the package to analyze (e.g., 'rag')
            root_path: Root path where the package is located
        """
        self.package_name = package_name
        self.root_path = root_path

    def dump(self) -> str:
        """Dump the public API of the package.

        Returns:
            String representation of the public API

        Raises:
            ImportError: If the package cannot be imported
            Exception: For other package inspection errors
        """
        # Temporarily add the root path to sys.path for importing
        original_path = sys.path[:]
        sys.path.insert(0, str(self.root_path))

        # Clear any existing modules from the package to ensure fresh imports
        modules_to_clear = [
            mod for mod in sys.modules.keys() if mod.startswith(self.package_name)
        ]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Clear import caches
        if hasattr(importlib, "invalidate_caches"):
            importlib.invalidate_caches()

        # Clear any finder caches
        if hasattr(sys, "path_importer_cache"):
            sys.path_importer_cache.clear()

        try:
            return self._walk_package()
        except ImportError as e:
            raise ImportError(f"Failed to import package '{self.package_name}': {e}")
        except Exception as e:
            raise Exception(f"Error analyzing package '{self.package_name}': {e}")
        finally:
            # Restore original sys.path
            sys.path[:] = original_path

    def _walk_package(self) -> str:
        """Walk through the package and extract public API."""
        try:
            package = importlib.import_module(self.package_name)
        except ImportError as e:
            return f"# Error: Could not import package '{self.package_name}': {e}"

        seen: set[str] = set()
        lines = self._dump_module(package, seen)

        # Walk submodules if this is a package
        if hasattr(package, "__path__"):
            for _, modname, _ in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."
            ):
                if modname in seen:
                    continue
                try:
                    mod = importlib.import_module(modname)
                    lines.extend(self._dump_module(mod, seen))
                except Exception as e:
                    lines.append(f"# {modname}: Error importing ({e})")

        return "\n".join(lines)

    def _dump_module(self, module: ModuleType, seen: set[str]) -> list[str]:
        """Dump the public API of a single module."""
        lines = [f"## {module.__name__}"]
        seen.add(module.__name__)

        # Extract public functions
        for name, obj in sorted(inspect.getmembers(module, inspect.isfunction)):
            if self._is_public(name) and obj.__module__ == module.__name__:
                lines.append(self._dump_function(f"{module.__name__}.{name}", obj))

        # Extract public classes
        for name, obj in sorted(inspect.getmembers(module, inspect.isclass)):
            if self._is_public(name) and obj.__module__ == module.__name__:
                lines.extend(self._dump_class(f"{module.__name__}.{name}", obj))

        return lines

    def _dump_function(self, fqname: str, func: object) -> str:
        """Dump a function signature and description."""
        try:
            # Get the original signature
            original_sig = inspect.signature(func)

            # Create a clean signature without default values to avoid memory addresses
            clean_params = []
            for param in original_sig.parameters.values():
                # Create new parameter without default value
                if param.annotation != inspect.Parameter.empty:
                    # Keep the type annotation
                    clean_param = param.replace(default=inspect.Parameter.empty)
                else:
                    # No annotation, just the name
                    clean_param = param.replace(default=inspect.Parameter.empty)
                clean_params.append(clean_param)

            # Create new signature with clean parameters
            clean_sig = original_sig.replace(parameters=clean_params)
            sig = str(clean_sig)
        except Exception:
            sig = "(...)"

        doc = inspect.getdoc(func)
        first_line = doc.strip().splitlines()[0] if doc else ""
        return f"- `{fqname}{sig}`: {first_line}"

    def _dump_class(self, fqname: str, cls: object) -> list[str]:
        """Dump a class and its public methods."""
        lines = [f"### class {fqname}"]

        # Get public methods defined in this class
        methods = [
            (name, method)
            for name, method in inspect.getmembers(cls, inspect.isfunction)
            if self._is_public(name) and method.__module__ == cls.__module__
        ]

        if not methods:
            lines.append("- `pass`")
        else:
            for name, method in sorted(methods):
                lines.append(self._dump_function(f"{fqname}.{name}", method))

        return lines

    @staticmethod
    def _is_public(name: str) -> bool:
        """Check if a name represents a public API element."""
        return not name.startswith("_")


class GitWorktree:
    """Context manager for creating temporary git worktrees."""

    def __init__(self, ref: str):
        """Initialize the git worktree manager.

        Args:
            ref: Git reference to checkout (branch, tag, commit hash)
        """
        self.ref = ref
        self.path = Path(tempfile.mkdtemp(prefix="api_diff_"))

    def __enter__(self) -> Path:
        """Create the git worktree and return its path."""
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(self.path), self.ref],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return self.path
        except subprocess.CalledProcessError as e:
            # Clean up the directory if git worktree creation failed
            shutil.rmtree(self.path, ignore_errors=True)
            error_msg = e.stderr.strip() if e.stderr else "Unknown error"
            raise RuntimeError(
                f"Failed to create git worktree for '{self.ref}': {error_msg}"
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the git worktree."""
        try:
            # Remove the git worktree
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(self.path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            # If git worktree remove fails, still try to clean up the directory
            pass
        finally:
            # Always attempt to remove the directory
            shutil.rmtree(self.path, ignore_errors=True)


class APIDiffRenderer:
    """Renders colorized diffs between API dumps."""

    def __init__(self, base: str, current: str, base_ref: str):
        """Initialize the diff renderer.

        Args:
            base: Base API dump (old version)
            current: Current API dump (new version)
            base_ref: Name of the base reference for labeling
        """
        self.base = base.strip().splitlines(keepends=True)
        self.current = current.strip().splitlines(keepends=True)
        self.base_ref = base_ref

    def render(self) -> str:
        """Render a colorized unified diff."""
        diff = difflib.unified_diff(
            self.base,
            self.current,
            fromfile=f"{self.base_ref} (old)",
            tofile="HEAD (new)",
            lineterm="",
        )

        return self._colorize_diff(diff)

    def _colorize_diff(self, diff_lines) -> str:
        """Apply ANSI color codes to diff lines."""
        output = []
        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                output.append(f"{GREEN}{line}{RESET}")
            elif line.startswith("-") and not line.startswith("---"):
                output.append(f"{RED}{line}{RESET}")
            elif line.startswith("@@"):
                output.append(f"{YELLOW}{line}{RESET}")
            else:
                output.append(line)
        return "\n".join(output)


def check_git_repository() -> None:
    """Check if we're in a git repository."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"{RED}Error: Not in a git repository{RESET}")
        sys.exit(1)


def check_git_ref_exists(ref: str) -> None:
    """Check if a git reference exists."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        print(f"{RED}Error: Git reference '{ref}' does not exist{RESET}")
        sys.exit(1)


def run_api_dump_subprocess(package: str, root_path: Path) -> str:
    """Run API dump in a separate subprocess for complete isolation."""
    # Create a temporary Python script to run the API dump
    script_content = f'''
import sys
# Keep essential Python paths but prioritize the target directory
original_paths = sys.path[:]
sys.path = ["{root_path}"] + [p for p in original_paths if not p.endswith("/src") and "{root_path}" not in p]

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import List, Set

def is_public(name: str) -> bool:
    return not name.startswith("_")

def dump_function(fqname: str, func: object) -> str:
    try:
        original_sig = inspect.signature(func)
        clean_params = []
        for param in original_sig.parameters.values():
            clean_param = param.replace(default=inspect.Parameter.empty)
            clean_params.append(clean_param)
        clean_sig = original_sig.replace(parameters=clean_params)
        sig = str(clean_sig)
    except Exception:
        sig = "(...)"
    
    doc = inspect.getdoc(func)
    first_line = doc.strip().splitlines()[0] if doc else ""
    return f"- `{{fqname}}{{sig}}`: {{first_line}}"

def dump_class(fqname: str, cls: object) -> List[str]:
    lines = [f"### class {{fqname}}"]
    methods = [
        (name, method) for name, method in inspect.getmembers(cls, inspect.isfunction)
        if is_public(name) and method.__module__ == cls.__module__
    ]
    if not methods:
        lines.append("- `pass`")
    else:
        for name, method in sorted(methods):
            lines.append(dump_function(f"{{fqname}}.{{name}}", method))
    return lines

def dump_module(module: ModuleType, seen: Set[str]) -> List[str]:
    lines = [f"## {{module.__name__}}"]
    seen.add(module.__name__)
    
    for name, obj in sorted(inspect.getmembers(module, inspect.isfunction)):
        if is_public(name) and obj.__module__ == module.__name__:
            lines.append(dump_function(f"{{module.__name__}}.{{name}}", obj))
    
    for name, obj in sorted(inspect.getmembers(module, inspect.isclass)):
        if is_public(name) and obj.__module__ == module.__name__:
            lines.extend(dump_class(f"{{module.__name__}}.{{name}}", obj))
    
    return lines

def walk_package() -> str:
    try:
        package = importlib.import_module("{package}")
    except ImportError as e:
        return f"# Error: Could not import package '{package}': {{e}}"
    
    seen: Set[str] = set()
    lines = dump_module(package, seen)
    
    if hasattr(package, '__path__'):
        for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
            if modname in seen:
                continue
            try:
                mod = importlib.import_module(modname)
                lines.extend(dump_module(mod, seen))
            except Exception as e:
                lines.append(f"# {{modname}}: Error importing ({{e}})")
    
    return "\\n".join(lines)

print(walk_package())
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", script_content],
            capture_output=True,
            text=True,
            check=True,
            cwd=root_path,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.strip() if e.stderr else "Unknown error"
        raise RuntimeError(f"Failed to analyze API: {error_output}")


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print(f"{BLUE}Usage: python {sys.argv[0]} <package_name> <git_ref>{RESET}")
        print(f"{BLUE}Example: python {sys.argv[0]} rag main{RESET}")
        sys.exit(1)

    package = sys.argv[1]
    ref = sys.argv[2]

    print(f"{BLUE}🔍 Diffing public API of `{package}` against `{ref}`...{RESET}\n")

    # Validate git environment
    check_git_repository()
    check_git_ref_exists(ref)

    try:
        # Dump current (HEAD) API
        print(f"{YELLOW}📊 Analyzing current API...{RESET}")
        current_api = run_api_dump_subprocess(package, Path.cwd())

        # Dump base (target ref) API
        print(f"{YELLOW}📊 Analyzing API at {ref}...{RESET}")
        with GitWorktree(ref) as tmpdir:
            base_api = run_api_dump_subprocess(package, tmpdir)

        # Render and show diff
        print(f"{YELLOW}🔄 Computing diff...{RESET}\n")
        diff = APIDiffRenderer(base_api, current_api, ref).render()

        if not diff.strip():
            print(
                f"{GREEN}✅ No public API differences between HEAD and `{ref}`.{RESET}"
            )
        else:
            print(diff)

    except ImportError as e:
        print(f"{RED}❌ Import Error: {e}{RESET}")
        print(f"{YELLOW}💡 Make sure the package is installed and importable.{RESET}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"{RED}❌ Git Error: {e}{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}❌ Unexpected Error: {e}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
