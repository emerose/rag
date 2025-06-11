#!/usr/bin/env python3
"""API diff tool for comparing public APIs between git references.

Usage:
    python scripts/diff_api.py <package_name> <git_ref>
    python scripts/diff_api.py rag main
    python scripts/diff_api.py rag HEAD~5

This tool creates a diff showing the changes to the public API of a Python package
between the current working tree and any git reference.
"""

import importlib
import inspect
import os
import pkgutil
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from rich.console import Console
from rich.text import Text


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
    """Renders the diff between two API dumps using Rich formatting."""

    def __init__(self, base: str, current: str, base_ref: str):
        """Initialize the diff renderer.

        Args:
            base: Base API dump string
            current: Current API dump string
            base_ref: Git reference name for the base
        """
        self.base = base
        self.current = current
        self.base_ref = base_ref
        self.console = Console()

    def render(self) -> str:
        """Render the API diff with Rich formatting."""
        base_api = self._parse_api_dump(self.base)
        current_api = self._parse_api_dump(self.current)

        # Find additions, removals, and modifications
        base_items = set(base_api.keys())
        current_items = set(current_api.keys())

        added = current_items - base_items
        removed = base_items - current_items
        common = base_items & current_items
        modified = {item for item in common if base_api[item] != current_api[item]}

        if not (added or removed or modified):
            return ""  # No differences

        # Build rich output
        output_parts = []

        # Show removals
        if removed:
            for item in sorted(removed):
                text = Text()
                text.append("- ", style="red bold")
                text.append(base_api[item], style="red")
                output_parts.append(text)

        # Show additions
        if added:
            for item in sorted(added):
                text = Text()
                text.append("+ ", style="green bold")
                text.append(current_api[item], style="green")
                output_parts.append(text)

        # Show modifications
        if modified:
            for item in sorted(modified):
                # Show old version
                text_old = Text()
                text_old.append("- ", style="red bold")
                text_old.append(base_api[item], style="red")
                output_parts.append(text_old)

                # Show new version
                text_new = Text()
                text_new.append("+ ", style="green bold")
                text_new.append(current_api[item], style="green")
                output_parts.append(text_new)

        # Render all parts to string
        with self.console.capture() as capture:
            for part in output_parts:
                self.console.print(part)

        return capture.get()

    def _parse_api_dump(self, api_dump: str) -> dict[str, str]:
        """Parse an API dump string into a dictionary of API items.

        Returns:
            Dict mapping fully qualified names to their descriptions
        """
        api_items = {}
        lines = api_dump.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("- `") and "`:" in line:
                # Extract function/method signature
                # Format: - `fully.qualified.name(args) -> return`: Description
                try:
                    # Find the closing backtick for the signature
                    end_backtick = line.find("`:")
                    if end_backtick != -1:
                        signature = line[
                            3:end_backtick
                        ]  # Extract between "- `" and "`:"
                        # Use the full line as the value for comparison
                        api_items[signature] = line
                except Exception:
                    # If parsing fails, use the whole line as both key and value
                    api_items[line] = line

        return api_items


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
        console = Console()
        console.print("Error: Not in a git repository", style="red")
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
        console = Console()
        console.print(f"Error: Git reference '{ref}' does not exist", style="red")
        sys.exit(1)


def run_api_dump_subprocess(package: str, root_path: Path) -> str:
    """Run API dump in a separate subprocess for complete isolation."""
    # Create a temporary Python script to run the API dump
    script_content = f'''
import sys
from pathlib import Path
# Clear sys.path completely and only add essential paths plus target directory
essential_paths = [p for p in sys.path if any(essential in p for essential in ['python3.', 'lib-dynload', '.zip'])]
src_path = Path("{root_path}") / "src"
if src_path.exists():
    sys.path = [str(src_path)] + essential_paths
else:
    sys.path = ["{root_path}"] + essential_paths

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
            env={**os.environ, "PYTHONPATH": ""},  # Clear PYTHONPATH
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.strip() if e.stderr else "Unknown error"
        raise RuntimeError(f"Failed to analyze API: {error_output}")


def main():
    """Main entry point."""
    console = Console()

    if len(sys.argv) < 3:
        console.print(
            f"Usage: python {sys.argv[0]} <package_name> <git_ref>", style="blue"
        )
        console.print(f"Example: python {sys.argv[0]} rag main", style="blue")
        sys.exit(1)

    package = sys.argv[1]
    ref = sys.argv[2]

    console.print(
        f"🔍 Diffing public API of `{package}` against `{ref}`...", style="blue"
    )
    console.print()

    # Validate git environment
    check_git_repository()
    check_git_ref_exists(ref)

    try:
        # Dump current (HEAD) API
        console.print("📊 Analyzing current API...", style="yellow")
        current_api = run_api_dump_subprocess(package, Path.cwd())

        # Dump base (target ref) API
        console.print(f"📊 Analyzing API at {ref}...", style="yellow")
        with GitWorktree(ref) as tmpdir:
            base_api = run_api_dump_subprocess(package, tmpdir)

        # Render and show diff
        console.print("🔄 Computing diff...", style="yellow")
        console.print()
        diff = APIDiffRenderer(base_api, current_api, ref).render()

        if not diff.strip():
            console.print(
                f"✅ No public API differences between HEAD and `{ref}`.", style="green"
            )
        else:
            print(diff)

    except ImportError as e:
        console.print(f"❌ Import Error: {e}", style="red")
        console.print(
            "💡 Make sure the package is installed and importable.", style="yellow"
        )
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"❌ Git Error: {e}", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Unexpected Error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
