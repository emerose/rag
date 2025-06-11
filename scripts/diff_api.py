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
from rich.table import Table
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
        self.console = Console(width=120, soft_wrap=True)

    def render(self) -> str:
        """Render the API diff with Rich formatting in tabular format."""
        base_structure = self._parse_api_structure(self.base)
        current_structure = self._parse_api_structure(self.current)

        # Find all modules/classes that have changes
        all_containers = set(base_structure.keys()) | set(current_structure.keys())
        containers_with_changes = []

        for container in all_containers:
            base_items = set(base_structure.get(container, {}).keys())
            current_items = set(current_structure.get(container, {}).keys())

            added = current_items - base_items
            removed = base_items - current_items
            common = base_items & current_items
            modified = {
                item
                for item in common
                if base_structure.get(container, {}).get(item)
                != current_structure.get(container, {}).get(item)
            }

            if added or removed or modified:
                containers_with_changes.append((container, added, removed, modified))

        if not containers_with_changes:
            return ""  # No differences

        # Build rich output organized by container
        output_parts = []

        for container, added, removed, modified in sorted(containers_with_changes):
            # Add container header
            if container.startswith("class "):
                header = Text(f"## {container}", style="white")
            else:
                header = Text(f"# {container}", style="white")
            output_parts.append(header)

            # Create table for this container
            table = Table(
                show_header=False,
                show_edge=False,
                pad_edge=False,
                padding=(0, 1),
                box=None,
            )
            table.add_column("prefix", width=2, no_wrap=True)  # +/-
            table.add_column("name", min_width=8, no_wrap=True)  # function name
            table.add_column("params", min_width=12)  # parameters (one per line)
            table.add_column("arrow", width=4, no_wrap=True, justify="center")  # ->
            table.add_column("return_type", min_width=8, no_wrap=True)  # return type
            table.add_column("description", min_width=15)  # description

            # Process all changes for this container
            all_changes = []

            # Process removals
            for item in sorted(removed):
                full_line = base_structure[container][item]
                name, params, return_type, description = self._parse_function_for_table(
                    item, full_line, container
                )
                all_changes.append(("-", name, params, return_type, description, "red"))

            # Process additions
            for item in sorted(added):
                full_line = current_structure[container][item]
                name, params, return_type, description = self._parse_function_for_table(
                    item, full_line, container
                )
                all_changes.append(
                    ("+", name, params, return_type, description, "green")
                )

            # Process modifications
            for item in sorted(modified):
                # Old version
                full_line = base_structure[container][item]
                name, params, return_type, description = self._parse_function_for_table(
                    item, full_line, container
                )
                all_changes.append(("-", name, params, return_type, description, "red"))

                # New version
                full_line = current_structure[container][item]
                name, params, return_type, description = self._parse_function_for_table(
                    item, full_line, container
                )
                all_changes.append(
                    ("+", name, params, return_type, description, "green")
                )

            # Add rows to table
            for prefix, name, params, return_type, description, color in all_changes:
                table.add_row(
                    Text(prefix, style=f"{color} bold"),
                    Text(name, style=color),
                    Text(params, style=color),
                    Text("->", style=color),
                    Text(return_type, style=color),
                    Text(description, style=color),
                )

            # Render table to capture
            with self.console.capture() as capture:
                self.console.print(table)

            output_parts.append(Text(capture.get().rstrip()))

            # Add empty line after each container (except the last)
            if container != sorted(containers_with_changes)[-1][0]:
                output_parts.append(Text())

        # Render all parts to string
        with self.console.capture() as capture:
            for part in output_parts:
                self.console.print(part)

        return capture.get()

    def _parse_function_for_table(
        self, signature: str, full_line: str, container: str
    ) -> tuple[str, str, str, str]:
        """Parse function for tabular display with one parameter per line.

        Args:
            signature: The full function signature
            full_line: The full API line
            container: The container name (module or class)

        Returns:
            Tuple of (name, parameters, return_type, description)
        """
        # Extract description from full line
        colon_idx = full_line.find("`: ")
        description = full_line[colon_idx + 3 :] if colon_idx != -1 else ""

        # Parse the signature to extract parts
        arrow_idx = signature.find(" -> ")
        if arrow_idx != -1:
            func_and_params = signature[:arrow_idx]
            return_type = signature[arrow_idx + 4 :]
        else:
            func_and_params = signature
            return_type = ""

        # Extract function name and parameters
        paren_idx = func_and_params.find("(")
        if paren_idx != -1:
            full_name = func_and_params[:paren_idx]
            params_str = func_and_params[paren_idx + 1 : -1]  # Remove ( and )
        else:
            full_name = func_and_params
            params_str = ""

        # Get short name based on container
        if container.startswith("class "):
            class_name = container[6:]  # Remove "class " prefix
            if full_name.startswith(class_name + "."):
                short_name = full_name[len(class_name) + 1 :]
            else:
                short_name = full_name.split(".")[-1]
        # Module function
        elif full_name.startswith(container + "."):
            short_name = full_name[len(container) + 1 :]
        else:
            short_name = full_name.split(".")[-1]

        # Format parameters one per line
        if params_str.strip():
            # Split parameters and clean them up
            params = []
            current_param = ""
            paren_depth = 0
            bracket_depth = 0

            for char in params_str:
                if char == "," and paren_depth == 0 and bracket_depth == 0:
                    if current_param.strip():
                        params.append(current_param.strip())
                    current_param = ""
                else:
                    if char == "(":
                        paren_depth += 1
                    elif char == ")":
                        paren_depth -= 1
                    elif char == "[":
                        bracket_depth += 1
                    elif char == "]":
                        bracket_depth -= 1
                    current_param += char

            # Don't forget the last parameter
            if current_param.strip():
                params.append(current_param.strip())

            # Join parameters with newlines for one-per-line display
            formatted_params = "\n".join(params)
        else:
            formatted_params = ""

        return short_name, formatted_params, return_type, description

    def _parse_api_structure(self, api_dump: str) -> dict[str, dict[str, str]]:
        """Parse an API dump into a hierarchical structure organized by module/class.

        Returns:
            Dict mapping container names (modules/classes) to their API items
        """
        structure = {}
        lines = api_dump.strip().split("\n")
        current_container = None

        for line in lines:
            line = line.strip()

            if line.startswith("## ") and not line.startswith("### "):
                # Module header
                current_container = line[3:]  # Remove "## " prefix
                if current_container not in structure:
                    structure[current_container] = {}
            elif line.startswith("### class "):
                # Class header
                current_container = line[4:]  # Remove "### " prefix
                if current_container not in structure:
                    structure[current_container] = {}
            elif line.startswith("- `") and "`:" in line and current_container:
                # Function/method line
                try:
                    # Find the closing backtick for the signature
                    end_backtick = line.find("`:")
                    if end_backtick != -1:
                        signature = line[
                            3:end_backtick
                        ]  # Extract between "- `" and "`:"
                        # Store the full line for later rendering
                        structure[current_container][signature] = line
                except Exception:
                    # If parsing fails, use the whole line as both key and value
                    structure[current_container][line] = line

        return structure

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
                f"✅ No public API differences between HEAD and `{ref}`", style="green"
            )
        else:
            console.print(diff)
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
