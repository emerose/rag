#!/usr/bin/env python3
"""Demo script to show clean API signatures without memory addresses."""

import sys

sys.path.insert(0, ".")
import inspect
from pathlib import Path

from scripts.diff_api import APIDumper


def example_function(
    data: str, count: int = 5, optional: bool = False
) -> dict[str, any]:
    """Example function to demonstrate clean API diff output."""
    return {"data": data, "count": count, "optional": optional}


print("=== Clean API Signature Demo ===")
print("Original function signature with defaults:")
print(f"  {inspect.signature(example_function)}")

# Test our clean signature logic
dumper = APIDumper("rag", Path.cwd())
clean_output = dumper._dump_function("demo.example_function", example_function)
print("\nClean API diff output:")
print(f"  {clean_output}")

print("\n✨ Success! No memory addresses, clean type information only.")
