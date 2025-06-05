#!/usr/bin/env python3
"""Quick fix for import issues."""

import os
from pathlib import Path

# Create all __init__.py files
init_files = [
    "src/__init__.py",
    "src/models/__init__.py",
    "src/preprocessing/__init__.py", 
    "src/utils/__init__.py",
    "src/annotation/__init__.py",
    "src/api/__init__.py"
]

for file_path in init_files:
    Path(file_path).touch()
    print(f"Created: {file_path}")

print("✅ All __init__.py files created!")