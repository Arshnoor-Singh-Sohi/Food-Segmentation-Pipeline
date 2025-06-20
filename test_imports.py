import sys
import os
from pathlib import Path

# Add genai_system to path
genai_path = Path(__file__).parent / "genai_system"
sys.path.insert(0, str(genai_path))

print(f"Looking for files in: {genai_path}")
print(f"Path exists: {genai_path.exists()}")

# List files
if genai_path.exists():
    files = list(genai_path.glob("*.py"))
    print(f"Python files found: {files}")
    
    for file in files:
        size = file.stat().st_size
        print(f"{file.name}: {size} bytes")

# Try importing
try:
    import genai_analyzer
    print("✅ genai_analyzer imported successfully")
except Exception as e:
    print(f"❌ genai_analyzer error: {e}")

try:
    import ceo_demo
    print("✅ ceo_demo imported successfully")
except Exception as e:
    print(f"❌ ceo_demo error: {e}")