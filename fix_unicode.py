#!/usr/bin/env python3
"""
Fix Unicode characters in Python files for Windows compatibility
Run this once to replace all emoji with ASCII alternatives
"""

import os
import glob

def fix_unicode_in_file(filepath):
    """Fix Unicode characters in a Python file"""
    replacements = {
        '[TOOL]': '[TOOL]',
        '[LOG]': '[LOG]',
        '[TIME]': '[TIME]',
        '[FOLDER]': '[FOLDER]',
        '[FILE]': '[FILE]',
        '[OK]': '[OK]',
        '[FAIL]': '[FAIL]',
        '[STEP]': '[STEP]',
        '[RUN]': '[RUN]',
        '[TIP]': '[TIP]',
        '[INFO]': '[INFO]',
        '[ERROR]': '[ERROR]',
        '[WARN]': '[WARN]',
        '[SUCCESS]': '[SUCCESS]',
        '[FOOD]': '[FOOD]',
        '[STAR]': '[STAR]',
        '[STAR]': '[STAR]',
        '[FIRE]': '[FIRE]',
        '[STRONG]': '[STRONG]',
        '[TARGET]': '[TARGET]',
        '[CLOCK]': '[CLOCK]',
        '[TIMER]': '[TIMER]',
        '[DATE]': '[DATE]',
        '[STATS]': '[STATS]',
        '[TEST]': '[TEST]',
        '[DONE]': '[DONE]',
        '[STOP]': '[STOP]'
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply replacements
        for unicode_char, ascii_replacement in replacements.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        # Write back with safe encoding
        with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
            f.write(content)
        
        print(f"[OK] Fixed: {filepath}")
        return True
    except Exception as e:
        print(f"[FAIL] Error fixing {filepath}: {e}")
        return False

def main():
    """Fix Unicode in all Python files"""
    python_files = glob.glob("*.py") + glob.glob("**/*.py", recursive=True)
    
    print("Fixing Unicode characters in Python files...")
    fixed_count = 0
    
    for filepath in python_files:
        if fix_unicode_in_file(filepath):
            fixed_count += 1
    
    print(f"\n[SUCCESS] Fixed {fixed_count} files!")

if __name__ == "__main__":
    main()
    