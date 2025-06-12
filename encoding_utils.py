#!/usr/bin/env python3
"""
Encoding Utilities for Windows Unicode Compatibility
Use this module to fix Unicode encoding issues across your project
"""

import sys
import os
import io

def setup_encoding():
    """Set up proper encoding for Windows compatibility"""
    try:
        # Set console to UTF-8 if possible
        if sys.platform.startswith('win'):
            os.system('chcp 65001 > nul 2>&1')
        
        # Reconfigure stdout/stderr with UTF-8 encoding
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        else:
            # Fallback for older Python versions
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # If UTF-8 setup fails, we'll use ASCII alternatives
        pass

def safe_print(text, use_ascii_fallback=True):
    """Print text with encoding fallback"""
    try:
        print(text)
    except UnicodeEncodeError:
        if use_ascii_fallback:
            # Replace Unicode characters with ASCII alternatives
            ascii_text = convert_to_ascii(text)
            print(ascii_text)
        else:
            print("[ENCODING ERROR - Cannot display special characters]")

def convert_to_ascii(text):
    """Convert Unicode characters to ASCII alternatives"""
    replacements = {
        '[FOOD]': '[FOOD]',
        '[TOOL]': '[TOOL]',
        '🔍': '[SEARCH]',
        '[OK]': '[OK]',
        '[FAIL]': '[FAIL]',
        '[RUN]': '[RUN]',
        '[TIP]': '[TIP]',
        '[TIME]': '[TIME]',
        '[INFO]': '[INFO]',
        '[ERROR]': '[ERROR]',
        '[WARN]': '[WARN]',
        '[SUCCESS]': '[SUCCESS]',
        '[STATS]': '[STATS]',
        '[FOLDER]': '[FOLDER]',
        '[TEST]': '[TEST]',
        '[STEP]': '[REFRESH]',
        '[DONE]': '[DONE]',
        '[TIMER]': '[TIMER]',
        '[DATE]': '[DATE]',
        '[TARGET]': '[TARGET]',
        '[CLOCK]': '[CLOCK]',
        '[STOP]': '[STOP]',
        '🔬': '[SCIENCE]',
        '📈': '[CHART]',
        '💻': '[COMPUTER]',
        '[STAR]': '[STAR]',
        '[FIRE]': '[FIRE]',
        '[STRONG]': '[STRONG]',
        '🎨': '[ART]',
        '🏆': '[TROPHY]',
        '[STAR]': '[SHINE]',
        '🎮': '[GAME]',
        '🎲': '[DICE]',
        '🎪': '[CIRCUS]',
        '🎭': '[MASK]',
        '🎬': '[MOVIE]',
        '🎵': '[MUSIC]',
        '🎤': '[MIC]',
        '🎧': '[HEADPHONE]',
        '🎸': '[GUITAR]',
        '🎹': '[PIANO]',
        '🎺': '[TRUMPET]',
        '🎻': '[VIOLIN]',
        '🥁': '[DRUM]'
    }
    
    result = text
    for unicode_char, ascii_replacement in replacements.items():
        result = result.replace(unicode_char, ascii_replacement)
    
    return result

def safe_write_file(filename, content, encoding='utf-8'):
    """Write file with proper encoding handling"""
    try:
        with open(filename, 'w', encoding=encoding, errors='replace') as f:
            f.write(content)
        return True
    except UnicodeEncodeError:
        # Fallback to ASCII
        ascii_content = convert_to_ascii(content)
        try:
            with open(filename, 'w', encoding='ascii', errors='replace') as f:
                f.write(ascii_content)
            return True
        except Exception as e:
            print(f"[FAIL] Could not write file {filename}: {e}")
            return False
    except Exception as e:
        print(f"[FAIL] Error writing file {filename}: {e}")
        return False

def safe_append_file(filename, content, encoding='utf-8'):
    """Append to file with proper encoding handling"""
    try:
        with open(filename, 'a', encoding=encoding, errors='replace') as f:
            f.write(content)
        return True
    except UnicodeEncodeError:
        # Fallback to ASCII
        ascii_content = convert_to_ascii(content)
        try:
            with open(filename, 'a', encoding='ascii', errors='replace') as f:
                f.write(ascii_content)
            return True
        except Exception as e:
            print(f"[FAIL] Could not append to file {filename}: {e}")
            return False
    except Exception as e:
        print(f"[FAIL] Error appending to file {filename}: {e}")
        return False

# Auto-setup encoding when module is imported
setup_encoding()

# Test function
def test_encoding():
    """Test the encoding utilities"""
    print("Testing encoding utilities...")
    
    # Test safe_print
    safe_print("Testing Unicode: [FOOD][TOOL][OK][FAIL][RUN][TIP]")
    
    # Test file writing
    test_content = "Test file with Unicode: [FOOD][TOOL][OK][FAIL][RUN][TIP]\nLine 2\nLine 3"
    if safe_write_file("test_encoding.txt", test_content):
        print("[OK] File writing test passed")
    else:
        print("[FAIL] File writing test failed")
    
    print("Encoding test complete!")

if __name__ == "__main__":
    test_encoding()