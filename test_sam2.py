# Test if SAM 2 is working at all

try:
    from sam2.build_sam import build_sam2
    print('✅ SAM 2 imported successfully')
except Exception as e:
    print(f'❌ SAM 2 issue: {e}')
