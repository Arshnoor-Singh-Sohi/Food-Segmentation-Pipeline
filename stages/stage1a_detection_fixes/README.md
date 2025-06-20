# Stage 1A: Detection Accuracy Fixes

## Objectives
This stage addresses the immediate detection issues identified:

1. **False positive bottle detection** - CEO feedback: "too many bottles count"
2. **Banana quantity counting errors** - Showing "whole on 3" instead of individual count
3. **Item classification accuracy** - General improvements
4. **Portion vs Complete Dish display** - Clear classification display

## Files in This Stage

- `detection_fixer.py` - Main detection improvement system
- `1a_runner.py` - Stage-specific runner  
- `config.yaml` - Stage 1A configuration
- `README.md` - This documentation

## How to Run Stage 1A

### From Project Root (Recommended):
```bash
# Test refrigerator scenario (main focus)
python run_stage.py 1a --refrigerator

# Test specific image
python run_stage.py 1a --image data/input/refrigerator.jpg

# Run all Stage 1A tests
python run_stage.py 1a --test-all
```

### Direct Execution (Alternative):
```bash
# From project root
cd stages/stage1a_detection_fixes
python 1a_runner.py --refrigerator
```

## Expected Outputs

- **Visualization**: Before/after comparison showing raw vs enhanced detections
- **JSON Report**: Detailed analysis with item counts and classifications
- **Console Summary**: Clear feedback on fixes applied

## Success Criteria

✅ Bottle false positives reduced (fewer bottles detected)
✅ Banana counting shows individual bananas correctly
✅ Food type classification displays clearly (COMPLETE DISH vs INDIVIDUAL ITEMS)
✅ Overall detection accuracy improved

## Files Created

Results are saved in: `data/output/stage1a_results/`

- `{image_name}_stage1a_{timestamp}.json` - Detailed analysis
- `{image_name}_stage1a_fixes_{timestamp}.png` - Visual comparison

## Next Stage

Once Stage 1A is working satisfactorily, proceed to:
**Stage 1B: Display Formatting Fixes**
