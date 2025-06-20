# Individual Item Detection - Data Preparation Guide

## Overview
To train your model for individual item detection, you need labeled images showing individual ingredients in refrigerator contexts.

## Required Dataset Structure
```
individual_item_dataset/
├── images/
│   ├── train/          # 70% of images
│   ├── val/            # 20% of images  
│   └── test/           # 10% of images
└── labels/
    ├── train/          # YOLO format labels
    ├── val/
    └── test/
```

## Target Classes (28 total)
0: banana_single
1: apple_red
2: apple_green
3: orange_single
4: strawberry_single
5: grape_cluster
6: tomato_single
7: carrot_single
8: lettuce_head
9: broccoli_head
10: bell_pepper
11: cucumber_single
12: bottle_milk
13: bottle_juice
14: bottle_water
15: bottle_soda
16: container_plastic
17: jar_glass
18: can_food
19: package_food
20: box_cereal
21: bag_frozen
22: carton_milk
23: carton_juice
24: egg_single
25: bread_loaf
26: cheese_block
27: yogurt_container

## Labeling Guidelines

### For Individual Items:
- **Each banana gets its own bounding box** (not grouped)
- **Each apple gets its own bounding box** (even if multiple)
- **Each bottle gets its own bounding box with specific type** (milk vs juice)

### YOLO Label Format:
```
class_id x_center y_center width height
```

Example for image with 3 bananas:
```
0 0.2 0.3 0.1 0.15    # banana_single 1
0 0.25 0.35 0.1 0.15  # banana_single 2  
0 0.3 0.4 0.1 0.15    # banana_single 3
```

## Data Collection Strategy

### Option 1: Use Existing Refrigerator Images
1. Collect 500-1000 refrigerator images
2. Label individual items manually
3. Use tools like Roboflow or LabelImg

### Option 2: Synthetic Data Generation
1. Take photos of individual ingredients
2. Composite them into refrigerator scenes
3. Generate automatic labels

### Option 3: Expand Existing Dataset
1. Start with your current images
2. Re-label for individual item detection
3. Add new refrigerator photos

## Recommended Labeling Tools
- **Roboflow**: Online labeling with export to YOLO format
- **LabelImg**: Desktop labeling tool
- **CVAT**: Computer Vision Annotation Tool
- **Supervisely**: Advanced annotation platform

## Quality Guidelines
- **Minimum 50 examples per class**
- **Variety of lighting conditions**
- **Different refrigerator types and contents**
- **Clear individual item boundaries**
- **Consistent labeling across similar items**

## Time Estimate
- **Manual labeling**: 2-4 seconds per bounding box
- **500 images with 10 items each**: ~3-6 hours total
- **Quality review**: Additional 1-2 hours

## Next Steps After Data Preparation
1. Upload dataset to RunPod
2. Run training script: `python train_individual_items.py`
3. Monitor training progress
4. Download enhanced model
5. Test on refrigerator images
