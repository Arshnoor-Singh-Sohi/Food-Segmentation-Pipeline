from src.models.fast_segmentation import FastFoodSegmentation

# Process your image FAST
segmenter = FastFoodSegmentation()
results = segmenter.process_image('data/input/image1.jpg')
segmenter.visualize(results)