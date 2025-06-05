from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model and image
model = YOLO('yolov8n.pt')
image = cv2.imread('data/input/image1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
results = model(image_rgb)

# Show results
results[0].show()
print("✅ YOLO detection working!")