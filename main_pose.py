from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    
    annotated_frame = result.plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()