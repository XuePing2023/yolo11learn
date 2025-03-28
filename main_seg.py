from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
    # Output all label results and their confidence rates
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = result.names[cls]
        print(f"Label: {label}, Confidence: {conf:.2f}")
    # Plot the result
    annotated_frame = result.plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
