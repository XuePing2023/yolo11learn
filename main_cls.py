from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n-cls.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

for result in results:
    
    # Output all label results and their confidence rates
    # for box in result.boxes:
    #     cls = int(box.cls[0])
    #     conf = float(box.conf[0])
    #     label = result.names[cls]
    #     print(f"Label: {label}, Confidence: {conf:.2f}")
    # Plot the result
    annotated_frame = result.plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()