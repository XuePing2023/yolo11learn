from ultralytics import YOLO
import cv2
import math

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/boats.jpg")  # predict on an image

# Access the results
for result in results:
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
    confs = result.obb.conf  # confidence score of each box
    
    # Calculate angle for each OBB
    for i, box in enumerate(xyxyxyxy):
        # Get the first two points (top-left and top-right)
        p1, p2 = box[0], box[1]
        
        # Calculate the angle in radians
        angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        
        # Convert to degrees
        angle_deg = math.degrees(angle_rad)
        
        print(f"Object {i}: Class={names[i]}, Confidence={confs[i]:.2f}, Angle={angle_deg:.2f}°")

    # Plot the result
    annotated_frame = result.plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()