
from ultralytics import YOLO, settings
import cv2
from PIL import Image
import random

rand_num = random.random()
# View all settings
print(settings)
# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

im1 = Image.open("bus.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

#[0.0, 1.0)

# from ndarray
# print(results)
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(result.to_json())
    print(result.verbose())
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk
im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
print(results)
