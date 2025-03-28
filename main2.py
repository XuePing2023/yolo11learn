import cv2 as cv
import numpy as np

from ultralytics.utils.plotting import Annotator, colors

names = {  
    0: "person",
    5: "bus",
    11: "stop sign",
}

image = cv.imread("bus.jpg")
ann = Annotator(
    image,
    line_width=None,  # default auto-size
    font_size=None,  # default auto-size
    font="Arial.ttf",  # must be ImageFont compatible
    pil=False,  # use PIL, otherwise uses OpenCV
)

xyxy_boxes = np.array(
    [
        [5, 22.878, 231.27, 804.98, 756.83],  # class-idx x1 y1 x2 y2
        [0, 48.552, 398.56, 245.35, 902.71],
        [0, 669.47, 392.19, 809.72, 877.04],
        [0, 221.52, 405.8, 344.98, 857.54],
        [0, 0, 550.53, 63.01, 873.44],
        [11, 0.0584, 254.46, 32.561, 324.87],
    ]
)

for nb, box in enumerate(xyxy_boxes):
    c_idx, *box = box
    label = f"{str(nb).zfill(2)}:{names.get(int(c_idx))}"
    ann.box_label(box, label, color=colors(c_idx, bgr=True))

image_with_bboxes = ann.result()
# Save the annotated image to a file
output_path = "annotated_bus.jpg"
cv.imwrite(output_path, image_with_bboxes)