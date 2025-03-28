import cv2 as cv
import numpy as np

from ultralytics.utils.plotting import Annotator, colors

obb_names = {10: "small vehicle"}
obb_image = cv.imread("datasets/dota8/images/train/P1142__1024__0___824.jpg")
obb_boxes = np.array(
    [
        [0, 635, 560, 919, 719, 1087, 420, 803, 261],  # class-idx x1 y1 x2 y2 x3 y2 x4 y4
        [0, 331, 19, 493, 260, 776, 70, 613, -171],
        [9, 869, 161, 886, 147, 851, 101, 833, 115],
    ]
)
ann = Annotator(
    obb_image,
    line_width=None,  # default auto-size
    font_size=None,  # default auto-size
    font="Arial.ttf",  # must be ImageFont compatible
    pil=False,  # use PIL, otherwise uses OpenCV
)
for obb in obb_boxes:
    c_idx, *obb = obb
    obb = np.array(obb).reshape(-1, 4, 2).squeeze()
    label = f"{obb_names.get(int(c_idx))}"
    ann.box_label(
        obb,
        label,
        color=colors(c_idx, True),
        rotated=True,
    )

image_with_obb = ann.result()