import cv2 as cv
from pathlib import Path
import sys
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1] # Vision
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.abspath(ROOT))
ROI_IMG = str(ROOT) + '/images/ROI_table.png'


def color_pixel(x,y):
    # @Description Function to show graphically the line of the height used to estimate block poses
    # @Parameters  x and y coordinates (pixels) to draw in the image

    img = cv.imread(ROI_IMG, cv.IMREAD_COLOR)   # opens ROI image)

    img_copy[y][x] = np.array([0, 0, 255])      # writes in red
    cv.imshow("Image", img_copy)
    cv.waitKey(0)
    cv.imwrite(str(ROOT)+"/images/line-height.png", img_copy)

if __name__ == '__main__':
    color_pixel(938, 634)