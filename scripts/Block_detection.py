from pathlib import Path
import sys
import os
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from IPython.display import display
from PIL import Image
import Region_of_interest as roi
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
import json

# --------------------- DIRECTORIES and PATHS ---------------------
ROOT = Path(__file__).resolve().parents[1]  # vision folder
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.abspath(ROOT))

IMG_PATH = str(ROOT) + '/images/ROI_table.png'
IMG_ZED = str(ROOT) + '/images/img_ZED_cam.png'
IMG_BLOCK_ISOLATED_PATH = str(ROOT) + 'images/BLOCK.png'

# --------------------- CONSTANTS ---------------------
debug = False
WEIGHTS = str(ROOT) + "/weights/best.pt"
CONFIDENCE = 0.5
yolo_model = YOLO(WEIGHTS) # YOLO model trained to detect lego pieces

SAVE_PREDICTION_TXT = False
SAVE_PREDICTION_IMG = True

LEGO_LABELS =  ['X1-Y1-Z2',
                'X1-Y2-Z1',
                'X1-Y2-Z2',
                'X1-Y2-Z2-CHAMFER',
                'X1-Y2-Z2-TWINFILLET',
                'X1-Y3-Z2',
                'X1-Y3-Z2-FILLET',
                'X1-Y4-Z1',
                'X1-Y4-Z2',
                'X2-Y2-Z2',
                'X2-Y2-Z2-FILLET']


class LegoBlock:

    def __init__(self, label, confidence, x1, y1, x2, y2):
        self.label = label
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.img = Image.open(IMG_PATH)
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        self.point_cloud_coord = []


    def info(self):
        # @Description Function that writes some info about the block
        print("\nBlock label: " + self.label)
        print("\tconfidence: " + str(round(self.confidence, 3)))
        print("\tTop-left corner: (" + str(int(self.x1)) + ", " + str(int(self.y1)) + ")")
        print("\tBottom-right corner: (" + str(int(self.x2)) + ", " + str(int(self.y2)) + ")")


def store_blocks(data):
    # @Description takes blocks data from json object from YOLO detection to create a list of blocks
    # @Parameters JSON/dictionary object
    # @Returns a list of Blocks
    blocks = []

    for lego in data:
        current_block = LegoBlock(lego["name"], lego["confidence"],
                                  lego["box"]["x1"], lego["box"]["y1"],
                                  lego["box"]["x2"], lego["box"]["y2"])

        current_block = find_point_cloud_roi(current_block)

        blocks.append(current_block)

    return blocks


def find_point_cloud_roi(block):
    cropped_block_img = get_bbox_image(block)

    # Converti l'immagine in formato HSV
    hsv_image = cv2.cvtColor(cropped_block_img, cv2.COLOR_BGR2HSV)

    # Definisci l'intervallo di colore per il grigio (in formato HSV)
    lower_gray = np.array([0, 0, 50])  # Cambia questi valori se necessario
    upper_gray = np.array([180, 50, 200])  # Cambia questi valori se necessario

    # Crea una maschera per isolare il colore grigio
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)

    # Inverti la maschera del grigio
    gray_mask_inv = cv2.bitwise_not(gray_mask)

    # Applica la maschera invertita all'immagine originale per rimuovere il grigio
    result = cv2.bitwise_and(cropped_block_img, cropped_block_img, mask=gray_mask_inv)

    block.point_cloud_coord = np.column_stack(np.where(gray_mask_inv > 0)).tolist()

    block.point_cloud_coord = block.point_cloud_coord + np.array([int(block.x1), int(block.y1)])

    # Mostra i risultati
    if debug:
        cv2.imshow('Original Image', cropped_block_img)
        cv2.imshow('Gray Mask', gray_mask_inv)
        cv2.imshow('Non-Gray Image', result)

        # Salva l'immagine risultante se necessario
        cv2.imwrite(str(ROOT) + '/images/block_isolated.png', result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return block


def get_bbox_image(block):
    # Leggi l'immagine
    image = cv2.imread(IMG_PATH)
    if image is None:
        raise ValueError("Immagine non trovata al percorso specificato")

    top_left = (block.x1, block.y1)
    bottom_right = (block.x2, block.y2)

    # Calcola gli altri due vertici
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    # Crea una lista dei vertici in senso orario
    vertices = [top_left, top_right, bottom_right, bottom_left]

    # Converti i vertici in un array numpy
    pts = np.array(vertices, dtype=np.float32)

    # Determina il rettangolo di destinazione
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype=np.float32)

    # Calcola la matrice di trasformazione
    M = cv2.getPerspectiveTransform(pts, dst)

    # Applica la trasformazione prospettica
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    global debug

    if debug:
        cv2.imwrite(str(ROOT)+'/images/cropped_block.png', warped)

    return warped


def print_block_info(blocks):
    # @Description prints list of blocks info
    # @Parameters List of Block objects

    for block in blocks:
        block.info()


def detection(img_path):
    # @Description It crops the ZED-cam image into the Region of Interest (only the table), block detection with YOLO follows
    # @Parameters path of the input image (from ZED camera)
    # @Returns the list of detected Blocks

    roi.find_roi(img_path)

    print("Detecting...")
    results = yolo_model.predict(source=roi.OUTPUT_FILE, line_width=1, conf=0.5, save_txt=SAVE_PREDICTION_TXT, save=SAVE_PREDICTION_IMG)

    for result in results:
        data = json.loads(result.tojson())

    blocks = store_blocks(data)

    return blocks


# CLI:  python detection.py /path/to/img.smth
if __name__ == '__main__':

    if len(sys.argv) > 1:   # img has been passed via CLI
        img = sys.argv[1]
    else:
        img = roi.INPUT_FILE

    block_list = detection(img)

    print_block_info(block_list)