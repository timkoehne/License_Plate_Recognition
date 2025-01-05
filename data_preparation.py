from typing import Tuple
from PIL import Image
import cv2
import numpy as np

from darknet_python import darknet

CHARACTERS_TO_MIRROR_HORIZONTAL = [
    "0",
    "1",
    "3",
    "8",
    "B",
    "C",
    "D",
    "E",
    "H",
    "I",
    "K",
    "O",
    "X",
]
CHARACTERS_TO_MIRROR_VERTICAL = [
    "0",
    "1",
    "8",
    "A",
    "H",
    "I",
    "M",
    "O",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]
CHARACTERS_TO_MIRROR_BOTH = [
    "0",
    "1",
    "6",
    "8",
    "9",
    "H",
    "I",
    "N",
    "O",
    "S",
    "X",
    "Z",
]


def generate_flipped_images(cv_image, letter: str):
    return_images = []
    
    # vertical mirror
    if letter in CHARACTERS_TO_MIRROR_VERTICAL:
        flipped = cv2.flip(cv_image, 0)
        return_images.append(flipped)

    # horizontal mirror
    if letter in CHARACTERS_TO_MIRROR_HORIZONTAL:
        flipped = cv2.flip(cv_image, 1)
        return_images.append(flipped)

    # vertical and horizontal mirror
    if letter in CHARACTERS_TO_MIRROR_BOTH:
        flipped = cv2.flip(cv_image, -1)
        return_images.append(flipped)
    return return_images


def generate_negative_image(cv_image):
    inverted_image = cv2.bitwise_not(cv_image)
    return inverted_image


def load_cv2_image(filename):
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def load_darknet_image(cv2_image, width: int, height: int):
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, cv2_image.tobytes())
    return darknet_image

def load_subimage(image, center_x: float, center_y: float, width: float, height: float):
    x = int(center_x - width/2)
    y = int(center_y - height/2)
    width = int(width)
    height = int(height)
    image = image[y:y+height, x:x+width]
    return image


def save_image(cv_image, filepath: str, width: int, height: int):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, cv_image)
