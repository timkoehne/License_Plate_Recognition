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
    
    if letter in CHARACTERS_TO_MIRROR_HORIZONTAL:
        flipped = cv2.flip(cv_image, 0)
        return_images.append(flipped)

    if letter in CHARACTERS_TO_MIRROR_VERTICAL:
        flipped = cv2.flip(cv_image, 1)
        return_images.append(flipped)

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


def adjust_bb(bb: Tuple[float, float, float, float], original_image_width: float, original_image_height: float, target_image_width: float, target_image_height: float):
    img_ratio = original_image_width / original_image_height
    target_ratio = target_image_width / target_image_height

    if img_ratio > target_ratio:
        new_width = target_image_width
        new_height = int(target_image_width / img_ratio)
    else:
        new_height = target_image_height
        new_width = int(target_image_height * img_ratio)

    width_ratio = new_width / original_image_width
    height_ratio = new_height / original_image_height

    left_pad = (target_image_width - new_width) // 2
    top_pad = (target_image_height - new_height) // 2
    
    center_x, center_y, width, height = bb
    new_center_x = center_x * width_ratio + left_pad
    new_center_y = center_y * height_ratio + top_pad
    new_width_box = width * width_ratio
    new_height_box = height * height_ratio

    return (new_center_x, new_center_y, new_width_box, new_height_box)


def resize_image(cv2_image, target_width: float, target_height: float):
    target_width = int(target_width)
    target_height = int(target_height)
    
    img_height, img_width = cv2_image.shape[:2]
    img_ratio = img_width / img_height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * img_ratio)
    
    resized_img = cv2.resize(cv2_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    result_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top_pad = (target_height - new_height) // 2
    left_pad = (target_width - new_width) // 2

    result_img[top_pad:top_pad + new_height, left_pad:left_pad + new_width] = resized_img
    return result_img

def save_image(cv_image, filepath: str, width: int, height: int):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv_image = resize_image(cv_image, width, height)
    cv2.imwrite(filepath, cv_image)
    
def draw_centered_box(image, center_x, center_y, width, height, color=(0, 255, 0), thickness=2):
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)