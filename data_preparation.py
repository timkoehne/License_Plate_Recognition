from PIL import Image
from PIL import ImageOps

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


def generate_flipped_images(
    image: Image.Image, letter: str):
    return_images = []
    
    # vertical mirror
    if letter in CHARACTERS_TO_MIRROR_VERTICAL:
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return_images.append(flipped)

    # horizontal mirror
    if letter in CHARACTERS_TO_MIRROR_HORIZONTAL:
        flipped = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return_images.append(flipped)

    # vertical and horizontal mirror
    if letter in CHARACTERS_TO_MIRROR_BOTH:
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped = flipped.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return_images.append(flipped)
    return return_images


def generate_negative_image(image: Image.Image):
    inverted_image = ImageOps.invert(image)
    return inverted_image