import glob
import os
import uuid
from PIL import Image, ImageTransform
from PIL import ImageOps
import tqdm

dataset_training = "E:/OpenScience Data/UFPR-ALPR dataset/training"

license_plate_folder = "license_plates/"
characters_folder = "characters/"


CHARACTERS_TO_MIRROR_VERTICAL = [
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
CHARACTERS_TO_MIRROR_HORIZONTAL = [
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


def _create_character_subimages(
    image: Image.Image, license_plate: str, chars: list[tuple[int, int, int, int]]
):
    for index, char in enumerate(chars):
        img = image.crop((char[0], char[1], char[0] + char[2], char[1] + char[3]))

        path = f"{characters_folder}{license_plate[index]}"
        if not os.path.exists(path):
            os.makedirs(path)
        img.save(path + "/" + str(uuid.uuid4()) + ".png")

        # vertical mirror
        if license_plate[index] in CHARACTERS_TO_MIRROR_VERTICAL:
            flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped.save(path + "/vert_" + str(uuid.uuid4()) + ".png")

        # horizontal mirror
        if license_plate[index] in CHARACTERS_TO_MIRROR_HORIZONTAL:
            flipped = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            flipped.save(path + "/hori_" + str(uuid.uuid4()) + ".png")

        # vertical and horizontal mirror
        if license_plate[index] in CHARACTERS_TO_MIRROR_BOTH:
            flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped = flipped.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            flipped.save(path + "/both_" + str(uuid.uuid4()) + ".png")


def _read_image(name: str) -> tuple[Image.Image, dict]:
    with open(name + ".txt", "r") as file:
        image_data_lines = file.readlines()

    meta_data = {
        "license_plate": "",
        "camera": "",
        "position_vehicle": [],
        "license_plate_corners": [],
        "vehicle_type": "",
        "vehicle_make": "",
        "vehicle_model": "",
        "vehicle_year": "",
        "char_positions": [],
    }
    for line in image_data_lines:
        line = line.strip()
        if line.startswith("plate:"):
            meta_data["license_plate"] = line[7:]
        if line.startswith("camera:"):
            meta_data["camera"] = line[8:]
        if line.startswith("position_vehicle:"):
            positions = line[18:].strip().split(" ")
            meta_data["position_vehicle"] = [int(p) for p in positions]
        if line.startswith("corners:"):
            corners = []
            corners_text = line[9:].split(" ")
            for corner in corners_text:
                c0, c1 = corner.split(",")
                corners.append((int(c0), int(c1)))
            meta_data["license_plate_corners"] = corners
        if line.startswith("type:"):
            meta_data["vehicle_type"] = line[6:]
        if line.startswith("make:"):
            meta_data["vehicle_make"] = line[6:]
        if line.startswith("model:"):
            meta_data["vehicle_model"] = line[7:]
        if line.startswith("year:"):
            meta_data["vehicle_year"] = line[6:]
        if line.startswith("char "):
            positions = line[9:].split(" ")
            positions = tuple([int(p) for p in positions])
            meta_data["char_positions"].append(positions)

    img = Image.open(name + ".png")
    return img, meta_data


def _create_license_plate_subimages_minmax(
    name: str, image: Image.Image, corners: list[tuple[int, int]]
):
    x_min = min(corner[0] for corner in corners)
    x_max = max(corner[0] for corner in corners)
    y_min = max(corner[1] for corner in corners)
    y_max = min(corner[1] for corner in corners)
    image = image.crop((x_min, y_max, x_max, y_min))

    path = license_plate_folder + "minmax/"
    if not os.path.exists(path):
        os.makedirs(path)

    image.save(f"{path}/{name}.png")

    inverted_image = ImageOps.invert(image)
    inverted_image.save(f"{path}/inv_{name}.png")


def _create_license_plate_subimages_rotated(
    name: str, image: Image.Image, corners: list[tuple[int, int]]
):
    correct_corner_order = [corners[0], corners[3], corners[2], corners[1]]
    transform = list(sum(correct_corner_order, ()))
    image = image.transform((200, 100), ImageTransform.QuadTransform(transform))

    path = license_plate_folder + "rotated/"
    if not os.path.exists(path):
        os.makedirs(path)

    image.save(f"{path}/{name}.png")

    inverted_image = ImageOps.invert(image)
    inverted_image.save(f"{path}/inv_{name}.png")


def generate_all_character_images():
    filenames = [
        name[:-4] for name in glob.glob(dataset_training + "/**/*.png", recursive=True)
    ]
    for filename in tqdm.tqdm(filenames):
        img, meta_data = _read_image(filename)
        chars = meta_data["char_positions"]
        license_plate = meta_data["license_plate"]
        _create_character_subimages(img, license_plate, chars)


def generate_all_license_plate_subimages():
    filenames = [
        name[:-4] for name in glob.glob(dataset_training + "/**/*.png", recursive=True)
    ]
    for filepath in tqdm.tqdm(filenames):
        img, meta_data = _read_image(filepath)
        corners = meta_data["license_plate_corners"]
        filename = filepath[filepath.rindex(os.sep) + 1 :]
        _create_license_plate_subimages_minmax(filename, img, corners)
        _create_license_plate_subimages_rotated(filename, img, corners)


print("---Generating character images---")
generate_all_character_images()
print("---Generating license_plate images---")
generate_all_license_plate_subimages()
