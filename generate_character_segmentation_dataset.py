import glob
import os
import shutil
from typing import Literal
from PIL import Image
import tqdm
from data_preparation import generate_negative_image
from read_image_label import read_label

training_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/training/"
validation_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/validation/"
output_path = "/home/tim/"
network_name = "character_segmentation"

CLASS_IDS = {"character": 0}

output_path = output_path + network_name + "/"
if not os.path.exists(output_path + "backup/"):
    os.makedirs(output_path + "backup/", exist_ok=True)


class DataPoint:
    def __init__(self, image_filepath, label_filepath) -> None:
        self.image_path = image_filepath
        self.filename = image_filepath[image_filepath.rindex(os.sep) + 1 :]
        self.label = read_label(label_filepath)
        self.char_text = self.label["license_plate"]
        
        image = Image.open(self.image_path)
        
        license_plate_top_left, license_plate_top_right, license_plate_bottom_right, license_plate_bottom_left = self.label["license_plate_corners"]
        license_plate_x_min = min(license_plate_top_left[0], license_plate_bottom_left[0])
        license_plate_x_max = max(license_plate_top_right[0], license_plate_bottom_right[0])
        license_plate_y_min = min(license_plate_top_left[1], license_plate_top_right[1])
        license_plate_y_max = max(license_plate_bottom_left[1], license_plate_bottom_right[1])
        license_plate_width = license_plate_x_max - license_plate_x_min
        license_plate_height = license_plate_y_max - license_plate_y_min
        self.image = image.crop((license_plate_x_min, license_plate_y_min,license_plate_x_min + license_plate_width, license_plate_y_min + license_plate_height))
        

    def generate_to_yolo_format(self):
        license_plate_top_left, license_plate_top_right, license_plate_bottom_right, license_plate_bottom_left = self.label["license_plate_corners"]
        license_plate_x_min = min(license_plate_top_left[0], license_plate_bottom_left[0])
        license_plate_y_min = min(license_plate_top_left[1], license_plate_top_right[1])
        license_plate_y_max = max(license_plate_bottom_left[1], license_plate_bottom_right[1])
        
        yolo_lines = []
        for character in self.label["char_positions"]:
            x_min, y_min, width, height = character
            x_min -= license_plate_x_min
            y_min -= license_plate_y_min
            # print((x_min, y_min, width, height))

            x_center = (x_min + width / 2) / self.image.width
            y_center = (y_min + height / 2) / self.image.height
            norm_width = width / self.image.width
            norm_height = height / self.image.height
            class_id = 0

            if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or norm_width < 0 or norm_width > 1 or norm_height < 0 or norm_height > 1:
                
                print(f"char pos: {character}")
                print(f"x_min: {x_min}")
                print(f"y_min: {y_min}")
                print(f"width: {width}")
                print(f"height: {height}")
                print(f"image_width: {self.image.width}")
                print(f"image_height: {self.image.height}")
                print(f"x_center={x_center} y_center={y_center} norm_width={norm_width} norm_height={norm_height}")
                print(f"license_plate_y_min={license_plate_y_min} license_plate_y_max={license_plate_y_max}")
                raise Exception("incorrect x_center value")


            # Schreibe das Ergebnis ins YOLO-Format
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

        return "".join(yolo_lines)

def save_image(image: Image.Image, yolo_format: str, save_images_path: str, filename:str, fileending: str):
    image.save(save_images_path + "/" + filename + fileending)
    with open(
        save_images_path + "/" + filename + ".txt", "w"
    ) as file:
        file.write(yolo_format)

def generate_data(
    load_images_path: str, save_images_path: str):
    image_files = [
        name for name in glob.glob(load_images_path + "**/*.png", recursive=True)
    ]
    text_files = [name[:-4] + ".txt" for name in image_files]
    data = [
        DataPoint(img, txt) for img, txt in tqdm.tqdm(zip(image_files, text_files), total=len(image_files))
    ]

    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path, exist_ok=True)

    print(f"loading data from {load_images_path}...")
    for d in tqdm.tqdm(data):
        image = d.image
        yolo_format = d.generate_to_yolo_format()
        
        filename = d.filename[:d.filename.index(".")]
        fileending = d.filename[d.filename.index("."):]
        save_image(image, yolo_format, save_images_path, filename, fileending)
        
        negative_img = generate_negative_image(image)
        negative_img_filename = filename + "-negative"
        save_image(negative_img, yolo_format, save_images_path, negative_img_filename, fileending)



def generate_names_file():
    with open(output_path + network_name + ".names", "w") as file:
        file.write("\n".join([c for c in CLASS_IDS]))


def generate_train_file():
    image_files = [name for name in glob.glob(output_path + "train/*.png")]
    with open(output_path + network_name + f"_train.txt", "w") as file:
        file.write("\n".join(image_files))


def generate_valid_file():
    image_files = [name for name in glob.glob(output_path + "valid/*.png")]
    with open(output_path + network_name + f"_valid.txt", "w") as file:
        file.write("\n".join(image_files))


def generate_data_file():
    lines = []
    lines.append(f"classes = {len(CLASS_IDS)}")
    lines.append(f"train = {output_path+network_name}_train.txt")
    lines.append(f"valid = {output_path+network_name}_valid.txt")
    lines.append(f"names = {output_path+network_name}.names")
    lines.append(f"backup = {output_path}backup/")
    with open(output_path + network_name + ".data", "w") as file:
        file.write("\n".join(lines))


def generate_cfg_file():
    shutil.copyfile("dataset_template_files/crnet_character_segmentation.cfg", output_path + network_name + ".cfg")

def generate_run_command():
    data_file = network_name + ".data"
    cfg_file = network_name + ".cfg"

    print("finished creating all data. Start training with:")
    print(
        f"darknet detector train -map -dont_show {data_file} {cfg_file}"
    )


generate_data(training_path, output_path + "train")
generate_data(validation_path, output_path + "valid")
generate_names_file()
generate_train_file()
generate_valid_file()
generate_data_file()
generate_cfg_file()
generate_run_command()


# test = DataPoint(f"{training_path}track0012/track0012[29].png", f"{training_path}track0012/track0012[29].txt")
# yolo_format = test.generate_to_yolo_format()
# print(yolo_format)



# image = test.image
# yolo_format = test.generate_to_yolo_format()

# filename = test.filename[:test.filename.index(".")]
# fileending = test.filename[test.filename.index("."):]
# save_image(image, yolo_format, "./", filename, fileending)

# negative_img = generate_negative_image(image)
# negative_img_filename = filename + "-negative"
# save_image(negative_img, yolo_format, "./", negative_img_filename, fileending)