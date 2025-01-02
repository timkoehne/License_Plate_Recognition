import glob
import os
import shutil
from typing import Literal
from PIL import Image
import PIL
import tqdm
from read_image_label import read_label

training_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/training/"
validation_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/validation/"
output_path = "/home/tim/"
network_name = "licenseplate"

CLASS_IDS = {"licenseplate": 0}

output_path = output_path + network_name + "/"
if not os.path.exists(output_path + "backup/"):
    os.makedirs(output_path + "backup/", exist_ok=True)


class DataPoint:
    def __init__(self, image_filepath, label_filepath) -> None:
        self.image_path = image_filepath
        self.filename = image_filepath[image_filepath.rindex(os.sep) + 1 :]
        self.label = read_label(label_filepath)
        
        image = Image.open(self.image_path)
        vehicle_x, vehicle_y, vehicle_width, vehicle_height = self.label["position_vehicle"]
        self.image = image.crop((vehicle_x, vehicle_y,vehicle_x + vehicle_width, vehicle_y + vehicle_height))
        

    def generate_to_yolo_format(self):
        top_left, top_right, bottom_right, bottom_left = self.label["license_plate_corners"]
        x_min = min(top_left[0], bottom_left[0])
        x_max = max(top_right[0], bottom_right[0])
        y_min = min(top_left[1], top_right[1])
        y_max = max(bottom_left[1], bottom_right[1])
        width = x_max - x_min
        height = y_max - y_min
        
        vehicle_x, vehicle_y, vehicle_width, vehicle_height = self.label["position_vehicle"]
        x_min -= vehicle_x
        x_max -= vehicle_x
        y_min -= vehicle_y
        y_max -= vehicle_y
        
        x_center = (x_min + width / 2) / self.image.width
        y_center = (y_min + height / 2) / self.image.height
        norm_width = width / self.image.width
        norm_height = height / self.image.height
        class_id = 0

        # Schreibe das Ergebnis ins YOLO-Format
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"

        return yolo_line


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
        img = d.image.resize((int(2.75*d.image.height), d.image.height))
        img.save(save_images_path + "/" + d.filename)
        
        with open(
            save_images_path + "/" + d.filename[: d.filename.index(".")] + ".txt", "w"
        ) as file:
            yolo_format = d.generate_to_yolo_format()
            file.write(yolo_format)


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
    shutil.copyfile("dataset_template_files/yolov2_licenseplate.cfg", output_path + network_name + ".cfg")


def add_pretrained_weights():
    shutil.copyfile("dataset_template_files/darknet53.conv.74", output_path + "darknet53.conv.74")


def generate_run_command():
    data_file = network_name + ".data"
    cfg_file = network_name + ".cfg"
    pretrained_weights_file = "darknet53.conv.74"

    print("finished creating all data. Start training with:")
    print(
        f"darknet detector train -map -dont_show {data_file} {cfg_file} -clear {pretrained_weights_file}"
    )


generate_data(training_path, output_path + "train")
generate_data(validation_path, output_path + "valid")
generate_names_file()
generate_train_file()
generate_valid_file()
generate_data_file()
generate_cfg_file()
add_pretrained_weights()
generate_run_command()

