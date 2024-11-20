import glob
import os
import shutil
from typing import Literal
from PIL import Image
import tqdm
from read_image_label import read_label

training_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/training/"
validation_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/validation/"
output_path = "/home/tim/"
network_name = "vehicles"

CLASS_IDS = {"car": 0, "motorcycle": 1}

output_path = output_path + network_name + "/"
if not os.path.exists(output_path + "backup/"):
    os.makedirs(output_path + "backup/", exist_ok=True)


class DataPoint:
    def __init__(self, image_filepath, label_filepath) -> None:
        self.image_path = image_filepath
        self.filename = image_filepath[image_filepath.rindex(os.sep) + 1 :]

        self.label = read_label(label_filepath)

    def generate_to_yolo_format(self):
        x_min, y_min, width, height = self.label["position_vehicle"]
        image = Image.open(self.image_path)
        x_center = (x_min + width / 2) / image.width
        y_center = (y_min + height / 2) / image.height
        norm_width = width / image.width
        norm_height = height / image.height
        class_id = CLASS_IDS[self.label["vehicle_type"]]

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
        DataPoint(img, txt)
        for img, txt in zip(image_files, text_files)
    ]

    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path, exist_ok=True)

    print(f"loading data from {load_images_path}...")
    for d in tqdm.tqdm(data):
        # print(f"copying {d.image_path} to {dataset_folder}")
        shutil.copyfile(d.image_path, save_images_path + "/" + d.filename)
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
    shutil.copyfile("dataset_template_files/yolov2.cfg", output_path + network_name + ".cfg")


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
