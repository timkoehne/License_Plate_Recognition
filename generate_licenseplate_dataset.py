import glob
import json
import os
import shutil
import cv2
import tqdm
import data_preparation
from read_image_label import read_label


NETWORK_NAME = "licenseplate"

with open("settings.json", "r") as file:
    settings = json.loads(file.read())
OUTPUT_PATH = settings["model_directory"]
UFPR_ALPR_DIRCTORY = settings["ufpr_alpr_dirctory"]


CLASS_IDS = {"licenseplate": 0}

output_path = OUTPUT_PATH + NETWORK_NAME + "/"
if not os.path.exists(output_path + "backup/"):
    os.makedirs(output_path + "backup/", exist_ok=True)


class DataPoint:
    def __init__(self, image_filepath, label_filepath) -> None:
        self.image_path = image_filepath
        self.filename = image_filepath[image_filepath.rindex(os.sep) + 1 :]
        self.label = read_label(label_filepath)
        self.image = data_preparation.load_cv2_image(self.image_path)
        
        vehicle_x, vehicle_y, vehicle_width, vehicle_height = self.label["position_vehicle"]
        self.image = self.image[vehicle_y:vehicle_y+vehicle_height, vehicle_x:vehicle_x+vehicle_width]
        

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
        
        
        x_center = (x_min + width / 2)
        y_center = (y_min + height / 2)
        
        
        
        bb = data_preparation.adjust_bb((x_center, y_center, width, height), len(self.image[0]), len(self.image), 416, 416)
        x_center, y_center, width, height = bb
        
        x_center_norm = x_center / 416
        y_center_norm = y_center / 416
        width_norm = width / 416
        height_norm = height / 416
        class_id = 0

        # Schreibe das Ergebnis ins YOLO-Format
        yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"

        return yolo_line


def generate_data(
    load_images_path: str, save_images_path: str):
    image_files = [
        name for name in glob.glob(load_images_path + "**/*.png", recursive=True)
    ]
    text_files = [name[:-4] + ".txt" for name in image_files]

    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path, exist_ok=True)

    print(f"loading files from {load_images_path}...")
    for img, txt in tqdm.tqdm(zip(image_files, text_files), total=len(image_files)):
        d = DataPoint(img, txt)
        filepath = save_images_path + "/" + d.filename
        data_preparation.save_image(d.image, filepath, 416, 416)
        
        with open(save_images_path + "/" + d.filename[: d.filename.index(".")] + ".txt", "w") as file:
            yolo_format = d.generate_to_yolo_format()
            file.write(yolo_format)


def generate_names_file():
    with open(output_path + NETWORK_NAME + ".names", "w") as file:
        file.write("\n".join([c for c in CLASS_IDS]))


def generate_train_file():
    image_files = [name for name in glob.glob(output_path + "train/*.png")]
    with open(output_path + NETWORK_NAME + f"_train.txt", "w") as file:
        file.write("\n".join(image_files))


def generate_valid_file():
    image_files = [name for name in glob.glob(output_path + "valid/*.png")]
    with open(output_path + NETWORK_NAME + f"_valid.txt", "w") as file:
        file.write("\n".join(image_files))


def generate_test_file():
    image_files = [name for name in glob.glob(output_path + "test/*.png")]
    with open(output_path + NETWORK_NAME + f"_test.txt", "w") as file:
        file.write("\n".join(image_files))

def generate_data_file():
    lines = []
    lines.append(f"classes = {len(CLASS_IDS)}")
    lines.append(f"train = {output_path+NETWORK_NAME}_train.txt")
    lines.append(f"valid = {output_path+NETWORK_NAME}_valid.txt")
    lines.append(f"names = {output_path+NETWORK_NAME}.names")
    lines.append(f"backup = {output_path}backup/")
    with open(output_path + NETWORK_NAME + ".data", "w") as file:
        file.write("\n".join(lines))


def generate_cfg_file():
    shutil.copyfile("dataset_template_files/yolov2_licenseplate.cfg", output_path + NETWORK_NAME + ".cfg")


def add_pretrained_weights():
    shutil.copyfile("dataset_template_files/darknet53.conv.74", output_path + "darknet53.conv.74")


def generate_run_command():
    data_file = NETWORK_NAME + ".data"
    cfg_file = NETWORK_NAME + ".cfg"
    pretrained_weights_file = "darknet53.conv.74"

    print("finished creating all data. Start training with:")
    print(
        f"darknet detector train -map -dont_show {data_file} {cfg_file} -clear {pretrained_weights_file}"
    )


generate_data(f"{UFPR_ALPR_DIRCTORY}training/", output_path + "train")
generate_data(f"{UFPR_ALPR_DIRCTORY}validation/", output_path + "valid")
generate_data(f"{UFPR_ALPR_DIRCTORY}testing/", output_path + "test")
generate_names_file()
generate_train_file()
generate_valid_file()
generate_test_file()
generate_data_file()
generate_cfg_file()
add_pretrained_weights()
generate_run_command()

