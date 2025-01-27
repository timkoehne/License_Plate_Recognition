from curses.ascii import isalpha
import glob
import json
import os
from random import shuffle
import shutil
from typing import Tuple
import tqdm
from data_preparation import generate_flipped_images, generate_negative_image
import data_preparation
from read_image_label import read_label


NETWORK_NAME = "character_recognition_letter"

with open("settings.json", "r") as file:
    settings = json.loads(file.read())
OUTPUT_PATH = settings["model_directory"]
PADDING = settings["letter_padding"]
UFPR_ALPR_DIRCTORY = settings["ufpr_alpr_dirctory"]


CLASS_IDS = { 
    'letter_A': 0, 'letter_B': 1, 'letter_C': 2, 'letter_D': 3, 'letter_E': 4, 'letter_F': 5, 'letter_G': 6, 'letter_H': 7, 'letter_I': 8, 'letter_J': 9,
    'letter_K': 10, 'letter_L': 11, 'letter_M': 12, 'letter_N': 13, 'letter_O': 14, 'letter_P': 15, 'letter_Q': 16, 'letter_R': 17, 'letter_S': 18, 'letter_T': 19,
    'letter_U': 20, 'letter_V': 21, 'letter_W': 22, 'letter_X': 23, 'letter_Y': 24, 'letter_Z': 25
}

output_path = OUTPUT_PATH + NETWORK_NAME + "/"
os.makedirs(output_path + "backup/", exist_ok=True)


class Character_Datapoint:
    def __init__(self, cv_image, filename: str, rect: Tuple[int, int, int, int] , text: str):
        x_min = rect[0] - PADDING
        y_min = rect[1] - PADDING
        x_max = rect[0] + rect[2] + PADDING
        y_max = rect[1] + rect[3] + PADDING
        self.image = cv_image[y_min:y_max, x_min:x_max]
        self.filename = filename
        self.text: str = text

    def generate_to_yolo_format(self):
        class_id = CLASS_IDS["letter_" + self.text]
        return f"{class_id} 0.5 0.5 1 1"


class DataPoint:
    def __init__(self, image_filepath, label_filepath) -> None:
        self.image_path = image_filepath
        self.filename = image_filepath[image_filepath.rindex(os.sep) + 1 :]
        self.label = read_label(label_filepath)
        self.image = data_preparation.load_cv2_image(self.image_path)
        
        char_positions = self.label["char_positions"]
        char_text = self.label["license_plate"]
        self.characters: list[Character_Datapoint] = []
        filename_without_fileending = self.filename[:self.filename.index(".")]
        fileending = self.filename[self.filename.index("."):]
        for i in range(0, 7):
            if isalpha(char_text[i]):
                self.characters.append(Character_Datapoint(self.image, f"{filename_without_fileending}-{i}{fileending}", char_positions[i], char_text[i]))
            if char_text[i] == "0":
                self.characters.append(Character_Datapoint(self.image, f"{filename_without_fileending}-{i}{fileending}", char_positions[i], "O"))
            if char_text[i] == "1":
                self.characters.append(Character_Datapoint(self.image, f"{filename_without_fileending}-{i}{fileending}", char_positions[i], "I"))


def save_image(cv_image, yolo_format: str, save_images_path: str, filename:str, fileending: str):
    filepath = save_images_path + "/" + filename + fileending
    data_preparation.save_image(cv_image, filepath, 288, 96)
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
    
    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path, exist_ok=True)
            
    print(f"loading files from {load_images_path}...")
    for img, txt in tqdm.tqdm(zip(image_files, text_files), total=len(image_files)):
        datapoint = DataPoint(img, txt)
        
        for character_datapoint in datapoint.characters:
            image = character_datapoint.image
            yolo_format = character_datapoint.generate_to_yolo_format()
            
            filename = character_datapoint.filename[:character_datapoint.filename.index(".")]
            fileending = character_datapoint.filename[character_datapoint.filename.index("."):]
            save_image(image, yolo_format, save_images_path, filename, fileending)
            
            negative_img = generate_negative_image(image)
            negative_img_filename = filename + "-negative"
            save_image(negative_img, yolo_format, save_images_path, negative_img_filename, fileending)
        
            flipped_imgs = generate_flipped_images(image, character_datapoint.text)
            for i, flipped_img in enumerate(flipped_imgs):
                flipped_img_filename = filename + f"-flipped-{i}"
                save_image(flipped_img, yolo_format, save_images_path, flipped_img_filename, fileending)
            

def generate_names_file():
    with open(output_path + NETWORK_NAME + ".names", "w") as file:
        file.write("\n".join([c for c in CLASS_IDS]))


def generate_train_file():
    image_files = [name for name in glob.glob(output_path + "train/*.png")]
    shuffle(image_files)
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
    with open(f"{output_path+NETWORK_NAME}.data", "w") as file:
        file.write("\n".join(lines))


def generate_cfg_file():
    shutil.copyfile("dataset_template_files/crnet_character_recognition_letter.cfg", output_path + NETWORK_NAME + ".cfg")

def generate_run_command():
    data_file = NETWORK_NAME + ".data"
    cfg_file = NETWORK_NAME + ".cfg"

    print("finished creating all data. Start training with:")
    print(
        f"darknet detector train -map -dont_show {data_file} {cfg_file}"
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
generate_run_command()