from curses.ascii import isalpha
import glob
import os
from random import shuffle
import shutil
from typing import Tuple
import tqdm
from data_preparation import generate_flipped_images, generate_negative_image
import data_preparation
from read_image_label import read_label

training_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/training/"
validation_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/validation/"
output_path = "/home/tim/"
network_name = "character_recognition_letter"

PADDING = 1
CLASS_IDS = { 
    'letter_A': 0, 'letter_B': 1, 'letter_C': 2, 'letter_D': 3, 'letter_E': 4, 'letter_F': 5, 'letter_G': 6, 'letter_H': 7, 'letter_I': 8, 'letter_J': 9,
    'letter_K': 10, 'letter_L': 11, 'letter_M': 12, 'letter_N': 13, 'letter_O': 14, 'letter_P': 15, 'letter_Q': 16, 'letter_R': 17, 'letter_S': 18, 'letter_T': 19,
    'letter_U': 20, 'letter_V': 21, 'letter_W': 22, 'letter_X': 23, 'letter_Y': 24, 'letter_Z': 25
}

output_path = output_path + network_name + "/"
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


def save_image(cv_image, classname: str, save_images_path: str, filename:str, fileending: str):
    class_path = save_images_path + f"/{classname}"
    filepath = class_path + "/" + filename + fileending
    data_preparation.save_image(cv_image, filepath, 270, 80)
    


def generate_data(
    load_images_path: str, save_images_path: str):
    image_files = [
        name for name in glob.glob(load_images_path + "**/*.png", recursive=True)
    ]
    text_files = [name[:-4] + ".txt" for name in image_files]
    
    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path, exist_ok=True)
        for classname in CLASS_IDS:
            os.makedirs(save_images_path + f"/{classname}", exist_ok=True)
            
    print(f"loading files from {load_images_path}...")
    for img, txt in tqdm.tqdm(zip(image_files, text_files), total=len(image_files)):
        datapoint = DataPoint(img, txt)
        
        for character_datapoint in datapoint.characters:
            image = character_datapoint.image
            image_classname = f"letter_{character_datapoint.text}"
            
            filename = character_datapoint.filename[:character_datapoint.filename.index(".")]
            fileending = character_datapoint.filename[character_datapoint.filename.index("."):]
            save_image(image, image_classname, save_images_path, filename, fileending)
            
            negative_img = generate_negative_image(image)
            negative_img_filename = filename + "-negative"
            save_image(negative_img, image_classname, save_images_path, negative_img_filename, fileending)
        
        
        flipped_imgs = generate_flipped_images(image, character_datapoint.text)
        for i, flipped_img in enumerate(flipped_imgs):
            flipped_img_filename = filename + f"-{i}"
            save_image(flipped_img, image_classname, save_images_path, flipped_img_filename, fileending)


def generate_names_file():
    with open(output_path + network_name + ".names", "w") as file:
        file.write("\n".join([c for c in CLASS_IDS]))


def generate_train_file():
    image_files = [name for name in glob.glob(output_path + "train/**/*.png", recursive=True)]
    shuffle(image_files)
    with open(output_path + network_name + f"_train.txt", "w") as file:
        file.write("\n".join(image_files))


def generate_valid_file():
    image_files = [name for name in glob.glob(output_path + "valid/**/*.png", recursive=True)]
    with open(output_path + network_name + f"_valid.txt", "w") as file:
        file.write("\n".join(image_files))


def generate_data_file():
    lines = []
    lines.append(f"classes = {len(CLASS_IDS)}")
    lines.append(f"train = {output_path+network_name}_train.txt")
    lines.append(f"valid = {output_path+network_name}_valid.txt")
    lines.append(f"labels = {output_path+network_name}.names")
    lines.append(f"backup = {output_path}backup/")
    with open(f"{output_path+network_name}.data", "w") as file:
        file.write("\n".join(lines))


def generate_cfg_file():
    shutil.copyfile("dataset_template_files/crnet_character_recognition_letter.cfg", output_path + network_name + ".cfg")

def generate_run_command():
    data_file = network_name + ".data"
    cfg_file = network_name + ".cfg"

    print("finished creating all data. Start training with:")
    print(
        f"darknet classifier train {data_file} {cfg_file}"
    )


generate_data(training_path, output_path + "train")
generate_data(validation_path, output_path + "valid")
generate_names_file()
generate_train_file()
generate_valid_file()
generate_data_file()
generate_cfg_file()
generate_run_command()