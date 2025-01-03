import glob
import os
import shutil
from typing import Literal, Tuple
from PIL import Image
import tqdm
from data_preparation import generate_flipped_images, generate_negative_image
from read_image_label import read_label

training_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/training/"
validation_path = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/validation/"
output_path = "/home/tim/"
network_name = "character_recognition_digit"

CLASS_IDS = { 
    'digit_0': 0, 'digit_1': 1, 'digit_2': 2, 'digit_3': 3, 'digit_4': 4, 'digit_5': 5, 'digit_6': 6, 'digit_7': 7, 'digit_8': 8, 'digit_9': 9
}

output_path = output_path + network_name + "/"
os.makedirs(output_path + "backup/", exist_ok=True)


class Character_Datapoint:
    def __init__(self, image: Image.Image, filename: str, rect: Tuple[int, int, int, int] , text: str):
        x_min = rect[0]
        y_min = rect[1]
        x_max = rect[0] + rect[2]
        y_max = rect[1] + rect[3]
        self.image: Image.Image = image.crop((x_min, y_min, x_max, y_max))
        self.filename = filename
        self.text: str = text


class DataPoint:
    def __init__(self, image_filepath, label_filepath) -> None:
        self.image_path = image_filepath
        self.filename = image_filepath[image_filepath.rindex(os.sep) + 1 :]
        self.label = read_label(label_filepath)
        self.image = Image.open(self.image_path)
        
        
        char_positions = self.label["char_positions"]
        char_text = self.label["license_plate"]
        self.characters: list[Character_Datapoint] = []
        filename_without_fileending = self.filename[:self.filename.index(".")]
        fileending = self.filename[self.filename.index("."):]
        for i in range(4, 7):
            self.characters.append(Character_Datapoint(self.image, f"{filename_without_fileending}-{i}{fileending}", char_positions[i], char_text[i]))


def save_image(image: Image.Image, classname: str, save_images_path: str, filename:str, fileending: str):
    class_path = save_images_path + f"/{classname}"
    image.save(class_path + "/" + filename + fileending)


def generate_data(
    load_images_path: str, save_images_path: str):
    image_files = [
        name for name in glob.glob(load_images_path + "**/*.png", recursive=True)
    ]
    text_files = [name[:-4] + ".txt" for name in image_files]
    
    
    data = []
    for d in tqdm.tqdm(zip(image_files, text_files), total=len(image_files)):
        datapoint = DataPoint(d[0], d[1])
        
        for character_datapoint in datapoint.characters:
            data.append(character_datapoint)

    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path, exist_ok=True)
        for classname in CLASS_IDS:
            os.makedirs(save_images_path + f"/{classname}", exist_ok=True)

    print(f"loading data from {load_images_path}...")
    for d in tqdm.tqdm(data):
        image = d.image
        image_classname = f"digit_{d.text}"
        
        filename = d.filename[:d.filename.index(".")]
        fileending = d.filename[d.filename.index("."):]
        save_image(image, image_classname, save_images_path, filename, fileending)
        
        negative_img = generate_negative_image(image)
        negative_img_filename = filename + "-negative"
        save_image(negative_img, image_classname, save_images_path, negative_img_filename, fileending)
        
        
        flipped_imgs = generate_flipped_images(image, d.text)
        for i, flipped_img in enumerate(flipped_imgs):
            flipped_img_filename = filename + f"-flipped-{i}"
            save_image(flipped_img, image_classname, save_images_path, flipped_img_filename, fileending)


def generate_names_file():
    with open(output_path + network_name + ".names", "w") as file:
        file.write("\n".join([c for c in CLASS_IDS]))


def generate_train_file():
    image_files = [name for name in glob.glob(output_path + "train/**/*.png", recursive=True)]
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
    shutil.copyfile("dataset_template_files/crnet_character_recognition_digit.cfg", output_path + network_name + ".cfg")

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