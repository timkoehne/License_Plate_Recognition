import glob
from typing import Literal, Tuple
from uuid import uuid4
import cv2
import tqdm
from darknet_python import darknet
from itertools import combinations

from data_preparation import get_scaling_factor_and_padding, resize_image
import data_preparation
from inference_detector import display_image_with_bbox, display_licenseplate, wait_until_cv2_window_exit
import read_image_label

LETTER_PADDING = 1
DIGIT_PADDING = 1

class Model:
    def __init__(self, network_name: str, prediction_threshold: float = 0.0) -> None:
        self.cfg_file = f"/home/tim/{network_name}/{network_name}.cfg"
        self.names_file = f"/home/tim/{network_name}/{network_name}.names"
        self.validation_files = f"/home/tim/{network_name}/{network_name}_valid.txt"
        self.weights_file = f"/home/tim/{network_name}/backup/{network_name}_final.weights"
        
        self.class_names = open(self.names_file).read().splitlines()
        self.network = darknet.load_net_custom(self.cfg_file.encode("ascii"), self.weights_file.encode("ascii"), 0, 1)
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.prediction_threshold = prediction_threshold
        
    def predict(self, darknet_image, top_k: int):
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.prediction_threshold)
        highest_confidence_detections = sorted(detections, key=lambda x: float(x[1]), reverse=True)[:top_k]
        return highest_confidence_detections




def scale_bounding_box_to_original_image(bb: Tuple[float, float, float, float], cv2_image, model: Model):
    scaling_factor, left_padding, top_padding = get_scaling_factor_and_padding(cv2_image, model.width, model.height)
    x, y, width, height = bb
    x -= left_padding
    y -= top_padding
    width = width * scaling_factor
    height = height * scaling_factor
    x = x * scaling_factor
    y = y * scaling_factor
    return x, y, width, height
    
def make_prediction(cv2_image, model: Model, top_k: int = 1):
    image_resized = resize_image(cv2_image, model.width, model.height)
    darknet_image = data_preparation.load_darknet_image(image_resized, model.width, model.height)
    predictions = model.predict(darknet_image, top_k)
    
    detected_results = []
    for prediction in predictions:
        # display_image_with_bbox(image_resized, prediction[2])
        center_x, center_y, width, height = scale_bounding_box_to_original_image(prediction[2], cv2_image, model)
        resized_prediction = (prediction[0], prediction[1], (center_x, center_y, width, height))
        # display_image_with_bbox(cv2_image, resized_prediction[2])
        # print(f"{resized_prediction}")
        detected_results.append(resized_prediction)
    darknet.free_image(darknet_image)
    return detected_results
    

def sort_licenseplate_bbs(bbs: list[float], vehicle_type: Literal["car"] | Literal["motorcycle"]):
        
    if vehicle_type == "car":
        sorted_bbs = sorted(bbs, key= lambda x: x[0])
    elif  vehicle_type == "motorcycle":
        bbs = sorted(bbs, key=lambda x: x[1])
        top_row = bbs[:3]
        top_row = sorted(top_row, key= lambda x: x[0])
        bottom_row = bbs[3:]
        bottom_row = sorted(bottom_row, key= lambda x: x[0])
        sorted_bbs = top_row + bottom_row
    return sorted_bbs



vehicle_model = Model("vehicles")
licenseplate_model = Model("licenseplate")
character_segmentation_model = Model("character_segmentation")
character_recognition_letter_model = Model("character_recognition_letter")
character_recognition_digit_model = Model("character_recognition_digit")


def predict_pipeline_licenseplate(cv2_image):

    # cv2.imwrite("test0.png", cv2_image)

    # cv2_image = data_preparation.resize_image(cv2_image, licenseplate_model.width, licenseplate_model.height)
    prediction = make_prediction(cv2_image, vehicle_model, top_k=1)[0]
    vehicle_type = prediction[0]
    bb = prediction[2]
    vehicle_bb = bb
    # display_image_with_bbox(cv2_image, bb)


    cv2_image = data_preparation.load_subimage(cv2_image, bb[0], bb[1], bb[2], bb[3])
    # cv2.imwrite("test1.png", cv2_image)
    prediction = make_prediction(cv2_image, licenseplate_model, top_k=1)[0]
    bb = prediction[2]
    
    vehicle_x = vehicle_bb[0] - vehicle_bb[2]/2
    vehicle_y = vehicle_bb[1] - vehicle_bb[3]/2
    
    licenseplate_bb = (bb[0] + vehicle_x, bb[1] + vehicle_y, bb[2], bb[3])

    cv2_image = data_preparation.load_subimage(cv2_image, bb[0], bb[1], bb[2], bb[3])
    # cv2.imwrite("test2.png", cv2_image)
    predictions = make_prediction(cv2_image, character_segmentation_model, top_k=7)
    bbs = [x[2] for x in predictions]
    sorted_bbs = sort_licenseplate_bbs(bbs, vehicle_type)

    for i, bb in enumerate(sorted_bbs, start=3):
        segment_image = data_preparation.load_subimage(cv2_image, bb[0], bb[1], bb[2]+LETTER_PADDING, bb[3]+LETTER_PADDING)
        # cv2.imwrite(f"test{i}.png", segment_image)

    licenseplate_text = ""
    for letter_bb in sorted_bbs[:3]:
        segment_image = data_preparation.load_subimage(cv2_image, letter_bb[0], letter_bb[1], letter_bb[2]+LETTER_PADDING, letter_bb[3]+LETTER_PADDING)
        res = make_prediction(segment_image, character_recognition_letter_model, top_k=3)
        # print(res)
        licenseplate_text += res[0][0][-1]


    for digit_bb in sorted_bbs[3:]:
        segment_image = data_preparation.load_subimage(cv2_image, digit_bb[0], digit_bb[1], digit_bb[2]+DIGIT_PADDING, digit_bb[3]+DIGIT_PADDING)
        res = make_prediction(segment_image, character_recognition_digit_model, top_k=3)
        # print(res)
        licenseplate_text += res[0][0][-1]
    return (licenseplate_bb, vehicle_bb, licenseplate_text)



def predict_pipeline(cv2_image):
    entire_licenseplate_prediction = []
    # cv2.imwrite("test0.png", cv2_image)

    # cv2_image = data_preparation.resize_image(cv2_image, licenseplate_model.width, licenseplate_model.height)
    prediction = make_prediction(cv2_image, vehicle_model, top_k=1)[0]
    vehicle_type = prediction[0]
    bb = prediction[2]
    # display_image_with_bbox(cv2_image, bb)


    cv2_image = data_preparation.load_subimage(cv2_image, bb[0], bb[1], bb[2], bb[3])
    # cv2.imwrite("test1.png", cv2_image)
    prediction = make_prediction(cv2_image, licenseplate_model, top_k=1)[0]
    bb = prediction[2]

    cv2_image = data_preparation.load_subimage(cv2_image, bb[0], bb[1], bb[2], bb[3])
    # cv2.imwrite("test2.png", cv2_image)
    predictions = make_prediction(cv2_image, character_segmentation_model, top_k=7)
    bbs = [x[2] for x in predictions]
    sorted_bbs = sort_licenseplate_bbs(bbs, vehicle_type)

    for i, bb in enumerate(sorted_bbs, start=3):
        segment_image = data_preparation.load_subimage(cv2_image, bb[0], bb[1], bb[2]+LETTER_PADDING, bb[3]+LETTER_PADDING)
        # cv2.imwrite(f"test{i}.png", segment_image)

    for letter_bb in sorted_bbs[:3]:
        segment_image = data_preparation.load_subimage(cv2_image, letter_bb[0], letter_bb[1], letter_bb[2]+LETTER_PADDING, letter_bb[3]+LETTER_PADDING)
        res = make_prediction(segment_image, character_recognition_letter_model, top_k=3)
        # print(res)
        entire_licenseplate_prediction.append(res)


    for digit_bb in sorted_bbs[3:]:
        segment_image = data_preparation.load_subimage(cv2_image, digit_bb[0], digit_bb[1], digit_bb[2]+DIGIT_PADDING, digit_bb[3]+DIGIT_PADDING)
        res = make_prediction(segment_image, character_recognition_digit_model, top_k=3)
        # print(res)
        entire_licenseplate_prediction.append(res)
    return entire_licenseplate_prediction

def run_pipeline_with_redundancy(files: list[str]):
    label_file = files[0][:-4]+".txt"
    correct_license_plate = read_image_label.read_label(label_file)["license_plate"]

    letter_predictions = [{} for _ in range(7)]
    for file in tqdm.tqdm(files): 
        cv2_image = data_preparation.load_cv2_image(file)
        res = predict_pipeline(cv2_image)
        for letter_pos in range(len(letter_predictions)):
            for prediction in res[letter_pos]:
                letter = prediction[0]
                confidence = prediction[1]
                if letter not in letter_predictions[letter_pos]:
                    letter_predictions[letter_pos][letter] = 0
                letter_predictions[letter_pos][letter] += float(confidence)

    correct_predicted_letters = 0
    for i, letter_prediction in enumerate(letter_predictions):
        predicted_letter = max(letter_prediction, key=lambda x: x[1])[-1]
        # print(f"{predicted_letter} == {correct_license_plate[i]}")
        if predicted_letter == correct_license_plate[i]:
            correct_predicted_letters += 1
    return correct_predicted_letters


def run_pipeline_without_redundancy(folders: list[str], min_correct_characters: int, percentage_correct_lp_required: float):
    num_correct_detections = 0
    for folder in tqdm.tqdm(folders):
        files = glob.glob(folder + "/*.png")#[:10]
        entire_licenseplate_correct = 0
        correct_characters = 0
        for file in tqdm.tqdm(files): 
            label_file = file[:-4]+".txt"
            correct_license_plate = read_image_label.read_label(label_file)["license_plate"]
            cv2_image = data_preparation.load_cv2_image(file)
            res = predict_pipeline(cv2_image)
            for correct_character, prediction in zip(correct_license_plate, res):
                predicted_character = prediction[0][0][-1]
                # print(f"comraing correct {correct_character} to predicted {predicted_character}")
                if predicted_character == correct_character:
                    # print("correct character")
                    correct_characters += 1
            if correct_characters >= min_correct_characters:
                # print(f"there are atleast {min_correct} correct detection in {"".join([p[0][0][-1] for p in res])} compared to correct answer: {correct_license_plate}")
                entire_licenseplate_correct += 1
                # print(f"there are sofar {entire_licenseplate_correct} / {len(files)} correct")
            else:
                # print(f"too few ({correct_characters}) correct characters in in {"".join([p[0][0][-1] for p in res])} compared to correct answer: {correct_license_plate}")
                pass
            correct_characters = 0
        
        if entire_licenseplate_correct > len(files) * percentage_correct_lp_required:
            num_correct_detections += 1
        entire_licenseplate_correct = 0

    return num_correct_detections


def test_pipeline_without_redundandy(min_characters_correct: int):
    folders = glob.glob("/mnt/f/OpenScience Data/UFPR-ALPR dataset/testing/*")
    entire_licenseplate_correct = run_pipeline_without_redundancy(folders, min_characters_correct, 0.5)
    print(f"There were a total of {entire_licenseplate_correct} correct licenseplate predictions out of {len(folders)}. Thats {entire_licenseplate_correct / len(folders) * 100}%")


def test_pipeline_with_redundancy(min_characters_correct: int):
    folders = glob.glob("/mnt/f/OpenScience Data/UFPR-ALPR dataset/testing/*")
    total_correct_characters = 0
    total_characters = 0
    entire_licenseplate_correct = 0
    for folder in tqdm.tqdm(folders):
        files = glob.glob(folder + "/*.png")#[:10]
        correct_characters = run_pipeline_with_redundancy(files)
        total_correct_characters += correct_characters
        if correct_characters >= min_characters_correct:
            entire_licenseplate_correct += 1
        total_characters += 7
        # print(f"There were {correct} correctly predicted characters")
    print(f"There were  {total_correct_characters} correctly predicted characters out of {total_characters}. Thats {total_correct_characters/total_characters*100}%")
    print(f"There were {entire_licenseplate_correct} correctly predicted licenseplates out of {len(folders)}. Thats {entire_licenseplate_correct/len(folders)*100}%")


def show_image_with_detection(filepath):
    img = cv2.imread(filepath)
    prediction = predict_pipeline_licenseplate(img)
    print(prediction)
    display_licenseplate(img, prediction[0], prediction[1], prediction[2])


# test_pipeline_without_redundandy(6)
# test_pipeline_without_redundandy(7)
# test_pipeline_with_redundancy(6)
# test_pipeline_with_redundancy(7)


filepath = "/mnt/f/OpenScience Data/UFPR-ALPR dataset/testing/track0142/track0142[04].png"
show_image_with_detection(filepath)


darknet.free_network_ptr(vehicle_model.network)
darknet.free_network_ptr(licenseplate_model.network)
darknet.free_network_ptr(character_segmentation_model.network)
darknet.free_network_ptr(character_recognition_letter_model.network)
darknet.free_network_ptr(character_recognition_digit_model.network)