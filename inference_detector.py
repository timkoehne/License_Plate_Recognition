import sys
from typing import Tuple
from uuid import uuid4

import tqdm
import cv2
from darknet_python import darknet
import data_preparation

bb_percent_error_allowed = 0.1


# https://stackoverflow.com/a/71708600
def wait_until_cv2_window_exit(window):
    while True:
        k = cv2.waitKey(100) 
        if k == 27:
            print('ESC')
            cv2.destroyAllWindows()
            break
        if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

def roughly_equal(val1, val2, image_size):
    difference = abs(val1 - val2)
    if difference <= bb_percent_error_allowed * image_size:
        return True
    else:
        return False

def close_by(guess_bounding_box, correct_bounding_box, width: int, height: int):
    close_enough = True
    for i in range(4):
        max_size = width if i % 2 == 0 else height
        if not roughly_equal(guess_bounding_box[i], correct_bounding_box[i], max_size):
            close_enough = False
            break
    return close_enough

def preprocess_image(filename: str, width: int, height: int):
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = data_preparation.resize_image(image_rgb, width, height)
    return image_resized

def load_darknet_image(preprocessed_image, width: int, height: int):
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, preprocessed_image.tobytes())
    return darknet_image

def load_image_label(filename: str, width: float, height: float):
    txtfile = filename[:-4] + ".txt"
    
    with open(txtfile, "r") as f:
        results = []
        for line in f.read().splitlines():
            values = line.strip().split(" ")
            correct_class = values[0]
            correct_bounding_box = (float(values[1]) * width, float(values[2]) * height, float(values[3]) * width, float(values[4]) * height)
            results.append((correct_class, correct_bounding_box))
    # print(f"image contains: Class: {class_names[int(correct_values[0])]} - Bounding Box: {bounding_box}")
    return results

def print_results(no_predictions: int, true_positives: int, false_positives: int, false_negatives: int, true_negatives: int):
    print(f"There were {true_positives} correct predictions out of {true_positives+false_positives+false_negatives+no_predictions+true_negatives}")
    print(f"There were {no_predictions} images where no predictions was made")
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2*(precision * recall)/(precision + recall)
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1_score}")

def display_image_with_bbox(cv2_image, bb: Tuple[float, float, float, float]):
    center_x, center_y, width, height = bb
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    bb_image = cv2_image.copy()
    
    cv2.rectangle(bb_image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
    
    cv2.imshow("Image with Bounding Box", bb_image)
    wait_until_cv2_window_exit("Image with Bounding Box")

def display_licenseplate(cv2_image, licenseplate_bb: Tuple[float, float, float, float], vehicle_bb: Tuple[float, float, float, float], text: str):
    center_x, center_y, width, height = licenseplate_bb
    licenseplate_x1 = int(center_x - width / 2)
    licenseplate_y1 = int(center_y - height / 2)
    licenseplate_x2 = int(center_x + width / 2)
    licenseplate_y2 = int(center_y + height / 2)

    center_x, center_y, width, height = vehicle_bb
    vehicle_x1 = int(center_x - width / 2)
    vehicle_y1 = int(center_y - height / 2)
    vehicle_x2 = int(center_x + width / 2)
    vehicle_y2 = int(center_y + height / 2)

    bb_image = cv2_image.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
      # Get text size

    text_size, _ = cv2.getTextSize(text, font, 1, 2)
    text_width, text_height = text_size

    # Create a background rectangle
    background_rect = (vehicle_x1, vehicle_y1 - text_height - 5), (vehicle_x1 + text_width + 2, vehicle_y1 + 2)
    cv2.rectangle(bb_image, background_rect[0], background_rect[1], (0, 0, 0), -1)

    # Draw the text
    cv2.putText(bb_image, text, (vehicle_x1, vehicle_y1 - 5), font, 1, (0, 255, 0), 2)
    
    
    cv2.rectangle(bb_image, (licenseplate_x1, licenseplate_y1), (licenseplate_x2, licenseplate_y2), (0, 255, 0), 2) 
    cv2.rectangle(bb_image, (vehicle_x1, vehicle_y1), (vehicle_x2, vehicle_y2), (0, 255, 0), 2) 
    
    cv2.imwrite(f"{uuid4()}.png", bb_image)
    cv2.imshow("Image with Bounding Box", bb_image)
    wait_until_cv2_window_exit("Image with Bounding Box")


def find_correct_detections(width, height, class_names, highest_confidence_detections, correct_results):
    true_positives = 0
    false_positives = 0

    for i in range(len(correct_results)):
        found_tp = False
        found_fp = False
        for j in range(len(highest_confidence_detections)):
            correct_class = class_names[int(correct_results[i][0])]
            correct_bb = correct_results[i][1]
            guess_class = highest_confidence_detections[j][0]
            guess_bb = highest_confidence_detections[j][2]
            # print(f"comparing correct class {correct_class} with guess_class {guess_class}")
            if correct_class == guess_class:
                if close_by(correct_bb, guess_bb, width, height):
                    true_positives += 1
                    found_tp = True
                    break
            else:
                if close_by(correct_bb, guess_bb, width, height):
                    found_fp = True
                    break
                
        if not found_tp and found_fp:
            false_positives += 1
            
    false_negative = len(correct_results) - true_positives - false_positives
    
    return true_positives, false_negative, false_positives



def main():
    network = darknet.load_net(cfg_file.encode("ascii"), weights_file.encode("ascii"), 0)
    class_names = open(names_file).read().splitlines()

    prediction_threshold = 0.125
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    with open(validation_files, "r") as file:
        validation_images = file.read().splitlines()
        
    no_prediction = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for filename in tqdm.tqdm(validation_images):
        # print(filename)
        
        cv2_image = preprocess_image(filename, width, height)
        darknet_image = load_darknet_image(cv2_image, width, height)
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=prediction_threshold)
        darknet.free_image(darknet_image)
    
        if len(detections) == 0:
            no_prediction += 1
            continue
        
        correct_results = load_image_label(filename, width, height)
        highest_confidence_detections = sorted(detections, key=lambda x: float(x[1]), reverse=True)
        highest_confidence_detections = highest_confidence_detections[:len(correct_results)]
        
        # print(filename)
        tp, fn, fp = find_correct_detections(width, height, class_names, highest_confidence_detections, correct_results)
        true_positives += tp
        false_negatives += fn 
        false_positives += fp

    print_results(no_prediction, true_positives, false_positives, false_negatives, true_negatives)

    darknet.free_network_ptr(network)
        
if __name__ == "__main__":
    
    
    options = ["vehicle", "licenseplate", "character_segmentation", "character_recognition_digit", "character_recognition_letter"]
    if len(sys.argv) > 1 and sys.argv[1] in options:
        network_name = sys.argv[1]
        
        cfg_file = f"/home/tim/{network_name}/{network_name}.cfg"
        names_file = f"/home/tim/{network_name}/{network_name}.names"
        validation_files = f"/home/tim/{network_name}/{network_name}_valid.txt"
        weights_file = f"/home/tim/{network_name}/backup/{network_name}_final.weights"
        main()
    else:
        print(f"provide network name. \nOptions: {options}")

    
    
    