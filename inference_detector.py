import glob
from typing import Tuple

import tqdm
import cv2
from darknet_python import darknet



network_name = "character_recognition_digit_fix"

cfg_file = f"/home/tim/{network_name}/{network_name}.cfg"
names_file = f"/home/tim/{network_name}/{network_name}.names"
validation_files = f"/home/tim/{network_name}/{network_name}_valid.txt"
weights_file = f"/home/tim/{network_name}/backup/{network_name}_final.weights"


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
    if difference <= 0.15 * image_size:
        return True
    else:
        return False

def close_by(highest_confidence_detection_bounding_box, correct_bounding_box):
    close_enough = True
    for i in range(4):
        max_size = width if i % 2 == 0 else height
        if not roughly_equal(highest_confidence_detection_bounding_box[i], correct_bounding_box[i], max_size):
            close_enough = False
            break
    return close_enough

def load_darknet_image(filename):
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    return darknet_image

def load_image_label():
    txtfile = filename[:-4] + ".txt"
    with open(txtfile, "r") as f:
        correct_values = f.readline().strip().split(" ")
        correct_bounding_box = (float(correct_values[1]) * width, float(correct_values[2]) * height, float(correct_values[3]) * width, float(correct_values[4]) * height)
    # print(f"image contains: Class: {class_names[int(correct_values[0])]} - Bounding Box: {bounding_box}")
    return int(correct_values[0]), correct_bounding_box

def calculate_results():
    print(f"There were {correct_predictions} correct predictions out of {len(validation_images)} images")
    print(f"There were {no_prediction} images where no predictions was made")
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2*(precision * recall)/(precision + recall)
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1_score}")

def display_detection(detection, image_filename):
    image_bgr = cv2.imread(image_filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # display the results on the console
    darknet.print_detections([detection], True)
    # draw some boxes and labels over what was detected
    image_with_boxes = darknet.draw_boxes([detection], image_resized, colours)
    cv2.imshow("annotated image", cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    wait_until_cv2_window_exit("annotated image")

network = darknet.load_net_custom(cfg_file.encode("ascii"), weights_file.encode("ascii"), 0, 1)
class_names = open(names_file).read().splitlines()

colours = darknet.class_colors(class_names)

prediction_threshold = 0.125
width = darknet.network_width(network)
height = darknet.network_height(network)

with open(validation_files, "r") as file:
    validation_images = file.read().splitlines()
    
correct_predictions = 0
no_prediction = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
for filename in tqdm.tqdm(validation_images):
    # print(filename)
    
    correct_class, correct_bounding_box = load_image_label()

    darknet_image = load_darknet_image(filename)
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=prediction_threshold)
    darknet.free_image(darknet_image)
    # print("detections: " + str(detections))
    
    if len(detections) == 0:
        no_prediction += 1
        continue
    
    highest_confidence_detection = max(detections, key=lambda x: float(x[1]))
    highest_confidence_detection_bounding_box = highest_confidence_detection[2]
    # print(f"highest_confidence_detection is {highest_confidence_detection}")
    
    if highest_confidence_detection[0] == class_names[int(correct_class)]:
        if close_by(highest_confidence_detection_bounding_box, correct_bounding_box):
            true_positives += 1
            correct_predictions += 1
        else:
            false_negatives += 1
            # display_detection(highest_confidence_detection, filename)
    else:
        if close_by(highest_confidence_detection_bounding_box, correct_bounding_box):
            false_positives += 1
        else: # 
            true_negatives += 1

calculate_results()

darknet.free_network_ptr(network)