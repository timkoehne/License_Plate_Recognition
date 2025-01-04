import glob
import sys
from typing import Tuple

import tqdm
import cv2
from darknet_python import darknet



network_name = "character_recognition_letter"

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

def load_darknet_image(filename):
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    return darknet_image

def load_image_label(filename):
    for classname in class_names:
        if classname in filename:
            return classname

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

prediction_threshold = 0.0
width = darknet.network_width(network)
height = darknet.network_height(network)

with open(validation_files, "r") as file:
    validation_images = file.read().splitlines()
    
correct_predictions = 0
no_prediction = 0
true_positives = 0
wrong_predictions = 0
true_negatives = 0
false_negatives = 0
for filename in tqdm.tqdm(validation_images):
    # print(filename)
    
    correct_class = load_image_label(filename)

    darknet_image = load_darknet_image(filename)
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    # detections = darknet.detect_image(network, class_names, darknet_image, thresh=prediction_threshold)
    darknet.free_image(darknet_image)
    # print("detections: " + str(detections))
    
    if len(predictions) == 0:
        no_prediction += 1
        continue
    
    highest_confidence_detection = max(predictions, key=lambda x: float(x[1]))
    # print(f"prediction: {highest_confidence_detection[0]}, correct: {correct_class}")
    
    if highest_confidence_detection[0] == correct_class:
            correct_predictions += 1
    else:
            wrong_predictions += 1


print(f"correct predictions: {correct_predictions}")
print(f"wrong predictions: {wrong_predictions}")
print(f"average: {correct_predictions / (correct_predictions + wrong_predictions):4f}")


darknet.free_network_ptr(network)