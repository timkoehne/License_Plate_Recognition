import glob
import cv2
from darknet_python import darknet, darknet_images

cfg_file = "/home/tim/test/test.cfg"
names_file = "/home/tim/test/test.names"
weights_file = "/home/tim/test/backup/test_final.weights"
validation_files = "/home/tim/test/test_valid.txt"


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


network = darknet.load_net_custom(cfg_file.encode("ascii"), weights_file.encode("ascii"), 0, 1)
class_names = open(names_file).read().splitlines()

colours = darknet.class_colors(class_names)

prediction_threshold = 0.5
width = darknet.network_width(network)
height = darknet.network_height(network)

with open(validation_files, "r") as file:
    validation_images = file.read().splitlines()
for filename in validation_images[:5]:
    print(filename)

    # use OpenCV to load the image and swap OpenCV's usual BGR for the RGB that Darknet requires
    image_bgr = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # create a Darknet-specific image structure with the resized image
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

    # this is where darknet is called to do the magic!
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=prediction_threshold)
    darknet.free_image(darknet_image)

    # display the results on the console
    darknet.print_detections(detections, True)

    # draw some boxes and labels over what was detected
    image_with_boxes = darknet.draw_boxes(detections, image_resized, colours)

    cv2.imshow("annotated image", cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))
    wait_until_cv2_window_exit("annotated image")

darknet.free_network_ptr(network)