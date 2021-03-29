import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
import os
import glob2
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Config YOLO variable:
YOLO_CLASSES = 'data/classes/yolo.names'
images_input = 'tested'
images_resize = 416
model_weights = 'data/weights/yolo-custom'
iou_threshold = 0.45
score_threshold = 0.50


def get_anchors(anchors_path):
    anchors = np.array(anchors_path)
    return anchors.reshape(3, 3, 2)


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def crop_objects(img_original_name, output_folder, img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(YOLO_CLASSES)
    # create dictionary to hold count of objects for image name
    counts = dict()
    img_name = ''
    stamp_name = ''
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
            # construct image name and join it to path for saving crop properly
            stamp_name = 'stamp_detected_' + img_original_name
            img_name = output_folder + '/' + stamp_name
            # save image
            cv2.imwrite(img_name, cropped_img)
        else:
            continue
    return img_name, stamp_name


def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes


def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = images_resize
    images_list = []
    for ext in ["*.png", "*.jpeg", "*.jpg"]:
        image_file = glob2.glob(os.path.join(images_input, ext))
        images_list += image_file
    print(images_list)

    # load model
    saved_model_loaded = tf.saved_model.load(model_weights, tags=[tag_constants.SERVING])
    # loop through images in list and run Yolov4 model on each
    for image_raw in images_list:
        re_name = image_raw.replace("\\", "/")
        original_image = cv2.imread(re_name)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        # get image name by using split method
        image_name = re_name.split('/')[-1]
        image_original_name = re_name.split('/')[-1]
        image_name = image_name.split('.')[0]

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = read_class_names(YOLO_CLASSES)
        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image
        crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
        crop_objects(image_original_name, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path,
                     allowed_classes)
        # image = Image.fromarray(image.astype(np.uint8))
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(output_folder+'/'+image_original_name, image)


def load_model():
    # load model
    saved_model_loaded = tf.saved_model.load(model_weights, tags=[tag_constants.SERVING])
    return saved_model_loaded


def detect_and_crop(image_input, saved_model_loaded, output_folder):
    output_folder = output_folder
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = images_resize

    saved_model_loaded = saved_model_loaded

    re_name = image_input.replace("\\", "/")
    print(image_input)
    original_image = cv2.imread(re_name)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # get image name by using split method
    image_name = re_name.split('/')[-1]
    image_original_name = re_name.split('/')[-1]
    image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    original_h, original_w, _ = original_image.shape
    bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    class_names = read_class_names(YOLO_CLASSES)
    allowed_classes = list(class_names.values())
    crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
    new_image_name, stamp_name = crop_objects(image_original_name, output_folder, cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                                  pred_bbox, crop_path, allowed_classes)
    return new_image_name, stamp_name
# if __name__ == '__main__':
#     main()
