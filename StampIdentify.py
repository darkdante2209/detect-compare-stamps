import os
import glob
import io
import requests
from flask import Flask, url_for, send_file, escape, request, render_template, redirect, jsonify, make_response, \
    session, send_from_directory
from flask_cors import CORS
from DetectAndCrop import detect_and_crop
from ImageClassifier import identify_images
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import time

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'ImageTrain'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

TEST_FOLDER = 'image'
app.config['TEST_FOLDER'] = TEST_FOLDER

URL = '127.0.0.1:5000'

model_weights = 'data/weights/yolo-custom'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

saved_model_loaded = tf.saved_model.load(model_weights, tags=[tag_constants.SERVING])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    labels = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('public/identify.html', labels=labels)

@app.route('/stamp-identify/upload', methods=['GET', 'POST'])
def upload():
    new_label_path = ''
    new_image_name = ''
    if request.method == 'POST':
        if request.form['new_label'] != '':
            new_label = request.form['new_label']
            new_label_path = os.path.join(app.config['UPLOAD_FOLDER'], new_label)
            new_image_name = str(time.time())+'.png'
            if os.path.exists(new_label_path):
                images = request.files["image"]
                images.save(os.path.join(new_label_path, new_image_name))
            else:
                os.makedirs(new_label_path)
                images = request.files["image"]
                images.save(os.path.join(new_label_path, new_image_name))
        else:
            new_label_path = os.path.join(app.config['UPLOAD_FOLDER'], request.form['labels'])
            new_image_name = str(time.time()) + '.png'
            images = request.files["image"]
            images.save(os.path.join(new_label_path, new_image_name))
    image_path = os.path.join(new_label_path, new_image_name)
    return jsonify(
        {'status': 1, 'image': image_path})

@app.route('/stamp-identify/identify', methods=['GET', 'POST'])
def identify():
    Data_path = 'ImageTrain/*/*'
    new_test_image = str(time.time()) + '_test.png'
    image_test = os.path.join(app.config["UPLOAD_FOLDER"], new_test_image)
    if request.method == 'POST':
        test_images = request.files["test_image"]
        test_images.save(image_test)
    stamp_test, stamp_test_name = detect_and_crop(image_test, saved_model_loaded, app.config['TEST_FOLDER'])
    matched, label_max, image_max, matching_image = identify_images(stamp_test, Data_path, app.config['UPLOAD_FOLDER'])
    if matched:
        return jsonify(
            {'status': 1, 'matched': matched, 'label': label_max, 'image_max': image_max, 'stamp_test': stamp_test_name, 'matching_image': matching_image})
    else:
        return jsonify(
            {'status': 0, 'matched': matched, 'stamp': stamp_test_name}
        )

@app.route('/stamp-identify/file/<path:filename>')
def send_file(filename):
    return send_from_directory(app.config['TEST_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=2209)