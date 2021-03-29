import os
import glob
import io
import requests
from flask import Flask, url_for, send_file, escape, request, render_template, redirect, jsonify, make_response, \
    session, send_from_directory
from flask_cors import CORS
from DetectAndCrop import detect_and_crop
from ImageClassifier import compare_images
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import time

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

URL = '127.0.0.1:5000'

model_weights = 'data/weights/yolo-custom'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

saved_model_loaded = tf.saved_model.load(model_weights, tags=[tag_constants.SERVING])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image1, image2):
    # Detect and cut the stamp

    stamp_1, stamp_1_name = detect_and_crop(image1, saved_model_loaded, app.config['UPLOAD_FOLDER'])
    stamp_2, stamp_2_name = detect_and_crop(image2, saved_model_loaded, app.config['UPLOAD_FOLDER'])
    # Compare 2 stamp if matching
    if stamp_1 == '' or stamp_2 == '':
        result_detect = 'false_detect'
        matched = ''
        matching_image_name = ''
        stamp1 = stamp_1_name
        stamp2 = stamp_2_name
        return result_detect, matched, matching_image_name, stamp1, stamp2
    else:
        result_detect = 'true'
        matched, matching_image_name = compare_images(stamp_1, stamp_2, app.config['UPLOAD_FOLDER'])
        stamp1 = stamp_1_name
        stamp2 = stamp_2_name
        return result_detect, matched, matching_image_name, stamp1, stamp2


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("public/upload_image.html")

@app.route('/stamp-rcg/file/<path:filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/stamp-rcg/predict', methods=['GET', 'POST'])
def steam_rcg():
    new_image_1 = str(time.time())+'_image1.png'
    new_image_2 = str(time.time())+'_image2.png'
    image1 = os.path.join(app.config["UPLOAD_FOLDER"], new_image_1)
    image2 = os.path.join(app.config["UPLOAD_FOLDER"], new_image_2)
    if request.method == "POST":
        if request.files:
            images = request.files["image"]
            images.save(image1)
            images2 = request.files["image2"]
            images2.save(image2)
    result_detect, matched, matching_image_name, stamp_1, stamp_2 = predict(image1, image2)
    if result_detect == 'false_detect':
        return jsonify({'status': 0, 'message': 'Không phát hiện con dấu trong ảnh đã tải lên'})
    else:
        if (matched == True):
            return jsonify(
                {'status': 1, 'stamp_1': stamp_1, 'stamp_2': stamp_2, 'result_img': matching_image_name, 'isMatch': matched})
        else:
            # return redirect(url_for('index', matching_image=matching_image, message="Hai con dấu khác nhau"))
            return jsonify(
                {'status': 0, 'stamp_1': stamp_1, 'stamp_2': stamp_2, 'result_img': matching_image_name, 'isMatch': matched})


if __name__ == '__main__':
    app.run(debug=True)
