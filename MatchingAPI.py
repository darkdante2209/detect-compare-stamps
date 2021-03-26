import os
import glob
import io
import requests
from flask import Flask, url_for, send_file, escape, request, render_template, redirect, jsonify, make_response, session,send_from_directory
from flask_cors import CORS
from DetectAndCrop import detect_and_crop
from ImageClassifier import compare_images
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

URL = '127.0.0.1:5000'

model_weights = 'data/weights/yolo-custom'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

saved_model_loaded = tf.saved_model.load(model_weights, tags=[tag_constants.SERVING])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(image1, image2):
    # Check image uploaded:
    files_exist = os.listdir(app.config["UPLOAD_FOLDER"])
    if len(files_exist)>2:
        return render_template("public/upload_image.html", message='Bạn đã upload nhiều hơn 2 ảnh, vui lòng chọn lại')
    if len(files_exist)==0:
        return render_template("public/upload_image.html", message='Bạn chưa upload ảnh nào, vui lòng chọn lại')


    stamp_detected_path = 'static/detected'
    files_detected = os.listdir(stamp_detected_path)
    for file in files_detected:
        os.remove(os.path.join(stamp_detected_path, file))
    matching_img_path = 'static/matching_img/matching_result.png'
    if os.path.exists(matching_img_path):
        os.remove(matching_img_path)

    # Detect and cut the stamp

    stamp_1 = detect_and_crop(image1, saved_model_loaded)
    stamp_2 = detect_and_crop(image2, saved_model_loaded)

    # Compare 2 stamp if matching

    matched, matching_image = compare_images(stamp_1, stamp_2)
    return matched, matching_image

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("public/upload_image.html")


@app.route('/upload-image-1', methods=['GET', 'POST'])
def upload1():
    files_exist = os.listdir(app.config["UPLOAD_FOLDER"])
    for file_exist in files_exist:
        if os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], file_exist)):
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], file_exist))
    image1 = os.path.join(app.config["UPLOAD_FOLDER"], 'image1.png')
    image2 = os.path.join(app.config["UPLOAD_FOLDER"], 'image2.png')
    if request.method == "POST":
        if request.files:
            images = request.files["image"]
            images.save(image1)
            images2 = request.files["image2"]
            images2.save(image2)
    matched, matching_image = predict(image1, image2)
    if (matched == True):
        # return redirect(url_for('index', matching_image=matching_image, message="Hai con dấu hoàn toàn trùng khớp"))
        return render_template("public/upload_image.html", matching_image=matching_image,
                               message="Hai con dấu hoàn toàn trùng khớp")
    else:
        # return redirect(url_for('index', matching_image=matching_image, message="Hai con dấu khác nhau"))
        return render_template("public/upload_image.html", matching_image=matching_image,
                               message="Hai con dấu khác nhau")

if __name__ == '__main__':
    app.run(debug=True)