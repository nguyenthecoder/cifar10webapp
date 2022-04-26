from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
from werkzeug.utils import secure_filename
import numpy as np
from ml.models import ModelPrediction
from utils.image import save_img_to_buffer
import os
import pathlib

# keras_model.hello
# keras_model.get_model()

CWD = pathlib.Path(__file__).parent.resolve()
BUFFER = 'buffer'
STATIC = 'static'
BUFFER_DIR = os.path.join(CWD, STATIC, BUFFER)

ml_model = ModelPrediction('vgg16_base')

app = Flask(__name__)

@app.route('/')
def helloworld():
    return 'hello world'

@app.route('/test/<param>')
def printParam(param):
    return 'hello %s' % param

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['name']
        return redirect(url_for('success', name=user))
    if request.method == 'GET':
        return render_template('login.html')

@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

@app.route('/conditional_render/<int:score>')
def condition_render(score):
    dict = {'score': score, 'name': 'Jame Nguyen'}
    return render_template('conditional_render.html', data = dict)

@app.route('/upload', methods=["POST", 'GET'])
def upload():
    if request.method == 'POST':
        print('uploaded a file')
        file = request.files['file']

        img_uuid = save_img_to_buffer(file, BUFFER_DIR )
        return redirect('/predict/' + img_uuid)
    if request.method == 'GET':
        return render_template('upload.html')

@app.route('/predict/<img_uuid>')
def predict(img_uuid):
    data = dict()

    for file in os.listdir(BUFFER_DIR):
        print(file)
        if file.startswith(img_uuid):
            #use file and predict the result
            data['image'] = os.path.join(BUFFER, file)
            image_path = os.path.join(BUFFER_DIR, file)
            label = ml_model.predict(image_path)
            data['label'] = label
            break
    print(data)
    return render_template('predict.html', data = data)

if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    app.run( debug = False, host = "0.0.0.0", port = port)