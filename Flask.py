import os
from flask import Flask, render_template,request,jsonify,url_for, session

import numpy as np
import pandas as pd
import sqlite3
from ultralytics import YOLO
from PIL import Image

from transformers import ViTImageProcessor, ViTForImageClassification # ViTFeatureExtractor
from PIL import Image

import json

import warnings
warnings.filterwarnings('ignore')

import joblib


app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

upload_folder = "D:/kofir/Documents/Mini Project 3/static/images"
app.config['UPLOAD_FOLDER'] = upload_folder

@app.route('/')
def hello():

    return render_template('Home.html')


@app.route("/yolo",methods=["GET","POST"]) 
def yolo():

    return render_template("Yolo.html")

@app.route('/vit', methods=["GET",'POST'])
def vit():
  
        return render_template("vit.html")
    

@app.route('/uploadYolo', methods=["GET",'POST'])
def uploadYolo():
    file = request.files['imgFile']

    if file.filename == '':
        return "No file selected"
    
    if file:
        name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
        
        message = "File was uploaded" # Message updated on the html page

        filenames = session.get('uploads', [])

        filenames.append(name)

        session['uploads'] = filenames

    return render_template("Yolo.html", Message = message)

@app.route('/uploadVit', methods=["GET",'POST'])
def uploadVit():
    file = request.files['imgFile']

    if file.filename == '':
        return "No file selected"
    
    if file:
        name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
        
        message = "File was uploaded" # Message updated on the html page

        filenames = session.get('uploads', [])

        filenames.append(name)

        session['uploads'] = filenames

    return render_template("Vit.html", Message = message)

@app.route('/yoloclass', methods=["GET",'POST'])
def yoloclass():
     filenames = session.get('uploads', [])  # Retrieve the filenames list from the session

     if filenames:
        file = filenames[-1]  # Access the most recently uploaded file
        

        path = f"static\images\{file}"
        model = YOLO('yolov8n.pt')


        results = model(path,verbose=False)  # results list

        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save('static/images/results.png')  # save image
     return render_template("yoloclass.html")

@app.route('/vitclass', methods=["GET",'POST'])
def vitclass():
    filenames = session.get('uploads', [])  # Retrieve the filenames list from the session

    if filenames:
        file = filenames[-1]  # Access the most recently uploaded file
        #print(last_uploaded_file)  # Print the last uploaded file

        path = f"static\images\{file}"
        

        image = Image.open(path)

        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predict = logits.argmax(-1).item()
        prediction = model.config.id2label[predict]
        
   
    return render_template("vitclass.html", Prediction = prediction)
if __name__ == '__main__':
   
    app.run()

