import base64
from io import BytesIO
import os
import time
from flask import Flask, render_template, request, jsonify, url_for, session  # type: ignore

import numpy as np
import pandas as pd
import sqlite3
from ultralytics import YOLO
from PIL import Image

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import json
import warnings
warnings.filterwarnings('ignore')

import joblib

# Load ViT once
feature_extractor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)

yolo_model = YOLO('yolov8n.pt')

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

upload_folder = os.path.join(app.root_path, "static", "images")
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder


@app.route('/')
def hello():
    return render_template('Home.html')


@app.route("/yolo", methods=["GET", "POST"])
def yolo():
    return render_template("Yolo.html")


@app.route('/vit', methods=["GET", 'POST'])
def vit():
    return render_template("vit.html")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploadYolo', methods=["GET", 'POST'])
def uploadYolo():
    if 'imgFile' not in request.files:
        return "No file part", 400
        
    file = request.files['imgFile']
    
    if file.filename == '':
        return "No file selected", 400
        
    if file and allowed_file(file.filename):
        name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))

        message = "File was uploaded"

        filenames = session.get('uploads', [])
        filenames.append(name)
        session['uploads'] = filenames

    return render_template("Yolo.html", Message=message)


@app.route('/uploadVit', methods=["GET", 'POST'])
def uploadVit():
    file = request.files['imgFile']

    if file.filename == '':
        return "No file selected"

    if file:
        name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))

        message = "File was uploaded"

        filenames = session.get('uploads', [])
        filenames.append(name)
        session['uploads'] = filenames

    return render_template("Vit.html", Message=message)


@app.route('/yoloclass', methods=["GET", 'POST'])
def yoloclass():
    filenames = session.get('uploads', [])

    if filenames:
        file = filenames[-1]
        path = f"static/images/{file}"  # Fixed: use forward slash
        model = YOLO('yolov8n.pt')

        results = model(path, verbose=False)

        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            im.save('static/images/results.png')

    return render_template("yoloclass.html")


@app.route('/vitclass', methods=["POST"])
def vitclass():
    filenames = session.get('uploads', [])

    if filenames:
        file = filenames[-1]
        path = os.path.join("static", "images", file)

        image = Image.open(path).convert("RGB")

        inputs = feature_extractor(
            images=image,
            return_tensors="pt"
        )

        outputs = vit_model(**inputs)
        prediction_index = outputs.logits.argmax(-1).item()
        prediction = vit_model.config.id2label[prediction_index]

    return render_template("vitclass.html", Prediction=prediction)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'GET':
        return render_template('compare.html')
    
    if request.method == 'POST':
        try:
            # Get file
            if 'imgFile' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['imgFile']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file'}), 400
            
            # Save file temporarily
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load image
            image = Image.open(filepath).convert("RGB")
            
            # --- Process with YOLO ---
            yolo_start = time.time()
            yolo_results = yolo_model(filepath, verbose=False)
            yolo_time = time.time() - yolo_start
            
            # Extract YOLO detections
            detections = []
            for r in yolo_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        detections.append({
                            'class': r.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy.tolist()
                        })
            
            # Generate YOLO visualization
            yolo_img = None
            for r in yolo_results:
                im_array = r.plot()
                yolo_pil = Image.fromarray(im_array[..., ::-1])
                buffered = BytesIO()
                yolo_pil.save(buffered, format="PNG")
                yolo_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
                break
            
            # --- Process with ViT ---
            vit_start = time.time()
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = vit_model(**inputs)
            vit_time = time.time() - vit_start
            
            # Get top 5 predictions
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            top_predictions = []
            for i in range(5):
                top_predictions.append({
                    'class': vit_model.config.id2label[top5_indices[0][i].item()],
                    'confidence': float(top5_probs[0][i])
                })
            
            # ViT image (original)
            buffered_vit = BytesIO()
            image.save(buffered_vit, format="PNG")
            vit_img = base64.b64encode(buffered_vit.getvalue()).decode('utf-8')
            
            # Clean up temp file
            os.remove(filepath)
            
            # Return results
            return jsonify({
                'yolo': {
                    'detections': detections,
                    'time': yolo_time,
                    'image': yolo_img
                },
                'vit': {
                    'top_prediction': top_predictions[0]['class'],
                    'top_confidence': top_predictions[0]['confidence'],
                    'top_predictions': top_predictions,
                    'time': vit_time,
                    'image': vit_img
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)