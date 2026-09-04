import base64
from io import BytesIO
import os
import time
import math
from threading import Lock
from flask import Flask, render_template, request, jsonify, url_for, session  # type: ignore

# Keep library-generated settings and caches local to this app.
runtime_folder = os.path.join(os.path.dirname(__file__), '.runtime')
os.makedirs(runtime_folder, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', runtime_folder)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(runtime_folder, 'matplotlib'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

from PIL import Image, UnidentifiedImageError
# Preserve Pillow's decoder before Ultralytics installs its HEIF fallback.
# Browser frames only need standard image formats, not automatic codec installs.
open_image = Image.open
from ultralytics import YOLO

from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import warnings
warnings.filterwarnings('ignore')

# Load ViT once
feature_extractor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)
vit_model.eval()

yolo_model = YOLO(os.path.join(os.path.dirname(__file__), 'yolov8n.pt'))
yolo_lock = Lock()

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

upload_folder = os.path.join(app.root_path, "static", "images")
os.makedirs(upload_folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 17 * 1024 * 1024


@app.route('/')
def hello():
    return render_template('Home.html')


@app.route("/yolo", methods=["GET", "POST"])
def yolo():
    return render_template("Yolo.html")


@app.post('/api/yolo/frame')
def yolo_frame():
    """Process one browser-sampled frame without storing camera/video media."""
    if request.content_length and request.content_length > 2 * 1024 * 1024:
        return jsonify(error='Frame too large. Maximum request size is 2 MB.'), 413
    file = request.files.get('frame')
    if file is None:
        return jsonify(error='A frame is required.'), 400
    try:
        confidence = float(request.form.get('confidence', '0.35'))
        if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
            return jsonify(error='Confidence must be between 0.1 and 0.9.'), 400
        with open_image(file.stream, formats=['JPEG', 'PNG', 'WEBP']) as source:
            if source.width * source.height > 4_000_000:
                return jsonify(error='Frame resolution is too large.'), 413
            frame = source.convert('RGB')
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return jsonify(error='Provide a valid image frame and confidence value.'), 400
    started = time.perf_counter()
    try:
        with yolo_lock:
            result = yolo_model(frame, conf=confidence, imgsz=640, verbose=False)[0]
        detections = [
            {'class': result.names[int(box.cls.item())],
             'confidence': float(box.conf.item()),
             'bbox': box.xyxy[0].tolist()}
            for box in result.boxes
        ] if result.boxes is not None else []
        return jsonify(detections=detections, width=frame.width, height=frame.height,
                       time=time.perf_counter() - started)
    except Exception:
        app.logger.exception('YOLO frame inference failed')
        return jsonify(error='Detection failed. Please stop and try again.'), 500


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
        return render_template('Compare.html')
    
    if request.method == 'POST':
        try:
            # Get file
            if 'imgFile' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['imgFile']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file'}), 400
            
            # Decode in memory: never overwrite or delete an existing image.
            with Image.open(file.stream) as uploaded_image:
                image = uploaded_image.convert("RGB")
            
            # --- Process with YOLO ---
            yolo_start = time.time()
            with yolo_lock:
                yolo_results = yolo_model(image, verbose=False)
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
            with torch.inference_mode():
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
    app.run(host='127.0.0.1', port=5000, debug=False)
