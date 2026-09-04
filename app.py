"""Model Compare: pages and bounded, in-memory image inference."""
import base64
from functools import wraps
from io import BytesIO
import math
import os
from pathlib import Path
import secrets
from threading import BoundedSemaphore
import time
from urllib.parse import urlsplit

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / '.runtime'
RUNTIME.mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(RUNTIME))
os.environ.setdefault('MPLCONFIGDIR', str(RUNTIME / 'matplotlib'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'true')

# Keep the standard decoder: Ultralytics' fallback can install extra codecs.
open_image = Image.open
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

production = os.environ.get('APP_ENV') == 'production'
secret = os.environ.get('SECRET_KEY')
if production and (not secret or len(secret) < 32):
    raise RuntimeError('Set SECRET_KEY to a random value of at least 32 characters.')

app = Flask(__name__, static_folder=None)
app.config.update(
    SECRET_KEY=secret or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=17 * 1024 * 1024,
    MAX_FORM_MEMORY_SIZE=512 * 1024,
    MAX_FORM_PARTS=8,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=production,
)

# The deployment image contains these files; local use retains the HF cache.
model_source = os.environ.get('VIT_MODEL_PATH', 'google/vit-base-patch16-224')
feature_extractor = ViTImageProcessor.from_pretrained(model_source)
vit_model = ViTForImageClassification.from_pretrained(model_source)
vit_model.eval()
yolo_model = YOLO(str(ROOT / 'yolov8n.pt'))
# One inference request at a time bounds memory and protects shared models.
inference_slot = BoundedSemaphore(1)

PUBLIC_ASSETS = {
    'css/styles.css', 'css/compare.css', 'css/yolo.css',
    'js/compare.js', 'js/yolo.js', 'images/car1.jpg',
}


@app.get('/static/<path:filename>')
def static_asset(filename):
    if filename not in PUBLIC_ASSETS:
        abort(404)
    return send_from_directory(ROOT / 'static', filename)


@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(self), microphone=()'
    if request.method == 'POST':
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.before_request
def reject_cross_site_uploads():
    if request.method == 'POST':
        origin = request.headers.get('Origin')
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            return jsonify(error='Cross-site uploads are not allowed.'), 403
        if origin:
            try:
                same_host = urlsplit(origin).netloc == request.host
            except ValueError:
                same_host = False
            if not same_host:
                return jsonify(error='Cross-site uploads are not allowed.'), 403


@app.errorhandler(HTTPException)
def http_error(error):
    messages = {
        400: 'Invalid request.', 404: 'Not found.', 405: 'Method not allowed.',
        413: 'Upload too large.', 415: 'Unsupported file type.',
    }
    return jsonify(error=messages.get(error.code, 'Request failed.')), error.code


def bounded_inference(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not inference_slot.acquire(blocking=False):
            return jsonify(error='The models are busy. Please try again shortly.'), 503, {'Retry-After': '1'}
        try:
            return function(*args, **kwargs)
        finally:
            inference_slot.release()
    return wrapped


def decode_image(field, formats, max_bytes, max_pixels):
    file = request.files.get(field)
    if file is None or not file.filename:
        raise ValueError('Choose an image first.')
    raw = file.stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        abort(413)
    try:
        with open_image(BytesIO(raw), formats=formats) as source:
            if source.width * source.height > max_pixels:
                raise ValueError('Image dimensions are too large.')
            image = source.convert('RGB')
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ValueError('This file is not a supported image.') from None
    return image


def image_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


@app.get('/')
def hello():
    return render_template('Home.html')


@app.get('/yolo')
def yolo():
    return render_template('Yolo.html')


@app.get('/healthz')
def health():
    # Imports and model loading have completed before this can return OK.
    return jsonify(status='ok')


@app.get('/compare')
def compare_page():
    return render_template('Compare.html')


@app.post('/api/yolo/frame')
@bounded_inference
def yolo_frame():
    if request.content_length and request.content_length > 2 * 1024 * 1024:
        abort(413)
    try:
        confidence = float(request.form.get('confidence', '0.35'))
        if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
            raise ValueError('Confidence must be between 0.1 and 0.9.')
        frame = decode_image('frame', ['JPEG', 'PNG', 'WEBP'], 2 * 1024 * 1024, 4_000_000)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    started = time.perf_counter()
    try:
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
        return jsonify(error='Detection failed. Please try again.'), 500


@app.post('/compare')
@bounded_inference
def compare():
    try:
        image = decode_image('imgFile', ['JPEG', 'PNG', 'GIF'], 16 * 1024 * 1024, 8_000_000)
        # Bound visualization size and response memory without changing aspect ratio.
        image.thumbnail((1600, 1600))
    except ValueError as error:
        return jsonify(error=str(error)), 400
    try:
        started = time.perf_counter()
        result = yolo_model(image, verbose=False)[0]
        yolo_time = time.perf_counter() - started
        detections = [
            {'class': result.names[int(box.cls.item())],
             'confidence': float(box.conf.item()),
             'bbox': box.xyxy.tolist()}
            for box in result.boxes
        ] if result.boxes is not None else []
        yolo_image = Image.fromarray(result.plot()[..., ::-1])

        started = time.perf_counter()
        inputs = feature_extractor(images=image, return_tensors='pt')
        with torch.inference_mode():
            outputs = vit_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities, 5)
        vit_time = time.perf_counter() - started
        predictions = [
            {'class': vit_model.config.id2label[index.item()], 'confidence': float(prob.item())}
            for prob, index in zip(top_probs[0], top_indices[0])
        ]
        return jsonify(
            yolo={'detections': detections, 'time': yolo_time, 'image': image_base64(yolo_image)},
            vit={'top_prediction': predictions[0]['class'],
                 'top_confidence': predictions[0]['confidence'],
                 'top_predictions': predictions, 'time': vit_time, 'image': image_base64(image)},
        )
    except Exception:
        app.logger.exception('Image comparison failed')
        return jsonify(error='Image analysis failed. Please try another image.'), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=False)
