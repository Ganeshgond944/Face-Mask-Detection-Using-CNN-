"""
Flask Web Application for Face Mask Detection using CNN.
Loads pre-trained Keras model once on startup and serves web routes for Home, Upload, Predict, Live Streaming, and About pages.
Includes optimized Multi-Person Face Detection & Batch Neural Inference for real-time webcam streams.
"""

import os
import sys
import time
import base64
import datetime
import cv2
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify, Response # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

import config
import utils

# ==============================================================================
# FLASK APP & DIRECTORY INITIALIZATION
# ==============================================================================
app = Flask(__name__)

# Ensure all static subdirectories exist
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(app.root_path, 'static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# Initialize logger and Haar Cascade
logger = utils.get_logger("FaceMaskWebApp")
cascade_path = utils.ensure_haar_cascade()
face_cascade = cv2.CascadeClassifier(cascade_path)

# ==============================================================================
# SINGLE-INSTANCE MODEL LOADING ON SERVER STARTUP
# ==============================================================================
MODEL_PATH = config.BEST_MODEL_PATH
if not os.path.exists(MODEL_PATH):
    alt_path = os.path.join(app.root_path, 'model', 'best_model.keras')
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path
    elif os.path.exists(config.FINAL_MODEL_PATH):
        MODEL_PATH = config.FINAL_MODEL_PATH

logger.info(f"Loading trained CNN model from: {MODEL_PATH}")
try:
    cnn_model = load_model(MODEL_PATH)
    logger.info("CNN Model successfully loaded into memory once on server startup.")
except Exception as e:
    logger.error(f"Critical Error: Failed to load trained CNN model: {e}")
    sys.exit(1)

CLASS_LABELS = ["WithMask", "WithoutMask"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================================================================
# WEB ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Home Page Route."""
    return render_template('index.html')

@app.route('/upload')
def upload():
    """Upload Image Route."""
    return render_template('upload.html')

@app.route('/about')
def about():
    """About Project Route."""
    return render_template('about.html')

@app.route('/live')
def live():
    """Live Video Stream Route."""
    return render_template('live.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image file upload, multi-person face detection via Haar Cascade, batch ROI preprocessing,
    CNN prediction using pre-loaded model, drawing colored bounding boxes for all persons, and rendering result.html.
    """
    start_time = time.time()

    if 'file' not in request.files:
        return render_template('upload.html', error='No file part selected.')

    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error='No file selected.')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        unique_filename = timestamp_str + filename

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        result_save_path = os.path.join(app.config['RESULTS_FOLDER'], unique_filename)
        file.save(save_path)

        img_bgr = cv2.imread(save_path)
        if img_bgr is None:
            return render_template('upload.html', error='Failed to decode uploaded image.')

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Multi-person Haar Cascade detection (tuned scaleFactor & minNeighbors for near/far faces)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(25, 25)
        )

        main_label = "WithoutMask"
        highest_confidence = 0.0
        faces_detected_count = len(faces)

        if faces_detected_count == 0:
            # Full image fallback if no face detected by Haar Cascade
            face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, config.IMG_SIZE)
            face_norm = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_norm, axis=0)

            preds = cnn_model.predict(face_input, verbose=0)[0]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx] * 100.0)

            main_label = CLASS_LABELS[pred_idx]
            highest_confidence = confidence

            color = (0, 255, 0) if pred_idx == 0 else (0, 0, 255)
            h, w, _ = img_bgr.shape
            cv2.rectangle(img_bgr, (10, 10), (w - 10, h - 10), color, 4)

            display_str = f"{'Mask' if pred_idx==0 else 'No Mask'} ({confidence:.2f}%)"
            (tw, th), _ = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_bgr, (10, 10), (10 + tw + 10, 10 + th + 10), color, -1)
            cv2.putText(img_bgr, display_str, (15, 20 + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Batch extraction & inference for multiple persons
            face_inputs = []
            valid_coords = []

            for (x, y, w, h) in faces:
                face_roi = img_bgr[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, config.IMG_SIZE)
                face_norm = face_resized.astype("float32") / 255.0
                face_inputs.append(face_norm)
                valid_coords.append((x, y, w, h))

            if len(face_inputs) > 0:
                batch_input = np.array(face_inputs)
                preds_batch = cnn_model.predict(batch_input, verbose=0)

                for idx, (x, y, w, h) in enumerate(valid_coords):
                    preds = preds_batch[idx]
                    pred_idx = np.argmax(preds)
                    confidence = float(preds[pred_idx] * 100.0)

                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        main_label = CLASS_LABELS[pred_idx]

                    color = (0, 255, 0) if pred_idx == 0 else (0, 0, 255)
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 3)

                    display_str = f"{'Mask' if pred_idx==0 else 'No Mask'} ({confidence:.1f}%)"
                    (tw, th), _ = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img_bgr, (x, y - th - 10), (x + tw + 10, y), color, -1)
                    cv2.putText(img_bgr, display_str, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Save output image to static/results/
        cv2.imwrite(result_save_path, img_bgr)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        formatted_time = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")

        return render_template(
            'result.html',
            result_filename=unique_filename,
            main_label=main_label,
            main_confidence=round(highest_confidence, 2),
            faces_count=faces_detected_count,
            processing_time=elapsed_ms,
            timestamp=formatted_time
        )

    return render_template('upload.html', error='Invalid file format. Only JPG, JPEG, and PNG are allowed.')

# ==============================================================================
# FAST MULTI-PERSON WEBCAM JSON PREDICTION API
# ==============================================================================
@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Receives base64 frame from browser tab HTML5 video/canvas, performs Multi-Person Haar Cascade
    face detection, executes batch CNN neural inference, and returns bounding box JSON for all persons.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'success': False, 'error': 'Image decoding failed'}), 400

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Multi-Person Face Detection Parameters (ScaleFactor=1.08, MinNeighbors=4, MinSize=(25,25))
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(25, 25)
        )

        if len(faces) == 0:
            return jsonify({'success': True, 'faces': [], 'faces_count': 0})

        face_inputs = []
        valid_coords = []

        for (x, y, w, h) in faces:
            face_roi = img_bgr[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, config.IMG_SIZE)
            face_norm = face_resized.astype("float32") / 255.0
            face_inputs.append(face_norm)
            valid_coords.append((x, y, w, h))

        if len(face_inputs) == 0:
            return jsonify({'success': True, 'faces': [], 'faces_count': 0})

        # Batch Neural Inference for all persons in a single call
        batch_input = np.array(face_inputs)
        preds_batch = cnn_model.predict(batch_input, verbose=0)

        results = []
        for idx, (x, y, w, h) in enumerate(valid_coords):
            preds = preds_batch[idx]
            pred_idx = np.argmax(preds)
            confidence = float(preds[pred_idx] * 100.0)

            label = "Mask" if pred_idx == 0 else "No Mask"
            color = "#16a34a" if pred_idx == 0 else "#dc2626"  # Emerald Green vs Red

            results.append({
                'x': int(x),
                'y': int(y),
                'w': int(w),
                'h': int(h),
                'label': label,
                'confidence': round(confidence, 2),
                'color': color
            })

        return jsonify({
            'success': True,
            'faces': results,
            'faces_count': len(results)
        })

    except Exception as e:
        logger.error(f"Error processing multi-person frame: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    logger.info("Starting Face Mask Detection Flask Web Server on http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
