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

# Detection settings tuned to reduce false positives from background objects.
FACE_MIN_SIZE = (60, 60)
FACE_SCALE_FACTOR = 1.10
FACE_MIN_NEIGHBORS = 6
FACE_PADDING_X = 0.10
FACE_PADDING_TOP = 0.10
FACE_PADDING_BOTTOM = 0.35


def expand_face_box(x, y, w, h, image_width, image_height):
    """Expand the detected face mostly downward to include the mask/chin."""
    pad_x = int(w * FACE_PADDING_X)
    pad_top = int(h * FACE_PADDING_TOP)
    pad_bottom = int(h * FACE_PADDING_BOTTOM)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)
    x2 = min(image_width, x + w + pad_x)
    y2 = min(image_height, y + h + pad_bottom)

    return x1, y1, x2 - x1, y2 - y1


def detect_realistic_faces(image_bgr):
    """
    Detect faces more reliably when the lower face is covered by a mask.
    Uses a tolerant Haar pass and filters out wide rectangular background
    detections such as towels, tables, and clothes.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    image_height, image_width = image_bgr.shape[:2]

    # A tolerant pass is important because a mask hides the nose and mouth.
    raw_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    valid_faces = []

    for x, y, w, h in raw_faces:
        aspect_ratio = w / float(max(h, 1))
        area_ratio = (w * h) / float(max(image_width * image_height, 1))

        # Reject likely background rectangles while allowing slightly
        # non-square face boxes caused by masks, head angle, or perspective.
        if not 0.55 <= aspect_ratio <= 1.80:
            continue
        if area_ratio < 0.001:
            continue

        candidate = (int(x), int(y), int(w), int(h))

        # Remove duplicate/overlapping detections.
        duplicate = False
        for old_x, old_y, old_w, old_h in valid_faces:
            cx = x + w / 2.0
            cy = y + h / 2.0
            old_cx = old_x + old_w / 2.0
            old_cy = old_y + old_h / 2.0
            distance = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
            if distance < 0.35 * max(w, h, old_w, old_h):
                duplicate = True
                break

        if not duplicate:
            valid_faces.append(candidate)

    # Keep the largest plausible face when several weak detections occur.
    # This is especially useful for a single-person webcam view.
    if len(valid_faces) > 1:
        valid_faces.sort(
            key=lambda box: box[2] * box[3],
            reverse=True
        )

    return valid_faces


def prepare_face_input(image_bgr, box):
    """Create a padded, normalized RGB input for the CNN."""
    image_height, image_width = image_bgr.shape[:2]
    x, y, w, h = box
    ex, ey, ew, eh = expand_face_box(
        x, y, w, h, image_width, image_height
    )

    roi = image_bgr[ey:ey + eh, ex:ex + ew]
    if roi.size == 0:
        return None, None

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        tuple(config.IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    normalized = resized.astype("float32") / 255.0

    return normalized, (ex, ey, ew, eh)


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

        faces = detect_realistic_faces(img_bgr)

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

        for face_box in faces:
            face_input, expanded_box = prepare_face_input(
                img_bgr, face_box
            )
            if face_input is None:
                continue

            face_inputs.append(face_input)
            valid_coords.append(expanded_box)

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

        faces = detect_realistic_faces(img_bgr)

        if len(faces) == 0:
            return jsonify({'success': True, 'faces': [], 'faces_count': 0})

        face_inputs = []
        valid_coords = []

        for face_box in faces:
            face_input, expanded_box = prepare_face_input(
                img_bgr, face_box
            )
            if face_input is None:
                continue

            face_inputs.append(face_input)
            valid_coords.append(expanded_box)

        if len(face_inputs) == 0:
            return jsonify({'success': True, 'faces': [], 'faces_count': 0})

        # Batch Neural Inference for all persons in a single call
        batch_input = np.array(face_inputs)
        preds_batch = cnn_model.predict(batch_input, verbose=0)

        results = []
        for idx, (x, y, w, h) in enumerate(valid_coords):
            preds = np.asarray(preds_batch[idx]).reshape(-1)

            if len(preds) < 2:
                continue

            preds = preds[:2].astype("float32")
            total = float(preds.sum())
            if total > 0:
                preds = preds / total

            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx] * 100.0)

            label = "Mask" if pred_idx == 0 else "No Mask"
            color = "#16a34a" if pred_idx == 0 else "#dc2626"

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
