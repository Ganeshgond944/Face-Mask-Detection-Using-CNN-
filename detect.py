"""
Real-time Face Mask Detection script using OpenCV webcam feed and trained CNN model.
Detects human faces via Haar Cascade, crops and normalizes ROI, feeds it into the CNN model,
draws Green bounding box for 'Mask' and Red for 'No Mask', displays confidence %, and calculates live FPS.
"""

import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

import config
import utils

def main():
    logger = utils.get_logger()
    logger.info("==================================================")
    logger.info("  FACE MASK DETECTION - REAL-TIME WEBCAM FEED    ")
    logger.info("==================================================")

    # 1. Ensure Haar Cascade XML is available
    cascade_path = utils.ensure_haar_cascade()
    if not os.path.exists(cascade_path):
        logger.error(f"Haar Cascade XML not found at {cascade_path}!")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logger.error("Failed to load Haar Cascade classifier!")
        sys.exit(1)

    logger.info(f"Loaded Haar Cascade face detector from: {cascade_path}")

    # 2. Load Trained Model
    model_path = config.BEST_MODEL_PATH if os.path.exists(config.BEST_MODEL_PATH) else config.FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}! Please run 'python train.py' first.")
        sys.exit(1)

    logger.info(f"Loading trained CNN model from: {model_path}")
    try:
        model = load_model(model_path)
        logger.info("Model successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Class mappings (Class 0: WithMask, Class 1: WithoutMask)
    class_labels = ["Mask", "No Mask"]

    # 3. Initialize Webcam with DirectShow backend for Windows compatibility
    logger.info("Initializing webcam stream... Press 'ESC' or 'q' to exit.")
    cap = None
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    camera_indices = [0, 1, 2]

    for index in camera_indices:
        for backend in backends:
            temp_cap = cv2.VideoCapture(index, backend)
            if temp_cap.isOpened():
                # Test reading a frame to verify camera access
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    cap = temp_cap
                    logger.info(f"Webcam successfully initialized on index {index} with backend {backend}.")
                    break
                else:
                    temp_cap.release()
        if cap is not None:
            break

    if cap is None or not cap.isOpened():
        logger.error("No working webcam stream available! Please check camera connection and privacy permissions.")
        sys.exit(1)

    # Frame Rate (FPS) Calculation Variables
    prev_frame_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to grab frame from webcam. Retrying...")
                time.sleep(0.1)
                continue

            # Convert frame to grayscale for Haar Cascade face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # Process each detected face ROI
            for (x, y, w, h) in faces:
                # Extract Face ROI
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                # Preprocess Face ROI for CNN Model
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, config.IMG_SIZE)
                face_normalized = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)

                # Prediction
                preds = model.predict(face_input, verbose=0)[0]
                pred_idx = np.argmax(preds)
                confidence = preds[pred_idx] * 100.0

                label_text = class_labels[pred_idx]

                # Bounding Box Color: Green for Mask, Red for No Mask
                color = (0, 255, 0) if pred_idx == 0 else (0, 0, 255)

                # Draw Bounding Box around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Label string: e.g. "Mask (99.34%)" or "No Mask (97.82%)"
                display_str = f"{label_text} ({confidence:.2f}%)"

                # Text Overlay Background Rectangle for contrast
                (text_w, text_h), baseline = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)

                # White text inside background rectangle
                cv2.putText(frame, display_str, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Calculate FPS
            curr_frame_time = time.time()
            fps = 1.0 / (curr_frame_time - prev_frame_time + 1e-6)
            prev_frame_time = curr_frame_time

            # Overlay FPS counter on top-left corner
            fps_str = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_str, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Display Output Window
            cv2.imshow("Real-Time Face Mask Detection", frame)

            # Exit condition: Press ESC (27) or 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                logger.info("User pressed exit key. Closing webcam feed...")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error during webcam detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam stream released and windows closed successfully.")

if __name__ == "__main__":
    main()
