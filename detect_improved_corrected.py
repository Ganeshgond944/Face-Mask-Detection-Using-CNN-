"""
Improved real-time face-mask detection using OpenCV + CNN.
"""

import os
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

import config
import utils


CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW = 5
FACE_PADDING_X = 0.10
FACE_PADDING_TOP = 0.10
FACE_PADDING_BOTTOM = 0.35
DETECT_EVERY_N_FRAMES = 2


def expand_box(x, y, w, h, image_width, image_height):
    """Expand mainly downward so the mask/chin area is included."""
    px = int(w * FACE_PADDING_X)
    top = int(h * FACE_PADDING_TOP)
    bottom = int(h * FACE_PADDING_BOTTOM)

    x1 = max(0, x - px)
    y1 = max(0, y - top)
    x2 = min(image_width, x + w + px)
    y2 = min(image_height, y + h + bottom)

    return x1, y1, x2 - x1, y2 - y1


def open_camera(logger):
    """Try common Windows camera backends and camera indexes."""
    for index in (0, 1, 2):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            success, frame = cap.read()
            if success and frame is not None:
                logger.info(
                    "Camera opened: index=%s backend=%s",
                    index,
                    backend,
                )
                return cap

            cap.release()

    return None


def get_labels(model):
    """
    Use labels configured in the project if available.
    Otherwise use the original project's expected order.
    """
    configured_labels = getattr(config, "CLASS_LABELS", None)

    if configured_labels and len(configured_labels) >= 2:
        return list(configured_labels[:2])

    # IMPORTANT:
    # Change this order only if your training code used the opposite order.
    return ["WithMask", "WithoutMask"]


def main():
    logger = utils.get_logger("FaceMaskImproved")

    cascade_path = utils.ensure_haar_cascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        logger.error("Could not load Haar Cascade: %s", cascade_path)
        sys.exit(1)

    model_path = (
        config.BEST_MODEL_PATH
        if os.path.exists(config.BEST_MODEL_PATH)
        else config.FINAL_MODEL_PATH
    )

    if not os.path.exists(model_path):
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    logger.info("Loading model: %s", model_path)
    model = load_model(model_path)
    labels = get_labels(model)
    logger.info("Using class labels: %s", labels)

    cap = open_camera(logger)
    if cap is None:
        logger.error("No working webcam found.")
        sys.exit(1)

    histories = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
    last_faces = []
    frame_number = 0
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            success, frame = cap.read()

            if not success or frame is None:
                continue

            frame_number += 1
            height, width = frame.shape[:2]

            if frame_number % DETECT_EVERY_N_FRAMES == 1 or not last_faces:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                detected_faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.10,
                    minNeighbors=6,
                    minSize=(80, 80),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                last_faces = list(detected_faces)

            face_inputs = []
            boxes = []

            for x, y, w, h in last_faces:
                ex, ey, ew, eh = expand_box(
                    x, y, w, h, width, height
                )

                roi = frame[ey:ey + eh, ex:ex + ew]

                if roi.size == 0:
                    continue

                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(
                    rgb,
                    tuple(config.IMG_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                normalized = resized.astype(np.float32) / 255.0

                face_inputs.append(normalized)
                boxes.append((ex, ey, ew, eh))

            if face_inputs:
                predictions = model.predict(
                    np.asarray(face_inputs),
                    verbose=0,
                )

                for index, (x, y, w, h) in enumerate(boxes):
                    probabilities = np.asarray(
                        predictions[index]
                    ).reshape(-1)

                    if probabilities.size < 2:
                        continue

                    probabilities = probabilities[:2].astype(
                        np.float32
                    )

                    # Handle sigmoid output as well as softmax output.
                    if (
                        np.min(probabilities) >= 0
                        and np.max(probabilities) <= 1
                        and abs(float(probabilities.sum()) - 1.0) > 0.05
                    ):
                        p = float(probabilities[0])
                        probabilities = np.array(
                            [1.0 - p, p],
                            dtype=np.float32,
                        )

                    total = float(probabilities.sum())
                    if total > 0:
                        probabilities /= total

                    center_x = x + (w // 2)
                    center_y = y + (h // 2)
                    track_id = (center_x // 60, center_y // 60)

                    histories[track_id].append(probabilities)
                    averaged = np.mean(
                        np.asarray(histories[track_id]),
                        axis=0,
                    )

                    predicted_index = int(np.argmax(averaged))
                    confidence = float(averaged[predicted_index])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        label = labels[predicted_index]
                        is_mask = (
                            "mask" in label.lower()
                            and "without" not in label.lower()
                            and "no" not in label.lower()
                        )

                        color = (0, 180, 0) if is_mask else (0, 0, 220)
                        text = (
                            f"{label}: "
                            f"{confidence * 100:.1f}%"
                        )
                    else:
                        color = (0, 180, 220)
                        text = (
                            f"Uncertain: "
                            f"{confidence * 100:.1f}%"
                        )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2,
                    )

                    text_y = max(30, y - 8)
                    (text_width, text_height), baseline = (
                        cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            2,
                        )
                    )

                    cv2.rectangle(
                        frame,
                        (
                            x,
                            text_y - text_height - baseline - 8,
                        ),
                        (
                            x + text_width + 10,
                            text_y + 2,
                        ),
                        color,
                        -1,
                    )

                    cv2.putText(
                        frame,
                        text,
                        (x + 5, text_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            fps = (
                0.9 * fps + 0.1 * instant_fps
                if fps
                else instant_fps
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Faces: {len(boxes)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Improved Face Mask Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

    except Exception:
        logger.exception("Unexpected error during webcam detection")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released.")


if __name__ == "__main__":
    main()
