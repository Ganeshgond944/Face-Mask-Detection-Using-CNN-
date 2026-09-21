"""
Utilities module for Face Mask Detection Project.
Includes logging, directory initialization, dataset statistics computation,
CNN model architecture creation, graph plotting, report generation, and prediction tools.
"""

import os
import sys
import time
import json
import logging
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support

import config

# ==============================================================================
# AUTOMATIC DIRECTORY CREATION & HAAR CASCADE INITIALIZATION
# ==============================================================================
def create_directories():
    """Creates all required output and model directories if they do not exist."""
    directories = [
        config.MODELS_DIR,
        config.OUTPUT_DIR,
        config.GRAPHS_DIR,
        config.CONF_MATRIX_DIR,
        config.PREDICTIONS_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
        config.SCREENSHOTS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
def get_logger(name="FaceMaskDetection"):
    """
    Sets up a custom logger that outputs to both console and a log file.
    """
    create_directories()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(config.LOG_FILE_PATH, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

logger = get_logger()

def ensure_haar_cascade():
    """
    Ensures that the haarcascade_frontalface_default.xml file is present in the project directory.
    If missing, attempts copying from OpenCV or downloads from official OpenCV repository.
    """
    create_directories()
    dest_path = config.HAAR_CASCADE_PATH
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    logger.info("Local Haar Cascade XML not found. Attempting to acquire XML...")
    try:
        cascade_src = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_src):
            shutil.copy(cascade_src, dest_path)
            logger.info(f"Successfully copied Haar Cascade XML to {dest_path}")
            return dest_path
    except Exception as e:
        logger.warning(f"Could not copy OpenCV Haar Cascade file locally: {e}")

    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        logger.info(f"Downloading Haar Cascade XML from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded Haar Cascade XML to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to download Haar Cascade XML: {e}")

    return dest_path

# ==============================================================================
# DATASET SCANNING & STATISTICS
# ==============================================================================
def get_dataset_stats():
    """
    Scans the dataset directory and returns split image counts and class counts.
    Automatically detects subfolder names for Mask and No Mask.
    """
    splits = {"Train": config.TRAIN_DIR, "Validation": config.VAL_DIR, "Test": config.TEST_DIR}
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    stats = {
        "splits": {},
        "class_counts": {"WithMask": 0, "WithoutMask": 0},
        "total_images": 0
    }

    for split_name, split_path in splits.items():
        stats["splits"][split_name] = {"WithMask": 0, "WithoutMask": 0, "total": 0}
        if not os.path.exists(split_path):
            logger.warning(f"Directory not found for split '{split_name}': {split_path}")
            continue

        for item in os.listdir(split_path):
            item_path = os.path.join(split_path, item)
            if os.path.isdir(item_path):
                # Auto-detect whether folder is WithMask or WithoutMask
                item_lower = item.lower()
                matched_class = None
                if any(m.lower() in item_lower for m in config.CLASS_SUBFOLDERS["MASK"]):
                    matched_class = "WithMask"
                elif any(m.lower() in item_lower for m in config.CLASS_SUBFOLDERS["NO_MASK"]):
                    matched_class = "WithoutMask"

                if matched_class:
                    img_count = sum(1 for f in os.listdir(item_path) if f.lower().endswith(valid_extensions))
                    stats["splits"][split_name][matched_class] += img_count
                    stats["splits"][split_name]["total"] += img_count
                    stats["class_counts"][matched_class] += img_count
                    stats["total_images"] += img_count

    return stats

def plot_dataset_distribution(stats):
    """
    Generates and saves a high-quality dataset distribution graph.
    """
    logger.info("Generating dataset distribution graphs...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # Chart 1: Split Distribution (Train, Validation, Test)
    split_names = list(stats["splits"].keys())
    mask_counts = [stats["splits"][s]["WithMask"] for s in split_names]
    no_mask_counts = [stats["splits"][s]["WithoutMask"] for s in split_names]

    x = np.arange(len(split_names))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, mask_counts, width, label='With Mask', color='#2ecc71', edgecolor='black')
    rects2 = axes[0].bar(x + width/2, no_mask_counts, width, label='Without Mask', color='#e74c3c', edgecolor='black')

    axes[0].set_title('Image Count per Dataset Split', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(split_names, fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].bar_label(rects1, padding=3, fontsize=10)
    axes[0].bar_label(rects2, padding=3, fontsize=10)

    # Chart 2: Total Class Ratio (Pie Chart)
    labels = ['With Mask', 'Without Mask']
    sizes = [stats["class_counts"]["WithMask"], stats["class_counts"]["WithoutMask"]]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    axes[1].pie(
        sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'}
    )
    axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    save_path = os.path.join(config.GRAPHS_DIR, "dataset_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Dataset distribution graph saved to: {save_path}")

# ==============================================================================
# CNN MODEL BUILDER
# ==============================================================================
def build_cnn_model(input_shape=config.INPUT_SHAPE, num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE):
    """
    Creates and compiles the custom CNN architecture according to specifications:
    - Conv2D(32, ReLU) -> MaxPooling
    - Conv2D(64, ReLU) -> MaxPooling
    - Conv2D(128, ReLU) -> MaxPooling
    - Conv2D(256, ReLU) -> MaxPooling
    - Flatten
    - Dense(256, ReLU) -> Dropout(0.5)
    - Dense(128, ReLU) -> Dropout(0.5)
    - Output Dense(2, Softmax)
    """
    model = models.Sequential([
        # Stage 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='maxpool_1'),

        # Stage 2
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='maxpool_2'),

        # Stage 3
        layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        layers.MaxPooling2D((2, 2), name='maxpool_3'),

        # Stage 4
        layers.Conv2D(256, (3, 3), activation='relu', name='conv2d_4'),
        layers.MaxPooling2D((2, 2), name='maxpool_4'),

        # Dense Classifier Stage
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.Dropout(0.5, name='dropout_1'),
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.Dropout(0.5, name='dropout_2'),

        # Output Stage
        layers.Dense(num_classes, activation='softmax', name='output_softmax')
    ], name="FaceMaskCNN")

    # Compile Model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def print_model_summary(model):
    """
    Logs the CNN model architecture summary and total trainable parameters.
    """
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    summary_str = "\n".join(summary_lines)

    total_params = model.count_params()
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    logger.info("==================================================")
    logger.info("              CNN MODEL ARCHITECTURE              ")
    logger.info("==================================================")
    logger.info("\n" + summary_str)
    logger.info(f"Total Parameters:          {total_params:,}")
    logger.info(f"Trainable Parameters:      {trainable_params:,}")
    logger.info(f"Non-Trainable Parameters:  {non_trainable_params:,}")
    logger.info("==================================================")

    return trainable_params

# ==============================================================================
# TRAINING CURVES & HISTORY SAVING
# ==============================================================================
def plot_training_curves(history, save_dir=config.GRAPHS_DIR):
    """
    Generates and saves high-resolution Accuracy and Loss graphs.
    """
    history_dict = history.history if hasattr(history, 'history') else history

    acc = history_dict.get('accuracy', [])
    val_acc = history_dict.get('val_accuracy', [])
    loss = history_dict.get('loss', [])
    val_loss = history_dict.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    # 1. Accuracy Graph
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, acc, 'o-', color='#2980b9', linewidth=2.5, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 's--', color='#27ae60', linewidth=2.5, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    logger.info(f"Accuracy curve saved to: {acc_path}")

    # 2. Loss Graph
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, loss, 'o-', color='#c0392b', linewidth=2.5, label='Training Loss')
    plt.plot(epochs_range, val_loss, 's--', color='#e67e22', linewidth=2.5, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to: {loss_path}")

def save_training_history(history, elapsed_time):
    """Saves training metrics and elapsed duration to JSON and text format."""
    history_dict = history.history if hasattr(history, 'history') else history
    # Convert float32/float64 to standard float for JSON serialization
    serialized_history = {k: [float(val) for val in v] for k, v in history_dict.items()}
    serialized_history["training_time_seconds"] = elapsed_time

    json_path = os.path.join(config.REPORTS_DIR, "training_history.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serialized_history, f, indent=4)

    logger.info(f"Training history saved to: {json_path}")

# ==============================================================================
# EVALUATION METRICS & PLOTTING
# ==============================================================================
def plot_confusion_matrix(cm, class_names=config.CLASS_NAMES):
    """
    Generates and saves the Confusion Matrix heatmap in multiple required folders.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"})

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save inside output/graphs/ and output/confusion_matrix/
    path1 = os.path.join(config.GRAPHS_DIR, "confusion_matrix.png")
    path2 = os.path.join(config.CONF_MATRIX_DIR, "confusion_matrix.png")

    plt.savefig(path1, dpi=300)
    plt.savefig(path2, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix heatmaps saved to:\n  - {path1}\n  - {path2}")

def plot_roc_curve(y_true, y_pred_probs, class_names=config.CLASS_NAMES):
    """
    Computes and plots the Receiver Operating Characteristic (ROC) curve and AUC score.
    """
    # Assuming index 0 is WithMask, index 1 is WithoutMask
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#8e44ad', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    roc_path = os.path.join(config.GRAPHS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info(f"ROC Curve saved to: {roc_path} (AUC Score: {roc_auc:.4f})")
    return roc_auc

def save_evaluation_reports(test_loss, test_acc, precision, recall, f1, cm, report_str, roc_auc):
    """
    Formats and writes comprehensive evaluation summary into reports folder.
    """
    results_path = os.path.join(config.REPORTS_DIR, "evaluation_results.txt")
    clf_path = os.path.join(config.REPORTS_DIR, "classification_report.txt")

    # Classification report file
    with open(clf_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("            CLASSIFICATION REPORT                 \n")
        f.write("==================================================\n\n")
        f.write(report_str)

    # Detailed Evaluation results file
    summary_text = (
        "==================================================\n"
        "         MODEL EVALUATION METRICS SUMMARY         \n"
        "==================================================\n"
        f"Test Loss:            {test_loss:.4f}\n"
        f"Test Accuracy:        {test_acc * 100:.2f}%\n"
        f"Precision (Macro):    {precision:.4f}\n"
        f"Recall (Macro):       {recall:.4f}\n"
        f"F1 Score (Macro):     {f1:.4f}\n"
        f"ROC AUC Score:        {roc_auc:.4f}\n"
        "--------------------------------------------------\n"
        "CONFUSION MATRIX:\n"
        f"{cm}\n"
        "==================================================\n\n"
        "DETAILED CLASSIFICATION REPORT:\n"
        f"{report_str}\n"
    )

    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    logger.info(f"Evaluation summary text written to:\n  - {results_path}\n  - {clf_path}")

# ==============================================================================
# SAMPLE PREDICTION GENERATOR
# ==============================================================================
def generate_sample_predictions(model, test_generator, num_samples=10):
    """
    Generates annotated sample prediction images with bounding boxes, predictions,
    and confidence scores saved in output/predictions/ folder.
    """
    logger.info(f"Generating {num_samples} sample prediction images...")
    test_generator.reset()

    images_processed = 0
    class_indices = {v: k for k, v in test_generator.class_indices.items()}

    for i in range(len(test_generator)):
        batch_x, batch_y = test_generator[i]
        predictions = model.predict(batch_x, verbose=0)

        for j in range(len(batch_x)):
            if images_processed >= num_samples:
                break

            img_array = (batch_x[j] * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            true_idx = np.argmax(batch_y[j])
            pred_idx = np.argmax(predictions[j])
            confidence = predictions[j][pred_idx] * 100

            pred_label = class_indices.get(pred_idx, "Unknown")
            true_label = class_indices.get(true_idx, "Unknown")

            # Determine color: Green for WithMask, Red for WithoutMask
            color = (0, 255, 0) if pred_label == "WithMask" else (0, 0, 255)

            # Draw a decorative bounding box around image center to represent face crop
            h, w, _ = img_bgr.shape
            cv2.rectangle(img_bgr, (5, 5), (w - 5, h - 5), color, 3)

            # Text overlay with background box for high visibility
            label_str = f"{pred_label} ({confidence:.2f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            cv2.rectangle(img_bgr, (5, 5), (5 + text_w + 10, 5 + text_h + 10), (0, 0, 0), cv2.FILLED)
            cv2.putText(img_bgr, label_str, (10, 15 + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            save_file = os.path.join(config.PREDICTIONS_DIR, f"sample_prediction_{images_processed + 1}.png")
            cv2.imwrite(save_file, img_bgr)
            images_processed += 1

        if images_processed >= num_samples:
            break

    logger.info(f"Sample prediction images saved in: {config.PREDICTIONS_DIR}")
