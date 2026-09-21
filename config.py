"""
Configuration module for Face Mask Detection Deep Learning Project.
Contains dataset paths, model parameters, hyperparameter settings,
and output directory definitions.
"""

import os

# ==============================================================================
# RANDOM SEED
# ==============================================================================
SEED = 42

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
# Base dataset directory
DATASET_DIR = "Face Mask Dataset"

# Dataset splits
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
VAL_DIR = os.path.join(DATASET_DIR, "Validation")
TEST_DIR = os.path.join(DATASET_DIR, "Test")

# Folder name mapping (Supports automatic detection or custom mapping)
# Priority order for class subfolders: ('WithMask', 'WithoutMask') or ('with_mask', 'without_mask')
CLASS_SUBFOLDERS = {
    "MASK": ["WithMask", "with_mask", "mask", "With_Mask"],
    "NO_MASK": ["WithoutMask", "without_mask", "no_mask", "Without_Mask"]
}

# Standardized Class Names
CLASS_NAMES = ["WithMask", "WithoutMask"]
NUM_CLASSES = 2

# ==============================================================================
# IMAGE & MODEL HYPERPARAMETERS
# ==============================================================================
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Data Augmentation Parameters (Applied to Training set only)
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "zoom_range": 0.15,
    "horizontal_flip": True,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.15,
    "fill_mode": "nearest"
}

# ==============================================================================
# DIRECTORY STRUCTURE & MODEL SAVE PATHS
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.keras")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
CONF_MATRIX_DIR = os.path.join(OUTPUT_DIR, "confusion_matrix")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "face_mask_detection.log")

HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
