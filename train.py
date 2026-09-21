"""
Training script for Face Mask Detection Deep Learning Project.
Loads dataset with ImageDataGenerator, applies training data augmentation,
builds the CNN model, trains with callbacks, logs performance metrics, and saves trained models.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

import config
import utils

def main():
    logger = utils.get_logger()
    logger.info("==================================================")
    logger.info("       FACE MASK DETECTION - TRAINING PIPELINE    ")
    logger.info("==================================================")

    # 1. Set Random Seeds for Reproducibility
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

    # 2. Ensure Directories Exist & Ensure Haar Cascade is present
    utils.create_directories()
    utils.ensure_haar_cascade()

    # 3. Analyze and Display Dataset Statistics
    logger.info("Analyzing dataset directory structure and scanning image counts...")
    stats = utils.get_dataset_stats()

    train_total = stats["splits"]["Train"]["total"]
    val_total = stats["splits"]["Validation"]["total"]
    test_total = stats["splits"]["Test"]["total"]
    mask_total = stats["class_counts"]["WithMask"]
    no_mask_total = stats["class_counts"]["WithoutMask"]

    logger.info("--------------------------------------------------")
    logger.info("             DATASET SUMMARY STATISTICS           ")
    logger.info("--------------------------------------------------")
    logger.info(f"  - Training Images:    {train_total}")
    logger.info(f"  - Validation Images:  {val_total}")
    logger.info(f"  - Testing Images:     {test_total}")
    logger.info(f"  - Total Mask Images:  {mask_total}")
    logger.info(f"  - Total No Mask Images:{no_mask_total}")
    logger.info(f"  - Total Dataset Size: {stats['total_images']}")
    logger.info("--------------------------------------------------")

    # Generate and Save Dataset Distribution Graph
    utils.plot_dataset_distribution(stats)

    # 4. Prepare ImageDataGenerators
    # Augmentations ONLY for training data
    logger.info("Initializing ImageDataGenerator data loaders...")
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=config.AUGMENTATION_CONFIG["rotation_range"],
        zoom_range=config.AUGMENTATION_CONFIG["zoom_range"],
        horizontal_flip=config.AUGMENTATION_CONFIG["horizontal_flip"],
        width_shift_range=config.AUGMENTATION_CONFIG["width_shift_range"],
        height_shift_range=config.AUGMENTATION_CONFIG["height_shift_range"],
        shear_range=config.AUGMENTATION_CONFIG["shear_range"],
        fill_mode=config.AUGMENTATION_CONFIG["fill_mode"]
    )

    # Rescaling ONLY for validation and test data (NO augmentation)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Flow from directory
    logger.info(f"Loading Training data from: {config.TRAIN_DIR}")
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.SEED
    )

    logger.info(f"Loading Validation data from: {config.VAL_DIR}")
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Log Detected Class Mapping
    logger.info(f"Detected Class Mapping: {train_generator.class_indices}")

    # 5. Build and Display CNN Model Architecture
    logger.info("Constructing custom CNN model...")
    model = utils.build_cnn_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE
    )

    utils.print_model_summary(model)

    # 6. Configure Training Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 7. Model Training Execution
    logger.info(f"Starting CNN Model Training for {config.EPOCHS} epochs...")
    start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    logger.info("==================================================")
    logger.info(f" Training Completed in {int(minutes)}m {seconds:.2f}s ")
    logger.info("==================================================")

    # 8. Save Final Model
    model.save(config.FINAL_MODEL_PATH)
    logger.info(f"Saved best model to:  {config.BEST_MODEL_PATH}")
    logger.info(f"Saved final model to: {config.FINAL_MODEL_PATH}")

    # 9. Plot and Save Training Curves & History
    utils.plot_training_curves(history)
    utils.save_training_history(history, elapsed_time)

    # Final epoch metrics logging
    final_epoch = len(history.history['accuracy'])
    logger.info("--------------------------------------------------")
    logger.info("                FINAL TRAINING METRICS            ")
    logger.info("--------------------------------------------------")
    logger.info(f"  - Total Epochs Trained: {final_epoch}")
    logger.info(f"  - Training Accuracy:    {history.history['accuracy'][-1] * 100:.2f}%")
    logger.info(f"  - Validation Accuracy:  {history.history['val_accuracy'][-1] * 100:.2f}%")
    logger.info(f"  - Training Loss:        {history.history['loss'][-1]:.4f}")
    logger.info(f"  - Validation Loss:      {history.history['val_loss'][-1]:.4f}")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
