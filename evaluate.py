"""
Evaluation script for Face Mask Detection Deep Learning Project.
Loads the trained model, evaluates performance on the un-augmented Test dataset,
computes metrics (Loss, Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC),
generates graphs, and outputs sample prediction images.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import config
import utils

def main():
    logger = utils.get_logger()
    logger.info("==================================================")
    logger.info("     FACE MASK DETECTION - EVALUATION PIPELINE   ")
    logger.info("==================================================")

    # 1. Ensure Directories Exist
    utils.create_directories()

    # 2. Check if Best Model File Exists
    model_path = config.BEST_MODEL_PATH if os.path.exists(config.BEST_MODEL_PATH) else config.FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        logger.error(f"Trained model file not found at {model_path}! Please run 'python train.py' first.")
        sys.exit(1)

    logger.info(f"Loading trained CNN model from: {model_path}")
    model = load_model(model_path)

    # 3. Prepare Test ImageDataGenerator (Rescaling ONLY, no data augmentation)
    logger.info(f"Preparing Test dataset from: {config.TEST_DIR}")
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    if test_generator.samples == 0:
        logger.error(f"No test images found in {config.TEST_DIR}!")
        sys.exit(1)

    logger.info(f"Total Test Images Loaded: {test_generator.samples}")
    logger.info(f"Class Mapping: {test_generator.class_indices}")

    # 4. Evaluate Model on Test Generator
    logger.info("Running model evaluation on Test set...")
    eval_results = model.evaluate(test_generator, verbose=1)
    test_loss, test_acc = eval_results[0], eval_results[1]

    # 5. Predict on Test Set to get Detailed Classification Metrics
    logger.info("Generating predictions for confusion matrix & ROC curve...")
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Compute Metrics
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    report_str = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    )

    # 6. Plot and Save Visual Evaluation Graphs
    logger.info("Generating evaluation graphs and charts...")
    utils.plot_confusion_matrix(cm, class_names=config.CLASS_NAMES)
    roc_auc = utils.plot_roc_curve(y_true, y_pred_probs, class_names=config.CLASS_NAMES)

    # 7. Write Evaluation Results & Classification Reports
    utils.save_evaluation_reports(
        test_loss=test_loss,
        test_acc=test_acc,
        precision=precision,
        recall=recall,
        f1=f1,
        cm=cm,
        report_str=report_str,
        roc_auc=roc_auc
    )

    # 8. Generate Sample Prediction Images
    utils.generate_sample_predictions(model, test_generator, num_samples=10)

    # 9. Console Output Summary
    logger.info("==================================================")
    logger.info("           FINAL TEST EVALUATION RESULTS          ")
    logger.info("==================================================")
    logger.info(f"  - Test Accuracy:        {test_acc * 100:.2f}%")
    logger.info(f"  - Test Loss:            {test_loss:.4f}")
    logger.info(f"  - Precision (Macro):    {precision:.4f}")
    logger.info(f"  - Recall (Macro):       {recall:.4f}")
    logger.info(f"  - F1 Score (Macro):     {f1:.4f}")
    logger.info(f"  - ROC AUC Score:        {roc_auc:.4f}")
    logger.info("--------------------------------------------------")
    logger.info("CONFUSION MATRIX:\n" + str(cm))
    logger.info("--------------------------------------------------")
    logger.info("CLASSIFICATION REPORT:\n" + report_str)
    logger.info("==================================================")

if __name__ == "__main__":
    main()
