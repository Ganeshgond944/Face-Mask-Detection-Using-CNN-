# 🛡️ NEUROSHIELD AI — Face Mask & Safety Intelligence Platform

A production-grade, enterprise-ready **Real-Time Face Mask Detection Web Application & Computer Vision Engine** powered by a custom **4-Stage Convolutional Neural Network (CNN)** and **OpenCV Haar Cascades**.

Achieves **99.09% Test Accuracy**, **0.9992 ROC AUC**, and sub-second inference speeds (**< 30 ms**).

---

## ⚡ Quick Start: How to Run

### Option 1: Direct Execution (Recommended & Fastest)
You can run the web server directly using the project's virtual environment without manual activation:

```bash
# From project directory d:\Deep_Learning
.venv\Scripts\python.exe app.py
```

### Option 2: Standard Virtual Environment Activation

#### Windows PowerShell:
```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Start the web application
python app.py
```

#### Windows Command Prompt (cmd):
```cmd
# 1. Activate virtual environment
.venv\Scripts\activate.bat

# 2. Start the web application
python app.py
```

---

## 🌐 Accessing the Application

Once the server is running, open your web browser and navigate to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📌 Features & Application Routes

1. **Dashboard (`/`)**:
   - Neo-Brutalist architectural grid theme.
   - Empirical benchmark metrics display (**99.09% Test Accuracy**, **0.9992 ROC AUC**).
   - Quick navigation to Upload, Live Stream, and System Specs.

2. **Upload Image Pipeline (`/upload` & `/predict`)**:
   - Drag & Drop or file browse interface accepting **JPG, JPEG, and PNG**.
   - Real-time client-side image preview.
   - Crops facial regions of interest via Haar Cascade and draws colored bounding boxes:
     - 🟩 **Green Box**: Wearing Mask (`WithMask`)
     - 🟥 **Red Box**: No Mask (`WithoutMask`)
   - Detailed classification report page with confidence percentage, face count, latency, date & timestamp.

3. **In-Tab Live Camera Stream (`/live` & `/predict_frame`)**:
   - Client-side HTML5 `getUserMedia` video capture scoped strictly to that tab.
   - **Explicit Browser Permission Prompt** before accessing webcam hardware.
   - Automatically releases webcam hardware when tab is closed or switched.
   - Real-time bounding box overlays and live **FPS Counter**.

4. **About & System Architecture (`/about`)**:
   - Technical breakdown of the 4-stage CNN pipeline, hyperparameters, and dataset evaluation metrics.

---

## 🏗️ Project Directory Structure

```
Deep_Learning / (NEUROSHIELD AI)
│
├── app.py                      # Flask backend web application & prediction API
├── config.py                   # Paths, hyperparameters, and configuration settings
├── utils.py                    # CNN model builder, logger, evaluation graphs & metrics
├── train.py                    # Model training script with data augmentation
├── evaluate.py                 # Test set evaluation & confusion matrix generator
├── detect.py                   # Standalone desktop OpenCV webcam script
├── haarcascade_frontalface_default.xml # OpenCV Face Cascade classifier
├── requirements.txt            # Package dependencies
├── README.md                   # Comprehensive documentation
│
├── model/                      # Trained model directory
│   └── best_model.keras        # Pre-trained CNN model file (Loaded ONCE on server boot)
│
├── models/                     # Checkpoint model backup directory
│   ├── best_model.keras
│   └── final_model.keras
│
├── static/                     # Web assets
│   ├── css/
│   │   └── style.css           # Production Neo-Brutalist Architectural Grid CSS
│   ├── js/
│   │   └── main.js             # Drag-and-drop, preview, getUserMedia JS
│   ├── uploads/                # User uploaded images
│   └── results/                # Prediction images with bounding box overlays
│
└── templates/                  # Jinja2 HTML templates
    ├── index.html              # Dashboard home page
    ├── upload.html             # Image upload page
    ├── result.html             # Inference classification report page
    ├── live.html               # Live camera stream page
    └── about.html              # System specs page
```

---

## 📊 Empirical Benchmarks & Performance Metrics

| Metric | Score | Details |
| :--- | :--- | :--- |
| **Test Accuracy** | **99.09%** | Evaluated on 992 unaugmented test set images |
| **ROC AUC Score** | **0.9992** | Perfect class separation confidence |
| **Test Loss** | **0.0296** | Categorical Cross-Entropy Loss |
| **F1 Score** | **0.9909** | Weighted harmonic mean of Precision and Recall |
| **Inference Time** | **< 30 ms** | Sub-second predictions via pre-loaded memory model |

---

## 🛠️ CLI Script Commands

If you wish to run standalone training, evaluation, or desktop detection scripts:

### 1. Run Real-Time Desktop Camera (OpenCV Window):
```bash
.venv\Scripts\python.exe detect.py
```

### 2. Run Test Set Evaluation & Generate Report Graphs:
```bash
.venv\Scripts\python.exe evaluate.py
```

### 3. Re-train the CNN Model from Scratch:
```bash
.venv\Scripts\python.exe train.py
```

---

## 👨‍💻 Tech Stack & License

- **Backend**: Flask 3.x (Python)
- **Deep Learning Engine**: TensorFlow 2.x & Keras
- **Computer Vision**: OpenCV 4.x
- **Frontend Stack**: HTML5, CSS3, Bootstrap 5, JavaScript (`Space Grotesk` Typography)
- **License**: Apache License 2.0
