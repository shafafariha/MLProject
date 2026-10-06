# 🖐️ BISINDO Hand Sign Recognition & Classification

A Machine Learning project for real-time and static image recognition of the **Indonesian Sign Language (BISINDO - Bahasa Isyarat Indonesia)** alphabet (A–Z) using **MediaPipe Hand Landmarks** and a **Random Forest Classifier**. Features an interactive web-based interface built with **Streamlit**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Workflow & Pipeline](#-workflow--pipeline)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Directory Structure](#-directory-structure)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Dashboard Usage](#-dashboard-usage)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 📖 Project Overview

Indonesian Sign Language (BISINDO) is the natural sign language used by the Deaf community across Indonesia. This project aims to bridge communication gaps by building an automated, accurate sign language alphabet translator using *Computer Vision* and *Machine Learning*:

1. **Geometric Feature Extraction**: MediaPipe Hands extracts 3D coordinates ($x, y, z$) from hand joints.
2. **Data Augmentation**: Enhances image diversity using rotation, horizontal flipping, and contrast/lighting variations.
3. **Machine Learning Modeling**: Compares 7 classification algorithms, with **Random Forest** achieving the highest performance at **~99.5% test accuracy**.
4. **Interactive Dashboard**: A user-friendly Streamlit web app providing real-time webcam inference and static image analysis.

---

## ✨ Key Features

- 🔴 **Real-Time Webcam Detection**: Live sign detection with real-time skeleton overlay and predicted sign labels.
- 🖼️ **Image Upload Mode**: Test single images in `.jpg`, `.jpeg`, or `.png` formats.
- 🎛️ **Camera & Image Controls**: Interactive sidebar sliders for **Brightness**, **Contrast**, and **Saturation** adjustments to handle various lighting conditions.
- ⚙️ **Detection Sensitivity Settings**: Configurable *Min Detection Confidence* and *Min Tracking Confidence* thresholds.
- 👐 **Dual-Hand Support**: Extracts landmarks for up to 2 hands simultaneously (up to 126 feature dimensions), accommodating both single-handed and two-handed gestures.

---

## 🔄 Workflow & Pipeline

```
[ Input: Live Webcam / Image ]
             │
             ▼
[ MediaPipe Hand Landmark Extraction ] ──► (21 hand joints × 3 coordinates [x, y, z] per hand)
             │
             ▼
[ Feature Vector Construction (126-D) ] ──► (Zero-padding applied if only 1 hand is detected)
             │
             ▼
[ Random Forest Classifier ]           ──► (model/rf_bisindo_classifier_99.pkl)
             │
             ▼
[ Output: Predicted Letter (A–Z) & Landmark Visualization ]
```

---

## 📊 Model Performance & Evaluation

Extensive benchmarking was performed across 7 classification algorithms in [`ipynb/BISINDO 1.ipynb`](ipynb/BISINDO%201.ipynb):

| Algorithm | Validation Accuracy | Validation F1-Score | Test Accuracy | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** 🏆 | **99.54%** | **99.54%** | **99.45% - 99.59%** | **99.45% - 99.59%** |
| **Gradient Boosting** | 98.89% | 98.90% | 98.71% | 98.71% |
| **K-Nearest Neighbors (KNN)** | 98.30% | 98.30% | 97.79% | 97.77% |
| **Decision Tree** | 96.08% | 96.10% | 95.72% | 95.70% |
| **Support Vector Machine (SVM)** | 86.37% | 85.70% | 86.51% | 85.77% |
| **Logistic Regression** | 83.74% | 83.42% | 83.61% | 83.37% |
| **Naive Bayes** | 41.32% | 31.50% | 41.71% | 31.42% |

> The production model deployed in the dashboard is the **Random Forest Classifier** (`model/rf_bisindo_classifier_99.pkl`).

---

## 📁 Directory Structure

```plaintext
MLProject/
├── .devcontainer/             # Dev container configuration
│   └── devcontainer.json
├── .streamlit/                # Streamlit theme and server configuration
│   └── config.toml
├── dashboard/                 # Streamlit web application source code
│   ├── dashboard.py           # Main application (Webcam + Image Upload)
│   ├── local.py               # Lightweight local webcam app
│   └── static.py              # Static image classifier app
├── Datasets/                  # Raw image datasets and extracted feature CSVs
│   ├── bisindo/               # Original image dataset organized by letter (A–Z)
│   ├── bisindo-augmented/     # Augmented image dataset
│   ├── bisindo-features.csv   # Extracted 126-D MediaPipe landmark dataset
│   ├── bisindo-train.csv      # Training split
│   ├── bisindo-val.csv        # Validation split
│   └── bisindo-test.csv       # Test split
├── ipynb/                     # Jupyter Notebooks for exploration and training
│   ├── BISINDO 1.ipynb        # Multi-model benchmarking & evaluation
│   └── BISINDO 2.ipynb        # Random Forest detailed analysis & confusion matrix
├── model/                     # Trained model artifacts (.pkl)
│   └── rf_bisindo_classifier_99.pkl
├── packages.txt               # Linux system-level package dependencies
├── requirements.txt           # Python library dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Tech Stack

- **Language**: [Python 3.10+](https://www.python.org/)
- **Computer Vision & Hand Tracking**: [MediaPipe](https://developers.google.com/mediapipe), [OpenCV](https://opencv.org/)
- **Machine Learning & Modeling**: [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Data Manipulation & Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Web UI & Dashboard**: [Streamlit](https://streamlit.io/)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/shafafariha/MLProject.git
cd MLProject
```

### 2. Create and Activate a Virtual Environment
- **Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Linux Users**: If OpenCV complains about missing graphics libraries (`libGL.so.1`), install the system packages specified in `packages.txt`:
> ```bash
> sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libxrender1 libxext6
> ```

---

## 🖥️ Dashboard Usage

Run the main Streamlit dashboard using:

```bash
streamlit run dashboard/dashboard.py
```

Once running, navigate to `http://localhost:8501` in your web browser.

### Dashboard Modes:
1. **Image Upload Mode**:
   - Drag and drop or upload an image file containing hand sign gestures.
   - The app will extract landmarks, render hand skeleton overlays, and display the predicted BISINDO letter.
2. **Real-Time Webcam Mode**:
   - Toggle the **"Start Webcam"** checkbox.
   - Position your hands in front of the camera with adequate lighting.
   - Adjust the brightness, contrast, or detection confidence sliders in the sidebar for optimal results.

---

## 📄 License & Acknowledgments

- **Dataset**: Kaggle Dataset [achmadnoer/alfabet-bisindo](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)
- **Author**: [@shafafariha](https://github.com/shafafariha)
- Developed for educational, research, and assistive technology development in sign language accessibility.
