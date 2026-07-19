# Bearing Fault Diagnosis using Machine Learning (HUST Dataset)

A Streamlit-based web application for intelligent bearing fault diagnosis using machine learning models trained on the HUST Bearing Dataset.

The application accepts a vibration signal stored in a `.mat` file, extracts statistical and frequency-domain features, and predicts the bearing fault category using multiple trained machine learning models.

---

## Features

- Upload bearing vibration signals in `.mat` format
- Automatic signal segmentation
- Time-domain and frequency-domain feature extraction
- Multiple prediction models:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Neural Network
- Confidence score for each model
- Majority voting based final diagnosis
- Interactive signal visualization
- FFT spectrum visualization
- Export prediction results as CSV

---

## Project Structure

```
bearing-fault-diagnosis-HUST/
│
├── app.py                     # Streamlit application
├── models/                    # Trained ML models and scalers
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── tf_nn_model.keras
│   ├── svm_scaler.pkl
│   ├── knn_scaler.pkl
│   └── tf_scaler.pkl
│
├── notebooks/                 # Training notebook
├── features/                  # Extracted Features(.csv)
│
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10 or later
- pip

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will automatically open in your default browser.

If it does not open automatically, visit:

```
http://localhost:8501
```

---

## Usage

1. Launch the application.
2. Upload a compatible `.mat` vibration signal file.
3. Select one or more prediction models.
4. View:
   - Time-domain signal
   - FFT spectrum
   - Individual model predictions
   - Final diagnosis
   - Confidence score
5. Export the prediction results as a CSV file if required.

---

## Fault Classes

The trained models classify the vibration signal into one of the following bearing conditions:

| Label | Condition |
|--------|-----------|
| N | Normal |
| B | Ball Fault |
| I | Inner Race Fault |
| O | Outer Race Fault |
| IB | Inner + Ball Fault |
| IO | Inner + Outer Fault |
| OB | Outer + Ball Fault |

---

## Dataset

This repository **does not include** the HUST Bearing Dataset due to its size.

The application is designed to work with vibration signals stored in compatible `.mat` files.

---

## Models Used

- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- TensorFlow Neural Network

Predictions from multiple models are combined using majority voting to generate the final diagnosis.

---

## Technologies

- Python
- Streamlit
- NumPy
- SciPy
- Scikit-learn
- TensorFlow / Keras
- Plotly
- Pandas

---

