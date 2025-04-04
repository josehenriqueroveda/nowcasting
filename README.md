# Nowcasting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that predicts rain within the next hour using weather station data.

## 📌 Project Overview

This project trains and compares LightGBM and XGBoost models to forecast precipitation within the next 60 minutes using minute-by-minute (or hourly) meteorological data including:

- Temperature measurements
- Humidity levels
- Pressure readings
- Wind speed/direction
- Radiation data
- Precipitation history

## 🛠️ Technical Stack

- **Core ML**: LightGBM, XGBoost
- **Data Processing**: Pandas, NumPy
- **Feature Engineering**: Scikit-learn
- **Model Evaluation**: ROC-AUC, Precision-Recall
- **Production Ready**: Joblib serialization

## 📂 Project Structure

```
nowcasting/
├── data/
│ ├── raw/ # Original station data
│ └── processed/ # Processed feature data
├── models/
│ ├── trained/ # Serialized models
│ └── evaluation/ # Performance metrics
├── src/
│ ├── data/ # Data processing
│ ├── features/ # Feature engineering
│ ├── models/ # Training/prediction
├── config/ # Parameters
└── README.md
```

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
python src/data/preprocess.py
```

### 3. Feature Engineering

```bash
python src/features/build_features.py
```

### 4. Model Training

```bash
python src/models/train_model.py
```

### 5. Model Prediction

```bash
python src/models/predict_model.py
```

## 🧪 Model Performance

| Model    | ROC-AUC | Precision | Recall |
| -------- | ------- | --------- | ------ |
| LightGBM | to do   | to do     | to do  |
| XGBoost  | to do   | to do     | to do  |

## 🔑 Key Features

- **Temporal Feature Engineering**: Rolling averages, pressure variations
- **Wind Vectorization**: Polar to Cartesian conversion
- **Production Pipeline**: Complete train/predict workflow
