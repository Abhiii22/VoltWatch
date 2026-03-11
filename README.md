# ⚡ VoltWatch — Smart Grid Energy Anomaly Detection System

> A hybrid ML system combining Isolation Forest and LSTM Autoencoder to detect energy theft and equipment faults in real-time from smart meter time-series data, deployed with an automated AWS S3 ETL pipeline and Tableau analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?style=flat-square&logo=tensorflow)
![AWS](https://img.shields.io/badge/AWS-S3-yellow?style=flat-square&logo=amazon-aws)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-blue?style=flat-square&logo=tableau)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-IsolationForest-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Results & Performance](#-results--performance)
- [Dashboard Preview](#-dashboard-preview)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📖 Overview

**VoltWatch** is an end-to-end anomaly detection system built for smart electricity grids. It ingests raw smart meter readings from **AWS S3**, processes 2M+ time-series data points through an automated ETL pipeline, and applies a **hybrid detection model** — combining unsupervised **Isolation Forest** for broad anomaly flagging with a deep learning **LSTM Autoencoder** for precise temporal pattern analysis.

The system detects two critical failure modes: **energy theft** (abnormal consumption drops suggesting meter tampering) and **equipment faults** (irregular voltage/current spikes indicating hardware failure). Results are visualized in an interactive **Tableau dashboard** used for predictive maintenance planning and stakeholder reporting.

---

## 🎯 Problem Statement

Energy theft and undetected equipment faults cost utilities billions annually. Key challenges:

- **Energy theft** accounts for 10–40% of electricity distribution losses in developing countries
- **Manual inspection** of millions of meters is operationally infeasible
- **Rule-based threshold systems** produce too many false positives, causing alert fatigue
- **Equipment faults** detected late cause cascading failures and outages
- **No historical pattern learning** — existing systems don't adapt to seasonal or behavioral drift

VoltWatch addresses all of these with an adaptive, data-driven anomaly detection pipeline.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔄 **Automated ETL Pipeline** | Ingests raw sensor CSV files from AWS S3, cleans, transforms, and loads into analytical data store |
| ⚙️ **Hybrid Detection Model** | Isolation Forest (unsupervised) + LSTM Autoencoder (deep learning) ensemble for 91% precision |
| 📐 **Time-Series Feature Engineering** | Rolling statistics, lag features, Fourier transforms, hour-of-day and day-of-week encodings |
| 🚨 **Configurable Alert Thresholds** | Per-meter and per-region threshold tuning to minimize false positive rates |
| 📊 **Tableau Dashboard** | Geographic fault maps, anomaly cluster visualization, consumption forecasts, and KPI tracking |
| 📋 **Model Governance** | Confusion matrices, precision/recall tracking, data drift monitoring, and experiment logging |
| ☁️ **Cloud-Ready** | Fully designed for AWS S3 data ingestion; modular for Lambda or EC2 batch execution |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VoltWatch Pipeline                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ☁️  AWS S3 (Raw Sensor Data)                                        │
│       │   CSV files: timestamp, meter_id, kwh, voltage, current      │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     ETL Pipeline (Python)                     │   │
│  │  1. Ingestion   → Download from S3, validate schema           │   │
│  │  2. Cleaning    → Handle nulls, outlier clipping, resampling  │   │
│  │  3. Feature Eng → Rolling stats, lags, Fourier, time encodings│   │
│  │  4. Storage     → Save processed data to analytical store     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼                                                              │
│  ┌────────────────────┐      ┌───────────────────────────────────┐  │
│  │  Isolation Forest  │      │     LSTM Autoencoder              │  │
│  │  (Unsupervised)    │      │     (Deep Learning)               │  │
│  │  - Broad anomaly   │      │  - Learns normal consumption      │  │
│  │    flagging        │      │    sequences over time            │  │
│  │  - Fast inference  │      │  - High reconstruction error      │  │
│  │  - No labels needed│      │    = anomaly detected             │  │
│  └────────────────────┘      └───────────────────────────────────┘  │
│             │                              │                         │
│             └──────────┬───────────────────┘                        │
│                        ▼                                             │
│              ┌─────────────────────┐                                │
│              │  Ensemble Combiner  │                                │
│              │  Weighted vote:     │                                │
│              │  IF score + LSTM    │                                │
│              │  reconstruction err │                                │
│              └─────────────────────┘                                │
│                        │                                            │
│            ┌───────────┴────────────┐                               │
│            ▼                        ▼                               │
│   ┌──────────────────┐   ┌────────────────────────────┐            │
│   │  Alert Engine    │   │   Tableau Dashboard         │            │
│   │  Configurable    │   │   - Anomaly cluster maps    │            │
│   │  thresholds +    │   │   - Fault geo distribution  │            │
│   │  severity levels │   │   - Consumption forecasts   │            │
│   └──────────────────┘   └────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow 2.x / Keras (LSTM Autoencoder) |
| **ML** | Scikit-learn (Isolation Forest, preprocessing, evaluation) |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Feature Engineering** | statsmodels (Fourier), tsfresh (optional) |
| **Cloud Storage** | AWS S3 (boto3) |
| **Visualization** | Tableau, Matplotlib, Seaborn |
| **Experiment Tracking** | MLflow (optional) |
| **Utilities** | tqdm, python-dotenv, PyYAML, joblib |

---

## 📁 Project Structure

```
VoltWatch/
│
├── data/
│   ├── raw/                         # Raw meter readings (downloaded from S3)
│   ├── processed/                   # Cleaned and feature-engineered data
│   ├── anomalies/                   # Flagged anomaly records
│   └── sample/                      # Sample dataset for demo (1000 meters, 30 days)
│
├── models/
│   ├── isolation_forest/
│   │   └── if_model.pkl             # Saved Isolation Forest model
│   ├── lstm_autoencoder/
│   │   ├── model.h5                 # Saved LSTM Autoencoder weights
│   │   └── threshold.json           # Per-meter reconstruction thresholds
│   └── ensemble/
│       └── combiner_config.yaml     # Ensemble weight configuration
│
├── src/
│   ├── etl/
│   │   ├── s3_ingestor.py           # Download and validate data from AWS S3
│   │   ├── cleaner.py               # Null handling, resampling, outlier clipping
│   │   └── feature_engineer.py      # Rolling stats, lags, Fourier, time encodings
│   ├── models/
│   │   ├── isolation_forest.py      # IF training and scoring
│   │   ├── lstm_autoencoder.py      # LSTM model definition, training, inference
│   │   └── ensemble.py              # Weighted ensemble combiner
│   ├── alerts/
│   │   └── alert_engine.py          # Threshold logic, severity classification
│   ├── evaluation/
│   │   ├── metrics.py               # Precision, recall, F1, confusion matrix
│   │   └── drift_monitor.py         # Data drift detection over time
│   └── utils/
│       ├── config.py                # Load YAML config
│       └── logger.py                # Logging setup
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis on meter data
│   ├── 02_feature_engineering.ipynb # Feature exploration and selection
│   ├── 03_isolation_forest.ipynb    # IF training and threshold tuning
│   ├── 04_lstm_autoencoder.ipynb    # LSTM architecture, training curves
│   └── 05_ensemble_evaluation.ipynb # Final model evaluation and comparison
│
├── tableau/
│   └── VoltWatch_Dashboard.twbx     # Tableau workbook file
│
├── tests/
│   ├── test_etl.py
│   ├── test_feature_engineer.py
│   └── test_models.py
│
├── config.yaml                      # Global configuration
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- AWS account with S3 bucket access (or use sample data)
- Tableau Desktop (for dashboard, optional)
- 6GB+ RAM recommended for LSTM training

### 1. Clone the repository
```bash
git clone https://github.com/Abhiii22/VoltWatch.git
cd VoltWatch
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env:
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# AWS_REGION=ap-south-1
# S3_BUCKET_NAME=your_bucket
```

### 5. Update config
```bash
# Edit config.yaml to set:
# - meter_count, date_range
# - anomaly thresholds
# - model hyperparameters
```

---

## 🚀 Usage

### Step 1 — Run the full ETL pipeline
```bash
python src/etl/s3_ingestor.py --bucket your-bucket --prefix meter_data/2025/
# Downloads, cleans, and saves to data/processed/
```

### Step 2 — Feature engineering
```bash
python src/etl/feature_engineer.py --input data/processed/ --output data/features/
# Adds rolling mean/std, lag-1/lag-7, Fourier components, hour/day encodings
```

### Step 3 — Train Isolation Forest
```bash
python src/models/isolation_forest.py --train --data data/features/train.csv
# Saves model to models/isolation_forest/if_model.pkl
```

### Step 4 — Train LSTM Autoencoder
```bash
python src/models/lstm_autoencoder.py --train --data data/features/train.csv --epochs 50
# Saves model to models/lstm_autoencoder/model.h5
```

### Step 5 — Run ensemble inference on new data
```bash
python src/models/ensemble.py --input data/features/new_batch.csv --output data/anomalies/
# Outputs flagged anomalies with severity scores
```

### Step 6 — Run the alert engine
```bash
python src/alerts/alert_engine.py --anomalies data/anomalies/results.csv
# Prints/logs alerts classified as Low / Medium / High / Critical
```

### Quick demo with sample data
```bash
# Use bundled sample dataset (1,000 meters, 30 days)
python src/models/ensemble.py --input data/sample/sample_features.csv --output data/anomalies/
```

---

## 🤖 Model Details

### Isolation Forest (Unsupervised Baseline)
- **Purpose:** Fast, broad anomaly flagging across all meters
- **Contamination rate:** 0.05 (tuned on validation set)
- **Features used:** Rolling mean, rolling std, lag-1, lag-7, consumption delta
- **Strength:** No labels needed; handles high-dimensional tabular data well

### LSTM Autoencoder (Deep Learning)
- **Architecture:**
  ```
  Encoder: LSTM(128) → LSTM(64) → RepeatVector
  Decoder: LSTM(64) → LSTM(128) → TimeDistributed Dense
  ```
- **Input:** Sliding window of 24 hourly readings per meter
- **Training objective:** Minimize reconstruction error (MSE) on normal sequences
- **Anomaly criterion:** Reconstruction error > learned per-meter threshold (95th percentile on training data)
- **Epochs:** 50 | **Batch size:** 32 | **Optimizer:** Adam (lr=0.001)

### Ensemble Strategy
- **Soft voting:** Normalized IF anomaly score + LSTM reconstruction error (normalized)
- **Weights:** IF = 0.35, LSTM = 0.65 (tuned via grid search on labeled validation subset)
- **Final threshold:** Ensemble score > 0.60 → flagged as anomaly

### Feature Engineering Summary
| Feature | Description |
|---|---|
| `rolling_mean_24h` | 24-hour rolling average consumption |
| `rolling_std_24h` | 24-hour rolling standard deviation |
| `lag_1` | Consumption at t-1 |
| `lag_7d` | Same hour, 7 days prior (weekly seasonality) |
| `fourier_sin_24`, `fourier_cos_24` | Daily Fourier component |
| `hour_of_day`, `day_of_week` | Calendar encodings |
| `delta` | Consumption change from previous reading |

---

## 📊 Results & Performance

| Metric | Isolation Forest | LSTM Autoencoder | **Ensemble** |
|---|---|---|---|
| Precision | 0.78 | 0.87 | **0.91** |
| Recall | 0.82 | 0.85 | **0.88** |
| F1-Score | 0.80 | 0.86 | **0.89** |
| False Positive Rate | 12.4% | 7.1% | **5.3%** |

**Additional metrics:**
- ETL pipeline preprocessing time reduced by **60%** via vectorized Pandas operations and parallel S3 downloads
- Processed **2M+ records** across 10,000 simulated meters
- Batch inference time for 10,000 meters: **~4.2 minutes**

---

## 📸 Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚡ VoltWatch — Anomaly Detection Dashboard          [Date: Jan 2026]│
├──────────────────────┬──────────────────────────────────────────────┤
│  KPIs                │  Geographic Fault Distribution               │
│                      │                                              │
│  Total Meters: 10,000│      ● ●    ← High risk cluster             │
│  Anomalies: 523      │   ●         (Zone 4 - North)                │
│  Critical:  41       │        ●  ●                                 │
│  Precision: 91%      │  ● ●                                        │
│                      │                                              │
├──────────────────────┼──────────────────────────────────────────────┤
│  Anomaly Trend       │  Consumption Forecast vs Actual              │
│  (Last 30 days)      │                                              │
│   ╭─╮                │  ████████████░░░░░░░  Actual                │
│  ╭╯ ╰╮   ╭╮          │  ────────────────────  Forecast             │
│ ─╯    ╰──╯╰──        │  Deviation detected at 03:00–05:00 AM ⚠️   │
└──────────────────────┴──────────────────────────────────────────────┘
```

---

## 🔮 Future Improvements

- [ ] Real-time streaming pipeline using Apache Kafka + AWS Kinesis
- [ ] Transformer-based anomaly detection (PatchTST / Informer) for comparison
- [ ] Federated learning to train across utility companies without sharing raw data
- [ ] Automated retraining pipeline triggered on data drift detection
- [ ] REST API endpoint for real-time single-meter scoring
- [ ] Explainability module (SHAP values) to explain why a reading was flagged

---

## 👤 Author

**Abhyuday Singh**
- 📧 rajputabhyuday23258958@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/rajputabhyuday/)
- 🐙 [GitHub](https://github.com/Abhiii22/)

---

> ⭐ If you found this project interesting, please consider starring the repo!
