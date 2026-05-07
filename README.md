# Home Credit Default Risk Prediction - MLOps Project

## Project Overview

This project builds an **End-to-End Machine Learning System** to predict whether a client will have **payment difficulties** while repaying a loan using Machine Learning techniques.  
It focuses heavily on **MLOps practices** proper preprocessing, pipeline, experiment tracking with MLflow, model deployment, and version control.

The project is based on the Home Credit Default Risk dataset.

---

## Objectives

- Perform data preprocessing and cleaning
- Handle missing values and categorical variables
- Create useful engineered features
- Handle class imbalance using SMOTE
- Build ML Pipeline and train models
- Track experiments using MLflow
- Deploy the trained model using FastAPI
- Maintain clean GitHub repository with proper commit history

---

## Technologies Used

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, LightGBM
- **MLOps**: MLflow
- **Imbalance Handling**: SMOTE (imbalanced-learn)
- **Deployment**: FastAPI + Uvicorn
- **Version Control**: Git & GitHub

---

## Project Structure

```text
home-credit-default-risk-mlops/
│
├── data/                       # Raw and processed datasets
│
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis
│
├── src/
│   ├── preprocessing.py       # Data preprocessing
│   ├── balancing.py           # SMOTE balancing
│   ├── train.py               # Model training
│   └── app.py                 # FastAPI deployment
│
├── models/                    # Saved models and encoders
│
├── mlruns/                    # MLflow experiment tracking
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

- **Dataset used**: Home Credit Default Risk Dataset
- **Rows**: 307,511
- **Columns**: 122
- **Problem**: Binary Classification (Highly Imbalanced)
- **Source**: Kaggle Competition

Note:
Dataset files are not pushed to GitHub (added in .gitignore)

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Step 1: Add Dataset

Place `application_train.csv` inside the `data/` folder.

---

### Step 2: Run Preprocessing

```bash
python src/preprocessing.py
```

---

### Step 3: Run Data Balancing

```bash
python src/balancing.py
```

---

### Step 4: Train Model with MLflow

```bash
python src/train.py
```

---

### Step 5: Run MLflow UI

```bash
mlflow ui
```

---

### Step 6: Deploy Model

```bash
uvicorn src.app:app --reload
```

---

## Current Project Status

- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Preprocessing & Feature Engineering
- ✅ Class Imbalance Handling using SMOTE
- 🔄 Model Training with ML Pipeline + MLflow
- 🔄 Model Deployment using FastAPI (Upcoming)
- 🔄 GitHub Actions