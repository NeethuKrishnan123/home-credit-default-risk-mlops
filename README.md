# Home Credit Default Risk Prediction - MLOps Project

## Project Overview
This project predicts whether a client will have payment difficulties (loan default) using machine learning.  
It demonstrates a complete End-to-End ML Pipeline with MLOps practices using MLflow.


---

## Objective
- Build and deploy a classification model for loan default prediction.
- Show all steps: Preprocessing, Feature Engineering, Model Training, MLflow Tracking, and Deployment.
- Follow MLOps best practices as per internship requirements.

---

## Dataset
- **Name**: Home Credit Default Risk
- **Rows**: 307,511
- **Columns**: 122
- **Type**: Binary Classification (Imbalanced data)

---

## Technologies & Tools Used
- Python, Pandas, Scikit-learn, LightGBM
- MLflow (Experiment Tracking)
- FastAPI (Deployment)
- Git & GitHub
- GitHub Actions (Planned)

---

## Project Structure
home-credit-default-risk-mlops/
├── data/                    # Raw + Processed data (not in git)
├── notebooks/               # EDA Jupyter notebooks
├── src/                     # Main code
│   ├── preprocessing.py
│   ├── train.py
│   └── app.py
├── models/                  # Saved models & encoders
├── mlruns/                  # MLflow tracking data
├── requirements.txt
└── README.md
text

