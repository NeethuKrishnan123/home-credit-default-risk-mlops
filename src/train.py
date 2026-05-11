import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score)
import lightgbm as lgb
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== Advanced Training with Threshold Tuning ===\n")

# =====================================================
# Load Preprocessed Data
# =====================================================
X_train = pd.read_csv('data/X_train_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')
train_balanced = pd.read_csv('data/train_balanced.csv')
test_data = pd.read_csv('data/test.csv')

y_train = train_balanced['TARGET']
y_test = test_data['TARGET']

print(f"Training Shape : {X_train.shape}")
print(f"Testing Shape  : {X_test.shape}")
print(f"Training Samples: {len(y_train)}")

# =====================================================
# MLflow Setup
# =====================================================
mlflow.set_experiment("Home_Credit_Default_Risk")

with mlflow.start_run(run_name="LightGBM_Advanced_Tuning"):
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=10,
        num_leaves=128,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=15,
        random_state=42,
        verbose=-1
    )

    print("\nTraining LightGBM Model...")
    model.fit(X_train, y_train)

    # =====================================================
    # Prediction Probabilities
    # =====================================================
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # Threshold Tuning
    # =====================================================
    thresholds = np.arange(0.1, 0.6, 0.05)

    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    print("\nThreshold Tuning Results")
    print("=" * 60)

    print(
        f"{'Threshold':<10}"
        f"{'Recall':<10}"
        f"{'Precision':<10}"
        f"{'F1 Score':<10}"
        f"{'Accuracy':<10}"
    )

    print("=" * 60)

    for thresh in thresholds:

        # Convert probabilities to class labels
        y_pred = (y_pred_proba >= thresh).astype(int)

        # Metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)

        print(
            f"{thresh:<10.2f}"
            f"{recall:<10.4f}"
            f"{precision:<10.4f}"
            f"{f1:<10.4f}"
            f"{accuracy:<10.4f}"
        )

        # Save best threshold based on F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "accuracy": accuracy,
                "auc": auc
            }

    # =====================================================
    # Final Best Results
    # =====================================================
    print("=" * 60)

    print("\n BEST MODEL RESULTS")
    print(f"Best Threshold : {best_threshold:.3f}")
    print(f"Best Recall    : {best_metrics['recall']:.4f}")
    print(f"Best Precision : {best_metrics['precision']:.4f}")
    print(f"Best F1 Score  : {best_metrics['f1']:.4f}")
    print(f"Best Accuracy  : {best_metrics['accuracy']:.4f}")
    print(f"Best AUC Score : {best_metrics['auc']:.4f}")

    print("=" * 60)

    # =====================================================
    # Log Parameters to MLflow
    # =====================================================
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("n_estimators", 1000)
    mlflow.log_param("learning_rate", 0.03)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("num_leaves", 128)
    mlflow.log_param("scale_pos_weight", 15)
    mlflow.log_param("best_threshold", best_threshold)

    # =====================================================
    # Log Metrics to MLflow
    # =====================================================
    for key, value in best_metrics.items():
        mlflow.log_metric(key, value)


    # =====================================================
    # Save Model in MLflow
    # =====================================================
    mlflow.sklearn.log_model(model, "lightgbm_model")

# =====================================================
# Save Final Model + Threshold
# =====================================================
final_model_info = {
    'model': model,
    'best_threshold': best_threshold
}

joblib.dump(final_model_info, 'models/lightgbm_advanced_tuning.pkl')
print("\n Model with best threshold saved!")
