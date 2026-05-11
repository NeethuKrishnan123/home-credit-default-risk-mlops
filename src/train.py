import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== Training + Threshold Tuning ===\n")

# Load Data
X_train = pd.read_csv('data/X_train_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')
train_balanced = pd.read_csv('data/train_balanced.csv')
test_data = pd.read_csv('data/test.csv')

y_train = train_balanced['TARGET']
y_test = test_data['TARGET']

mlflow.set_experiment("Home_Credit_Default_Risk")

with mlflow.start_run(run_name="LightGBM_With_Threshold_Tuning"):
    
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=10,
        num_leaves=128,
        random_state=42,
        verbose=-1,
        scale_pos_weight=10,           # Aggressive weighting
    )
    
    model.fit(X_train, y_train)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
    best_f1 = 0
    best_threshold = 0.5
    
    print("\nThreshold Tuning Results:")
    print("-" * 50)
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Threshold: {thresh:.2f} | "
              f"Accuracy: {accuracy:.4f} | "
              f"Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | "
              f"F1: {f1:.4f}")
        
        # Save best threshold
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_accuracy = accuracy
            best_recall = recall
            best_precision = precision
            best_auc = auc
    print("-" * 50)

    # Final Results
    print("-" * 50)
    print(f"Best Threshold : {best_threshold}")
    print(f"Accuracy       : {best_accuracy:.4f}")
    print(f"Precision      : {best_precision:.4f}")
    print(f"Recall         : {best_recall:.4f}")
    print(f"F1 Score       : {best_f1:.4f}")
    print(f"AUC Score      : {best_auc:.4f}")
    print("=" * 65)
    
    # Final prediction with best threshold
    final_pred = (y_pred_proba >= best_threshold).astype(int)
    
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("best_threshold", best_threshold)
    mlflow.log_param("scale_pos_weight", 10)
    
    mlflow.log_metric("accuracy", best_accuracy)
    mlflow.log_metric("precision", best_precision)
    mlflow.log_metric("recall", best_recall)
    mlflow.log_metric("f1_score", best_f1)
    mlflow.log_metric("auc", best_auc)

    mlflow.sklearn.log_model(model, "lightgbm_model")
    
# Save Final Model
joblib.dump(model, 'models/lightgbm_threshold_tuned.pkl')
print("\n Training with Threshold Tuning Completed!")