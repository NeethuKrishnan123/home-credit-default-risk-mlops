import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== Improved Model Training with Class Weights ===\n")

# Load Data
X_train = pd.read_csv('data/X_train_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')
train_balanced = pd.read_csv('data/train_balanced.csv')
test_data = pd.read_csv('data/test.csv')

y_train = train_balanced['TARGET']
y_test = test_data['TARGET']

print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ============================
# MLflow Experiment
# ============================
mlflow.set_experiment("Home_Credit_Default_Risk")

with mlflow.start_run(run_name="LightGBM_Improved_Class_Weights"):
    
    # Calculate class weights (important for imbalance)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=10,
        num_leaves=128,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        class_weight='balanced'          # Important line
        # scale_pos_weight=scale_pos_weight  # Alternative method
    )
    
    print("Training Improved LightGBM Model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log everything
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("n_estimators", 800)
    mlflow.log_param("learning_rate", 0.03)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("features_used", X_train.shape[1])
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)
    
    mlflow.sklearn.log_model(model, "lightgbm_model")
    
    print("\n" + "="*70)
    print("IMPROVED MODEL TRAINING COMPLETED!")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("="*70)

# Save the improved model
    
joblib.dump(model, 'models/lightgbm_class_weights.pkl')
print("Improved model saved")

