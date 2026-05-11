import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== Strong Model Training with scale_pos_weight ===\n")

# Load Data
X_train = pd.read_csv('data/X_train_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')
train_balanced = pd.read_csv('data/train_balanced.csv')
test_data = pd.read_csv('data/test.csv')

y_train = train_balanced['TARGET']
y_test = test_data['TARGET']

# Calculate scale_pos_weight (very important for imbalance)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"scale_pos_weight used: {scale_pos_weight:.2f}")

# ============================
mlflow.set_experiment("Home_Credit_Default_Risk")

with mlflow.start_run(run_name="LightGBM_Strong_scale_pos_weight"):
    
    model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=9,
        num_leaves=80,
        min_child_samples=25,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,     # Key parameter
        is_unbalance=False
    )
    
    print("Training Strong LightGBM Model...")
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
    
    # Log in MLflow
    mlflow.log_param("model", "LightGBM")
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("n_estimators", 1200)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)
    
    mlflow.sklearn.log_model(model, "lightgbm_model")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("="*70)

joblib.dump(model, 'models/lightgbm_scale_pos_weight.pkl')
print("Model saved successfully!")