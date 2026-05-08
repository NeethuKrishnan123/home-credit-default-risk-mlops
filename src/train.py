import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================
# Load Preprocessed Data
# ============================
X_train = pd.read_csv('data/X_train_scaled.csv')
X_test = pd.read_csv('data/X_test_scaled.csv')

# Load y_train from balanced file
train_balanced = pd.read_csv('data/train_balanced.csv')
y_train = train_balanced['TARGET']

# Load y_test
test_data = pd.read_csv('data/test.csv')
y_test = test_data['TARGET']

print(f"Training Shape: {X_train.shape}")
print(f"Testing Shape : {X_test.shape}")
print(f"Training samples after SMOTE: {len(y_train)}")

# ============================
# MLflow Setup
# ============================
mlflow.set_experiment("Home_Credit_Default_Risk")

with mlflow.start_run(run_name="LightGBM_Final_Model"):
    
    model = lgb.LGBMClassifier(n_estimators=500,learning_rate=0.05,max_depth=12,num_leaves=64,random_state=42,verbose=-1)
    
    print("Training LightGBM Model...")
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
    
    # Log to MLflow
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 12)
    mlflow.log_param("features_used", X_train.shape[1])
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc", auc)
    
    mlflow.sklearn.log_model(model, "lightgbm_model")
    
    print("\n" + "="*65)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("="*65)

# Save final model
joblib.dump(model, 'models/final_model.pkl')
print("\nFinal model saved as 'final_model.pkl'")

