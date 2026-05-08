import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# df = pd.read_csv('data/processed_train_balanced.csv')
# print(f"Loaded Balanced Dataset Shape: {df.shape}")

# x = df.drop('TARGET',axis=1)
# y = df['TARGET']

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
# print(f"Training samples: {x_train.shape[0]}")
# print(f"Testing samples: {x_test.shape[0]}")

# # Set MLflow Experiment

# mlflow.set_experiment("Home_Credit_Default_Risk")

# # Train Model with MLflow Tracking

# with mlflow.start_run(run_name="LightGBM_Balanced"):
#     model = lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,max_depth=8,random_state=42,verbose=-1)
#     model.fit(x_train,y_train)
#     # Predictions
#     y_pred = model.predict(x_test)
#     y_pred_proba = model.predict_proba(x_test)[:,1]
#     # Calculate Metrics
#     accuracy = accuracy_score(y_test,y_pred)
#     precision = precision_score(y_test,y_pred)
#     recall = recall_score(y_test,y_pred)
#     f1 = f1_score(y_test,y_pred)
#     auc = roc_auc_score(y_test,y_pred_proba)

#     # Log Parameters
#     mlflow.log_metric("accuracy",accuracy)
#     mlflow.log_metric("precision",precision)
#     mlflow.log_metric("recall",recall)
#     mlflow.log_metric("f1_score",f1)
#     mlflow.log_metric("auc",auc)

#     # Log Model   
#     mlflow.sklearn.log_model(model,"model")

#     print("\n" + "=" * 50)
#     print("Model Training Completed")
#     print(f"Accuracy : {accuracy:'4f'}")
#     print(f"Precision : {precision:'4f'}")
#     print(f"Recall : {recall:'4f'}")
#     print(f"F1 Score : {f1:'4f'}")
#     print(f"AUC Score : {auc:'4f'}")
#     print("="*50)
# print("\nMLflow Run Completed. You can now view it in MLflow UI.")