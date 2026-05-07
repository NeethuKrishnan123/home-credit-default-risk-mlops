import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=== Starting Data Preprocessing ===\n")

df = pd.read_csv('data/application_train.csv')
print(f"Original Shape: {df.shape}")

# ===================== Missing Values =====================
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing Values Handled")

# ===================== Feature Engineering =====================
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)

df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25
df['YEARS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']) / 365.25

print("✅ Feature Engineering Done")

# ===================== Encoding =====================
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("✅ Categorical Encoding Done")

# Save Processed Data
df.to_csv('data/processed_train.csv', index=False)
import joblib
joblib.dump(label_encoders, 'models/label_encoders.pkl')

print(f"Final Shape: {df.shape}")
print("✅ Preprocessing Completed!")