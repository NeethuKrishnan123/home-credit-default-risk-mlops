import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================
# 1. Load Selected Features Data
# ============================
df = pd.read_csv('data/processed_train_selected.csv')
print(f"Loaded data with {df.shape[1]-1} features and {df.shape[0]} rows\n")

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# ============================
# 2. Train-Test Split
# ============================
print("Performing Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train Shape: {X_train.shape}")
print(f"Test Shape : {X_test.shape}")

# Save split data
pd.concat([X_train, y_train], axis=1).to_csv('data/train_split.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('data/test.csv', index=False)

print("Train-Test split data saved\n")

# ============================
# 3. Balancing - SMOTE (Only on Training Data)
# ============================
print("Applying SMOTE Balancing on Training Data only...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"After SMOTE → Training samples: {X_train_bal.shape[0]}")

# Save Balanced Training Data
train_balanced = pd.concat([X_train_bal.reset_index(drop=True), y_train_bal.reset_index(drop=True)], axis=1)
train_balanced.to_csv('data/train_balanced.csv', index=False)

print("Balanced training data saved as 'train_balanced.csv'\n")

# ============================
# 4. Scaling (StandardScaler)
# ============================
print("Applying Standard Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_bal.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save Scaled Data
X_train_scaled.to_csv('data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/X_test_scaled.csv', index=False)

# Save Scaler (Very Important for Deployment)
joblib.dump(scaler, 'models/scaler.pkl')
print("Scaling completed and scaler saved for deployment\n")

