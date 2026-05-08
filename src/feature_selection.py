import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== Feature Selection on Preprocessed Data ===\n")

# Load the preprocessed (but NOT balanced) data
df = pd.read_csv('data/processed_train.csv')
print(f"Original Features: {df.shape[1]-1}")

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Feature Importance
print("Calculating Feature Importance...")
rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Select Top 50 features
top_n = 50
top_features = feature_importance.head(top_n)['Feature'].tolist()

print(f"\nSelected Top {top_n} Features:")
print(feature_importance.head(12))

# Save selected data (still not balanced)
selected_df = df[top_features + ['TARGET']]
selected_df.to_csv('data/processed_train_selected.csv', index=False)

joblib.dump(top_features, 'models/selected_features.pkl')

print(f"\n Feature Selection Completed. Saved {top_n} features.")