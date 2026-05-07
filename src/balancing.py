import pandas as pd
from imblearn.over_sampling import SMOTE

print("=== Starting Data Balancing ===\n")

# Load preprocessed data
df = pd.read_csv('data/processed_train.csv')

X = df.drop('TARGET', axis=1)
y = df['TARGET']

print("Original Target Distribution:")
print(y.value_counts())

# Apply SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Create balanced dataframe
df_balanced = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), 
                        pd.DataFrame(y_balanced, columns=['TARGET'])], axis=1)

print("\nAfter Balancing:")
print(df_balanced['TARGET'].value_counts())

# Save balanced dataset
df_balanced.to_csv('data/processed_train_balanced.csv', index=False)
print("\n✅ Balanced dataset saved as 'processed_train_balanced.csv'")