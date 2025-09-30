import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data_path = "Patient-Readmission-Prediction/data/diabetic_data.csv"
df = pd.read_csv(data_path)

# Drop identifier and high-missing-value columns
cols_to_drop = [
    'encounter_id', 'patient_nbr', 'weight', 'payer_code',
    'medical_specialty', 'max_glu_serum', 'A1Cresult'
]
df.drop(columns=cols_to_drop, inplace=True)

# Replace '?' with NaN and drop missing rows
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Identify categorical columns BEFORE modifying target
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('readmitted')  # Safely remove readmitted if present

# Encode target variable: readmitted
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# One-hot encode all remaining categorical variables
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Split into features and target
X = df_encoded.drop('readmitted', axis=1)
y = df_encoded['readmitted']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Save preprocessed data
joblib.dump(X_train, 'Patient-Readmission-Prediction/models/X_train.pkl')
joblib.dump(X_test, 'Patient-Readmission-Prediction/models/X_test.pkl')
joblib.dump(y_train, 'Patient-Readmission-Prediction/models/y_train.pkl')
joblib.dump(y_test, 'Patient-Readmission-Prediction/models/y_test.pkl')

print("✅ Preprocessing complete.")
print(f"Features: {X_train.shape[1]}")
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
