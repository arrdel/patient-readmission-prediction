# import joblib
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from imblearn.over_sampling import SMOTE

# # Load preprocessed data
# X_train = joblib.load('patient-readmission-prediction/data/X_train.pkl')
# X_test = joblib.load('patient-readmission-prediction/data/X_test.pkl')
# y_train = joblib.load('patient-readmission-prediction/data/y_train.pkl')
# y_test = joblib.load('patient-readmission-prediction/data/y_test.pkl')

# # Apply SMOTE
# # Apply SMOTE
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Clean column names for XGBoost compatibility
# X_train_res.columns = X_train_res.columns.str.replace('[<>\[\]]', '', regex=True)
# X_test.columns = X_test.columns.str.replace('[<>\[\]]', '', regex=True)


# print("✅ SMOTE applied")
# print("Resampled target distribution:\n", y_train_res.value_counts())

# # Calculate scale_pos_weight = # negative / # positive (for original class imbalance)
# scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
# print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

# # Train XGBoost
# xgb_model = XGBClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=6,
#     random_state=42,
#     scale_pos_weight=scale_pos_weight,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )

# print("\n🔹 Training: XGBoost + SMOTE + Imbalance Handling")
# xgb_model.fit(X_train_res, y_train_res)

# # Evaluate
# preds = xgb_model.predict(X_test)
# proba = xgb_model.predict_proba(X_test)[:, 1]

# print("Classification Report:")
# print(classification_report(y_test, preds))

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, preds))

# print("ROC AUC Score:", roc_auc_score(y_test, proba))

# # Save the model
# joblib.dump(xgb_model, 'patient-readmission-prediction/data/readmission_model_xgboost.pkl')
# print("\n✅ XGBoost model saved as 'readmission_model_xgboost.pkl'")





# import joblib
# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from imblearn.over_sampling import SMOTE

# # Load preprocessed data
# X_train = joblib.load('patient-readmission-prediction/data/X_train.pkl')
# X_test = joblib.load('patient-readmission-prediction/data/X_test.pkl')
# y_train = joblib.load('patient-readmission-prediction/data/y_train.pkl')
# y_test = joblib.load('patient-readmission-prediction/data/y_test.pkl')

# # Apply SMOTE
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Ensure DataFrame format + column names
# X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
# y_train_res = pd.Series(y_train_res, name="target")
# X_test = pd.DataFrame(X_test, columns=X_train.columns)

# # Clean column names for XGBoost compatibility
# X_train_res.columns = X_train_res.columns.str.replace(r'[<>\[\]]', '', regex=True)
# X_test.columns = X_test.columns.str.replace(r'[<>\[\]]', '', regex=True)

# print("✅ SMOTE applied")
# print("Resampled target distribution:\n", y_train_res.value_counts())

# # Handle class imbalance
# if len(y_train.value_counts()) == 2:
#     scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
# else:
#     scale_pos_weight = 1.0
# print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

# # Train XGBoost
# xgb_model = XGBClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=6,
#     random_state=42,
#     scale_pos_weight=scale_pos_weight,
#     eval_metric='logloss'
# )

# print("\n🔹 Training: XGBoost + SMOTE + Imbalance Handling")
# xgb_model.fit(X_train_res, y_train_res)

# # Evaluate
# preds = xgb_model.predict(X_test)
# proba = xgb_model.predict_proba(X_test)[:, 1]

# print("Classification Report:")
# print(classification_report(y_test, preds))

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, preds))

# print("ROC AUC Score:", roc_auc_score(y_test, proba))

# # Save model
# joblib.dump(xgb_model, 'patient-readmission-prediction/data/readmission_model_xgboost.pkl')
# print("\n✅ XGBoost model saved as 'readmission_model_xgboost.pkl'")


import joblib
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# ------------------------
# Safe loader for joblib files
# ------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except ModuleNotFoundError:
        # Fallback: load with pickle if joblib + numpy mismatch
        with open(path, "rb") as f:
            return pickle.load(f)

# ------------------------
# Load preprocessed data
# ------------------------
X_train = safe_load('patient-readmission-prediction/models/X_train.pkl')
X_test = safe_load('patient-readmission-prediction/models/X_test.pkl')
y_train = safe_load('patient-readmission-prediction/models/y_train.pkl')
y_test = safe_load('patient-readmission-prediction/models/y_test.pkl')

# Ensure correct pandas types
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)
if not isinstance(y_train, pd.Series):
    y_train = pd.Series(y_train, name="target")
if not isinstance(y_test, pd.Series):
    y_test = pd.Series(y_test, name="target")

# ------------------------
# Apply SMOTE
# ------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Keep DataFrame + column names after SMOTE
X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
y_train_res = pd.Series(y_train_res, name="target")

# Clean column names for XGBoost compatibility
X_train_res.columns = X_train_res.columns.str.replace(r'[<>\[\]]', '', regex=True)
X_test.columns = X_test.columns.str.replace(r'[<>\[\]]', '', regex=True)

print("✅ SMOTE applied")
print("Resampled target distribution:\n", y_train_res.value_counts())

# ------------------------
# Handle class imbalance
# ------------------------
if len(y_train.value_counts()) == 2:
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
else:
    scale_pos_weight = 1.0
print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

# ------------------------
# Train XGBoost
# ------------------------
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

print("\n🔹 Training: XGBoost + SMOTE + Imbalance Handling")
xgb_model.fit(X_train_res, y_train_res)

# ------------------------
# Evaluate
# ------------------------
preds = xgb_model.predict(X_test)
proba = xgb_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

print("ROC AUC Score:", roc_auc_score(y_test, proba))

# ------------------------
# Save the model
# ------------------------
joblib.dump(xgb_model, 'patient-readmission-prediction/models/readmission_model_xgboost.pkl')
print("\n✅ XGBoost model saved as 'readmission_model_xgboost.pkl'")
