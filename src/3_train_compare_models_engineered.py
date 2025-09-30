import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve

# Load engineered dataset
df = joblib.load('Patient-Readmission-Prediction/models/df_engineered.pkl')

# Split features/target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# SMOTE for balanced training
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standardize for LogReg + MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Dataset ready. Train shape: {X_train_res.shape}")

# Output dir for plots
OUT_DIR = Path('Patient-Readmission-Prediction') / 'output/train'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_classification_report(y_true, y_pred, model_name, out_dir=OUT_DIR):
    """Create a visualization of the classification report.

    Produces a grouped bar chart for precision/recall/f1-score (for each class plus
    accuracy, macro avg and weighted avg) and a separate support bar chart.
    """
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, output_dict=True)
    # Convert to DataFrame; handle 'accuracy' which is a float in some sklearn versions
    rows = []
    for k, v in report.items():
        if k == 'accuracy':
            rows.append({'label': 'accuracy', 'precision': v, 'recall': v, 'f1-score': v, 'support': len(y_true)})
        else:
            rows.append({'label': k, 'precision': v.get('precision', 0), 'recall': v.get('recall', 0), 'f1-score': v.get('f1-score', 0), 'support': v.get('support', 0)})

    rep_df = pd.DataFrame(rows).set_index('label')

    # Order rows: put class labels first (sorted naturally), then accuracy, macro avg, weighted avg
    idx_order = [r for r in rep_df.index if r not in ('accuracy', 'macro avg', 'weighted avg')]
    for special in ('accuracy', 'macro avg', 'weighted avg'):
        if special in rep_df.index:
            idx_order.append(special)
    rep_df = rep_df.loc[idx_order]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 1]})

    # Grouped bar for precision/recall/f1
    metrics = ['precision', 'recall', 'f1-score']
    rep_df[metrics].plot(kind='bar', ax=axes[0], colormap='tab10')
    axes[0].set_title(f'{model_name} — Precision / Recall / F1 by label')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='upper right')

    # Support bar chart on right
    rep_df['support'].plot(kind='bar', ax=axes[1], color='C3')
    axes[1].set_title('Support')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle(f'Classification Report — {model_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = out_dir / f'{model_name.lower().replace(" ", "_")}_classification_report.png'
    plt.savefig(out_path)
    plt.close()


# -------------------
# Logistic Regression
# -------------------
print("\n🔹 Logistic Regression")
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
logreg.fit(X_train_scaled, y_train_res)
log_preds = logreg.predict(X_test_scaled)
log_proba = logreg.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, log_preds))
print("ROC AUC Score:", roc_auc_score(y_test, log_proba))
# save classification report plot for logistic regression
plot_classification_report(y_test, log_preds, 'Logistic Regression', out_dir=OUT_DIR)

# --- Logistic Regression diagnostics: coefficients ---
try:
    if hasattr(X_train_res, 'columns'):
        feature_names = list(X_train_res.columns)
    else:
        feature_names = list(X.columns)
    coefs = logreg.coef_
    # handle multiclass vs binary
    if coefs.ndim == 2 and coefs.shape[0] > 1:
        coef_vals = np.mean(np.abs(coefs), axis=0)
    else:
        coef_vals = np.abs(coefs).ravel()
    coef_df = pd.DataFrame({'feature': feature_names, 'coef_abs': coef_vals})
    coef_df = coef_df.sort_values('coef_abs', ascending=False)
    plt.figure(figsize=(8,10))
    sns.barplot(x='coef_abs', y='feature', data=coef_df.head(30), palette='coolwarm')
    plt.title('Top 30 Absolute Coefficients — Logistic Regression')
    plt.xlabel('Absolute Coefficient')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'logreg_top30_coefficients.png')
    plt.close()
except Exception as e:
    print('Warning: could not create logistic coefficient plot:', e)

# -------------------
# MLP Classifier
# -------------------
print("\n🔹 MLP Classifier")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=150,
                    random_state=42, early_stopping=True)
mlp.fit(X_train_scaled, y_train_res)
mlp_preds = mlp.predict(X_test_scaled)
mlp_proba = mlp.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, mlp_preds))
print("ROC AUC Score:", roc_auc_score(y_test, mlp_proba))
# save classification report plot for MLP
plot_classification_report(y_test, mlp_preds, 'MLP Classifier', out_dir=OUT_DIR)

# --- MLP diagnostics: loss curve ---
try:
    if hasattr(mlp, 'loss_curve_'):
        plt.figure(figsize=(6,4))
        plt.plot(mlp.loss_curve_)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('MLP Training Loss Curve')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'mlp_loss_curve.png')
        plt.close()
except Exception as e:
    print('Warning: could not create MLP loss plot:', e)

# -------------------
# XGBoost
# -------------------
print("\n🔹 XGBoost Classifier")
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric=['logloss', 'auc']
)
xgb.fit(
    X_train_res, y_train_res,
    eval_set=[(X_train_res, y_train_res), (X_test, y_test)],
    verbose=False
)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, xgb_preds))
print("ROC AUC Score:", roc_auc_score(y_test, xgb_proba))
# save classification report plot for XGBoost
plot_classification_report(y_test, xgb_preds, 'XGBoost', out_dir=OUT_DIR)

# --- XGBoost diagnostics: training logloss / AUC per round ---
try:
    evals_result = xgb.evals_result()
    # evals_result is like {'validation_0': {'logloss': [...], 'auc':[...}], 'validation_1': {...}}
    # Plot logloss
    plt.figure(figsize=(6,4))
    for k in evals_result:
        if 'logloss' in evals_result[k]:
            plt.plot(evals_result[k]['logloss'], label=f'{k} logloss')
    plt.xlabel('Round')
    plt.ylabel('Logloss')
    plt.title('XGBoost Logloss per Round')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'xgb_logloss_per_round.png')
    plt.close()

    # Plot AUC per round if available
    plt.figure(figsize=(6,4))
    for k in evals_result:
        if 'auc' in evals_result[k]:
            plt.plot(evals_result[k]['auc'], label=f'{k} auc')
    plt.xlabel('Round')
    plt.ylabel('AUC')
    plt.title('XGBoost AUC per Round')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'xgb_auc_per_round.png')
    plt.close()
except Exception as e:
    print('Warning: could not create XGBoost eval plots:', e)

# Save all models + scaler
joblib.dump(logreg, 'Patient-Readmission-Prediction/models/logreg_engineered.pkl')
joblib.dump(mlp, 'Patient-Readmission-Prediction/models/mlp_engineered.pkl')
joblib.dump(xgb, 'Patient-Readmission-Prediction/models/xgb_engineered.pkl')
joblib.dump(scaler, 'Patient-Readmission-Prediction/models/scaler_engineered.pkl')

# --- Combined ROC plot for comparison ---
try:
    plt.figure(figsize=(7,6))
    fpr, tpr, _ = roc_curve(y_test, log_proba)
    plt.plot(fpr, tpr, label=f'LogReg (AUC={roc_auc_score(y_test, log_proba):.3f})')
    fpr, tpr, _ = roc_curve(y_test, mlp_proba)
    plt.plot(fpr, tpr, label=f'MLP (AUC={roc_auc_score(y_test, mlp_proba):.3f})')
    fpr, tpr, _ = roc_curve(y_test, xgb_proba)
    plt.plot(fpr, tpr, label=f'XGB (AUC={roc_auc_score(y_test, xgb_proba):.3f})')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison — Models')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'models_comparison_roc.png')
    plt.close()
except Exception as e:
    print('Warning: could not create combined ROC plot:', e)

print("\n✅ All models trained and saved.")
