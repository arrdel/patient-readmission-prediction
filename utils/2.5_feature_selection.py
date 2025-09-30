import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load full training data and model
X_train = joblib.load('patient-readmission-prediction/models/X_train.pkl')
X_test = joblib.load('patient-readmission-prediction/models/X_test.pkl')
y_train = joblib.load('patient-readmission-prediction/models/y_train.pkl')
y_test = joblib.load('patient-readmission-prediction/models/y_test.pkl')

# Clean feature names (match how XGBoost sees them)
X_train.columns = X_train.columns.str.replace('[<>\[\]]', '', regex=True)
X_test.columns = X_test.columns.str.replace('[<>\[\]]', '', regex=True)

# Load trained model
model = joblib.load('patient-readmission-prediction/models/readmission_model_xgboost.pkl')


# Get feature importance from model
booster = model.get_booster()
feature_scores = booster.get_score(importance_type='gain')

importance_df = pd.DataFrame({
    'feature': list(feature_scores.keys()),
    'importance': list(feature_scores.values())
}).sort_values(by='importance', ascending=False)

# Select top N features
TOP_N = 100
top_features = importance_df['feature'].head(TOP_N).tolist()

print(f"✅ Selected Top {TOP_N} Features")

# Reduce datasets
X_train_reduced = X_train[top_features]
X_test_reduced = X_test[top_features]

# Save reduced datasets
joblib.dump(X_train_reduced, 'patient-readmission-prediction/models/X_train_reduced.pkl')
joblib.dump(X_test_reduced, 'patient-readmission-prediction/models/X_test_reduced.pkl')
joblib.dump(y_train, 'patient-readmission-prediction/models/y_train.pkl')
joblib.dump(y_test, 'patient-readmission-prediction/models/y_test.pkl')

# Save feature importance for review
importance_df.to_csv('patient-readmission-prediction/models/feature_importance.csv', index=False)

print(f"\n✅ Saved reduced datasets and importance list.")

# --- Visualization: selected vs non-selected features ---
OUT_DIR = Path('patient-readmission-prediction') / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_top_features_bar(imp_df, top_n=TOP_N):
    plt.figure(figsize=(10, 8))
    top = imp_df.head(top_n).copy()
    sns.barplot(x='importance', y='feature', data=top, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances (gain)')
    plt.xlabel('Importance (gain)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'top{top_n}_feature_importance.png')
    plt.close()


def plot_importance_distribution(imp_df, top_feats):
    imp_df = imp_df.copy()
    imp_df['selected'] = imp_df['feature'].isin(top_feats)
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=imp_df, x='importance', hue='selected', fill=True, common_norm=False, palette=['C0','C1'])
    plt.title('Importance Distribution: Selected (top) vs Not Selected')
    plt.xlabel('Importance (gain)')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'importance_distribution_selected_vs_rest.png')
    plt.close()


# For a concrete, easy-to-interpret comparison, pick a few numeric features that likely exist
# and plot their distributions grouped by whether the feature is selected. We'll find numeric
# columns from X_train and show boxplots for the top few numeric features and a few from rest.
def plot_numeric_feature_boxplots(X, selected_features, sample_features=5):
    # Identify numeric columns present in X
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return

    # Find numeric features in selected and not selected
    sel_num = [c for c in selected_features if c in numeric_cols]
    rest_num = [c for c in numeric_cols if c not in selected_features]

    # Limit to a few features for compact plotting
    sel_sample = sel_num[:sample_features]
    rest_sample = rest_num[:sample_features]

    # Plot side-by-side boxplots for selected numeric features
    for feature_list, label in [(sel_sample, 'selected'), (rest_sample, 'rest')]:
        if not feature_list:
            continue
        plt.figure(figsize=(max(6, len(feature_list)*1.5), 5))
        sns.boxplot(data=X[feature_list].astype(float), palette='Set3')
        plt.title(f'Boxplots of numeric {label} features (sample)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'boxplot_numeric_{label}_features.png')
        plt.close()


plot_top_features_bar(importance_df, TOP_N)
plot_importance_distribution(importance_df, top_features)
plot_numeric_feature_boxplots(X_train, top_features, sample_features=5)

print(f"\n✅ Saved feature selection visualizations to {OUT_DIR}")
