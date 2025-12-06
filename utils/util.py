# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path

# # Setup
# project_root = Path(__file__).parent.parent
# data_path = project_root / "data" / "diabetic_data.csv"
# output_path = project_root / "output" / "visualizations"
# output_path.mkdir(parents=True, exist_ok=True)

# # Load data
# df = pd.read_csv(data_path)

# # Convert readmitted to binary
# if df['readmitted'].dtype == 'object':
#     df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
# else:
#     df['readmitted_binary'] = df['readmitted']

# # Create figure with subplots
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Plot 1: Count plot
# sns.countplot(data=df, x='readmitted_binary', ax=axes[0], palette=['#3498db', '#e74c3c'])
# axes[0].set_xlabel('Readmission Status', fontsize=12, fontweight='bold')
# axes[0].set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
# axes[0].set_title('Class Distribution: Readmission vs No Readmission', fontsize=14, fontweight='bold')
# axes[0].set_xticklabels(['No Readmission (0)', 'Readmitted <30 days (1)'])

# # Add counts on bars
# for container in axes[0].containers:
#     axes[0].bar_label(container, fmt='%d', fontsize=11)

# # Plot 2: Pie chart
# counts = df['readmitted_binary'].value_counts()
# colors = ['#3498db', '#e74c3c']
# explode = (0, 0.1)  # Explode the minority class
# axes[1].pie(counts, labels=['No Readmission', 'Readmitted <30 days'], 
#             autopct='%1.1f%%', colors=colors, explode=explode,
#             startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
# axes[1].set_title('Class Imbalance Ratio', fontsize=14, fontweight='bold')

# plt.tight_layout()
# plt.savefig(output_path / "class_distribution.png", dpi=300, bbox_inches='tight')
# print(f"✓ Class distribution saved")

# # Print statistics
# print(f"\nClass Distribution:")
# print(f"  No Readmission: {counts[0]:,} ({counts[0]/len(df)*100:.1f}%)")
# print(f"  Readmitted <30d: {counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")
# print(f"  Imbalance Ratio: {counts[0]/counts[1]:.2f}:1")

# plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import SMOTE
# from pathlib import Path

# # Setup
# project_root = Path(__file__).parent.parent
# output_path = project_root / "output" / "visualizations"
# output_path.mkdir(parents=True, exist_ok=True)

# # Simulate data (replace with your actual preprocessed data)
# from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
#                           n_redundant=5, n_classes=2, weights=[0.89, 0.11],
#                           random_state=42)

# # Apply PCA for 2D visualization
# pca = PCA(n_components=2, random_state=42)
# X_pca = pca.fit_transform(X)

# # Apply SMOTE
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)
# X_resampled_pca = pca.transform(X_resampled)

# # Create visualization
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# # Before SMOTE
# scatter1 = axes[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], 
#                           c='#3498db', alpha=0.6, s=30, label='No Readmission')
# scatter2 = axes[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], 
#                           c='#e74c3c', alpha=0.6, s=30, label='Readmitted')
# axes[0].set_title('Before SMOTE: Imbalanced Dataset', fontsize=14, fontweight='bold')
# axes[0].set_xlabel('First Principal Component', fontsize=11)
# axes[0].set_ylabel('Second Principal Component', fontsize=11)
# axes[0].legend(loc='best', fontsize=10)
# axes[0].grid(alpha=0.3)
# axes[0].text(0.02, 0.98, f'Class 0: {sum(y==0):,}\nClass 1: {sum(y==1):,}',
#             transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# # After SMOTE
# scatter3 = axes[1].scatter(X_resampled_pca[y_resampled==0, 0], 
#                           X_resampled_pca[y_resampled==0, 1],
#                           c='#3498db', alpha=0.6, s=30, label='No Readmission')
# scatter4 = axes[1].scatter(X_resampled_pca[y_resampled==1, 0], 
#                           X_resampled_pca[y_resampled==1, 1],
#                           c='#e74c3c', alpha=0.6, s=30, label='Readmitted (Original)')
# # Highlight synthetic samples
# synthetic_indices = np.arange(len(y), len(y_resampled))
# synthetic_mask = np.isin(np.arange(len(y_resampled)), synthetic_indices) & (y_resampled == 1)
# axes[1].scatter(X_resampled_pca[synthetic_mask, 0], 
#                X_resampled_pca[synthetic_mask, 1],
#                c='#f39c12', alpha=0.4, s=20, marker='x', label='Synthetic Samples')

# axes[1].set_title('After SMOTE: Balanced Dataset', fontsize=14, fontweight='bold')
# axes[1].set_xlabel('First Principal Component', fontsize=11)
# axes[1].set_ylabel('Second Principal Component', fontsize=11)
# axes[1].legend(loc='best', fontsize=10)
# axes[1].grid(alpha=0.3)
# axes[1].text(0.02, 0.98, f'Class 0: {sum(y_resampled==0):,}\nClass 1: {sum(y_resampled==1):,}',
#             transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
#             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# plt.tight_layout()
# plt.savefig(output_path / "smote_effect.png", dpi=300, bbox_inches='tight')
# print("✓ SMOTE effect visualization saved")
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import precision_recall_curve, auc
# from pathlib import Path

# # Setup
# project_root = Path(__file__).parent.parent
# output_path = project_root / "output" / "visualizations"
# output_path.mkdir(parents=True, exist_ok=True)

# # Simulate predictions (replace with your actual model predictions)
# np.random.seed(42)
# y_true = np.random.choice([0, 1], size=1000, p=[0.89, 0.11])

# # Simulate predictions for three models
# y_pred_logreg = np.random.beta(2, 5, size=1000)
# y_pred_mlp = np.random.beta(2.2, 4.8, size=1000)
# y_pred_xgb = np.random.beta(2.5, 4.5, size=1000)

# models = {
#     'Logistic Regression': y_pred_logreg,
#     'MLP': y_pred_mlp,
#     'XGBoost': y_pred_xgb
# }

# colors = ['#3498db', '#2ecc71', '#e74c3c']

# plt.figure(figsize=(10, 8))

# for (name, y_pred), color in zip(models.items(), colors):
#     precision, recall, _ = precision_recall_curve(y_true, y_pred)
#     pr_auc = auc(recall, precision)
#     plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})',
#              linewidth=2, color=color)

# # Baseline (random classifier)
# baseline = sum(y_true) / len(y_true)
# plt.plot([0, 1], [baseline, baseline], 'k--', linewidth=1.5, 
#          label=f'Random Classifier (AUC = {baseline:.3f})')

# plt.xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
# plt.ylabel('Precision', fontsize=13, fontweight='bold')
# plt.title('Precision-Recall Curves: Model Comparison', fontsize=15, fontweight='bold', pad=20)
# plt.legend(loc='best', fontsize=11, framealpha=0.9)
# plt.grid(alpha=0.3)
# plt.xlim([0, 1])
# plt.ylim([0, 1])

# plt.tight_layout()
# plt.savefig(output_path / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
# print("✓ Precision-Recall curves saved")
# plt.show()


# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.metrics import confusion_matrix
# from pathlib import Path

# # Setup
# project_root = Path(__file__).parent.parent
# output_path = project_root / "output" / "visualizations"
# output_path.mkdir(parents=True, exist_ok=True)

# # Simulate confusion matrices (replace with actual predictions)
# np.random.seed(42)
# n_samples = 15265  # Test set size

# y_true = np.random.choice([0, 1], size=n_samples, p=[0.89, 0.11])

# # Simulate predictions
# y_pred_logreg = np.where(np.random.random(n_samples) > 0.12, 0, 1)
# y_pred_mlp = np.where(np.random.random(n_samples) > 0.11, 0, 1)
# y_pred_xgb = np.where(np.random.random(n_samples) > 0.13, 0, 1)

# models = {
#     'Logistic Regression': y_pred_logreg,
#     'MLP': y_pred_mlp,
#     'XGBoost': y_pred_xgb
# }

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for ax, (name, y_pred) in zip(axes, models.items()):
#     cm = confusion_matrix(y_true, y_pred)
    
#     # Normalize
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
#     # Plot
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
#                 cbar_kws={'label': 'Count'}, square=True,
#                 xticklabels=['Predicted: No', 'Predicted: Yes'],
#                 yticklabels=['Actual: No', 'Actual: Yes'])
    
#     # Add percentages
#     for i in range(2):
#         for j in range(2):
#             text = ax.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]*100:.1f}%)',
#                           ha="center", va="center", color="darkred", fontsize=9)
    
#     ax.set_title(f'{name}', fontsize=13, fontweight='bold', pad=10)
#     ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
#     ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

# plt.tight_layout()
# plt.savefig(output_path / "confusion_matrices_comparison.png", dpi=300, bbox_inches='tight')
# print("✓ Confusion matrices saved")
# plt.show()


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # Setup
# project_root = Path(__file__).parent.parent
# output_path = project_root / "output" / "visualizations"
# output_path.mkdir(parents=True, exist_ok=True)

# # Simulate feature importances (replace with actual model importances)
# features = ['num_inpatient', 'num_emergency', 'time_in_hospital', 'num_medications',
#             'num_diagnoses', 'num_procedures', 'age', 'num_lab_procedures']

# # Different models rank features differently
# logreg_importance = np.array([0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05])
# xgb_importance = np.array([0.22, 0.20, 0.16, 0.14, 0.11, 0.09, 0.05, 0.03])

# # Create DataFrame
# df_importance = pd.DataFrame({
#     'Feature': features,
#     'Logistic Regression': logreg_importance,
#     'XGBoost': xgb_importance
# })

# # Sort by average importance
# df_importance['Average'] = (df_importance['Logistic Regression'] + df_importance['XGBoost']) / 2
# df_importance = df_importance.sort_values('Average', ascending=True)

# # Plot
# fig, ax = plt.subplots(figsize=(12, 8))

# y_pos = np.arange(len(features))
# width = 0.35

# bars1 = ax.barh(y_pos - width/2, df_importance['Logistic Regression'], 
#                 width, label='Logistic Regression', color='#3498db', alpha=0.8)
# bars2 = ax.barh(y_pos + width/2, df_importance['XGBoost'], 
#                 width, label='XGBoost', color='#e74c3c', alpha=0.8)

# ax.set_yticks(y_pos)
# ax.set_yticklabels(df_importance['Feature'], fontsize=11)
# ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
# ax.set_title('Top Feature Importances: Model Comparison', fontsize=14, fontweight='bold', pad=20)
# ax.legend(loc='lower right', fontsize=11)
# ax.grid(axis='x', alpha=0.3)

# # Add value labels
# for bars in [bars1, bars2]:
#     for bar in bars:
#         width = bar.get_width()
#         ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
#                 f'{width:.3f}', ha='left', va='center', fontsize=9)

# plt.tight_layout()
# plt.savefig(output_path / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
# print("✓ Feature importance comparison saved")
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# # Setup
# project_root = Path(__file__).parent.parent
# output_path = project_root / "output" / "visualizations"
# output_path.mkdir(parents=True, exist_ok=True)

# # Simulate training history (replace with actual training logs)
# epochs = np.arange(1, 101)

# # MLP training curves
# train_loss = 0.4 * np.exp(-epochs/30) + 0.15
# val_loss = 0.45 * np.exp(-epochs/35) + 0.18 + np.random.normal(0, 0.01, len(epochs))

# fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# # Loss curve
# axes[0].plot(epochs, train_loss, label='Training Loss', linewidth=2, color='#3498db')
# axes[0].plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='#e74c3c')
# axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
# axes[0].set_ylabel('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
# axes[0].set_title('MLP Training: Loss Curves', fontsize=14, fontweight='bold')
# axes[0].legend(fontsize=11)
# axes[0].grid(alpha=0.3)

# # Sample size vs performance
# train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# train_scores = 0.65 + 0.28 * (1 - np.exp(-train_sizes * 5))
# val_scores = 0.60 + 0.33 * (1 - np.exp(-train_sizes * 4.5))

# axes[1].plot(train_sizes * 100, train_scores, 'o-', label='Training F1-Score', 
#             linewidth=2, markersize=8, color='#3498db')
# axes[1].plot(train_sizes * 100, val_scores, 's-', label='Validation F1-Score',
#             linewidth=2, markersize=8, color='#e74c3c')
# axes[1].fill_between(train_sizes * 100, train_scores, val_scores, alpha=0.2, color='gray')
# axes[1].set_xlabel('Training Data Size (%)', fontsize=12, fontweight='bold')
# axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
# axes[1].set_title('Learning Curve: Performance vs Training Size', fontsize=14, fontweight='bold')
# axes[1].legend(fontsize=11)
# axes[1].grid(alpha=0.3)
# axes[1].set_ylim([0.5, 1.0])

# plt.tight_layout()
# plt.savefig(output_path / "learning_curves.png", dpi=300, bbox_inches='tight')
# print("✓ Learning curves saved")
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# # Setup
# project_root = Path(__file__).parent.parent
# output_path = project_root / "output" / "visualizations"
# output_path.mkdir(parents=True, exist_ok=True)

# # Metrics
# categories = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Accuracy']
# N = len(categories)

# # Model scores (normalized to 0-1)
# logreg_scores = [0.88, 0.98, 0.93, 0.59, 0.86]
# mlp_scores = [0.88, 0.99, 0.93, 0.58, 0.85]
# xgb_scores = [0.88, 0.98, 0.93, 0.62, 0.86]

# # Compute angle for each axis
# angles = [n / float(N) * 2 * np.pi for n in range(N)]
# logreg_scores += logreg_scores[:1]
# mlp_scores += mlp_scores[:1]
# xgb_scores += xgb_scores[:1]
# angles += angles[:1]

# # Plot
# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# ax.plot(angles, logreg_scores, 'o-', linewidth=2, label='Logistic Regression', color='#3498db')
# ax.fill(angles, logreg_scores, alpha=0.15, color='#3498db')

# ax.plot(angles, mlp_scores, 's-', linewidth=2, label='MLP', color='#2ecc71')
# ax.fill(angles, mlp_scores, alpha=0.15, color='#2ecc71')

# ax.plot(angles, xgb_scores, '^-', linewidth=2, label='XGBoost', color='#e74c3c')
# ax.fill(angles, xgb_scores, alpha=0.15, color='#e74c3c')

# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
# ax.set_ylim(0, 1)
# ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
# ax.grid(True, alpha=0.3)

# plt.title('Model Performance Comparison\nMultiple Metrics', 
#           fontsize=16, fontweight='bold', pad=30)
# plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

# plt.tight_layout()
# plt.savefig(output_path / "performance_radar_chart.png", dpi=300, bbox_inches='tight')
# print("✓ Radar chart saved")
# plt.show()


"""
Master script to generate all presentation visualizations
"""
import subprocess
import sys
from pathlib import Path

# List of visualization scripts
scripts = [
    'visualizations/class_distribution.py',
    'visualizations/smote_effect.py',
    'visualizations/precision_recall_curve.py',
    'visualizations/confusion_matrices.py',
    'visualizations/feature_importance_comparison.py',
    'visualizations/learning_curves.py',
    'visualizations/performance_radar.py'
]

print("=" * 80)
print("GENERATING ALL PRESENTATION VISUALIZATIONS")
print("=" * 80)

for i, script in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] Running {script}...")
    try:
        result = subprocess.run([sys.executable, script], 
                              capture_output=True, text=True, check=True)
        print(f"✓ {script} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {script}:")
        print(e.stderr)
    except FileNotFoundError:
        print(f"⚠ {script} not found, skipping...")

print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS GENERATED!")
print("=" * 80)
print("\nCheck output/visualizations/ for all plots")