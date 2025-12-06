import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
project_root = Path(__file__).parent
data_path = "/home/adelechinda/home/semester_projects/fall_25/data_mining/patient-readmission-prediction/data/diabetic_data.csv"  # Using original data
output_path = project_root / "output" / "correlation_analysis"
output_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CORRELATION ANALYSIS - Hospital Readmission Prediction")
print("=" * 80)

# Load the data
print("\n[1/6] Loading data...")
try:
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"✗ Error: File not found at {data_path}")
    print("Please ensure the data file exists in the correct location.")
    exit(1)

# Select numerical features for correlation analysis
print("\n[2/6] Selecting numerical features...")
numerical_features = [
    'time_in_hospital',
    'num_lab_procedures', 
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses'
]

# Filter to only include features that exist in the dataframe
available_features = [col for col in numerical_features if col in df.columns]

# Add age if it exists (might need conversion)
if 'age' in df.columns:
    # Age might be categorical like '[70-80)', convert to numeric
    if df['age'].dtype == 'object':
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df['age_numeric'] = df['age'].map(age_map)
        available_features.append('age_numeric')
    else:
        available_features.append('age')

# Add readmitted target (convert to binary if needed)
if 'readmitted' in df.columns:
    if df['readmitted'].dtype == 'object':
        # Convert '<30', '>30', 'NO' to binary (1 for <30, 0 otherwise)
        df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
        available_features.append('readmitted_binary')
    else:
        available_features.append('readmitted')

print(f"✓ Selected {len(available_features)} features:")
for feat in available_features:
    print(f"  - {feat}")

# Create dataframe with selected features
df_selected = df[available_features].copy()

# Handle missing values
print("\n[3/6] Handling missing values...")
missing_before = df_selected.isnull().sum().sum()
df_selected = df_selected.dropna()
missing_after = df_selected.isnull().sum().sum()
print(f"✓ Removed {missing_before} missing values")
print(f"✓ Final dataset: {df_selected.shape[0]} rows, {df_selected.shape[1]} columns")

# Compute correlation matrix
print("\n[4/6] Computing correlation matrix...")
correlation_matrix = df_selected.corr()
print("✓ Correlation matrix computed")

# Create visualizations
print("\n[5/6] Generating visualizations...")

# 1. Full Correlation Heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}
)
plt.title('Feature Correlation Matrix\nHospital Readmission Prediction Dataset', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

heatmap_path = output_path / "feature_correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Heatmap saved: {heatmap_path}")
plt.close()

# 2. Correlation with Target Variable (if exists)
target_col = 'readmitted_binary' if 'readmitted_binary' in available_features else None
if target_col:
    target_corr = correlation_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in target_corr.values]
    bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
    plt.yticks(range(len(target_corr)), target_corr.index, fontsize=10)
    plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    plt.title('Feature Correlations with Readmission Target', 
              fontsize=14, fontweight='bold', pad=20)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    target_corr_path = output_path / "target_correlation_barplot.png"
    plt.savefig(target_corr_path, dpi=300, bbox_inches='tight')
    print(f"✓ Target correlation plot saved: {target_corr_path}")
    plt.close()

# 3. Clustermap for hierarchical clustering
plt.figure(figsize=(12, 10))
sns.clustermap(
    correlation_matrix,
    cmap='coolwarm',
    center=0,
    annot=True,
    fmt='.2f',
    linewidths=0.5,
    figsize=(12, 10),
    cbar_kws={"label": "Correlation Coefficient"}
)
plt.suptitle('Hierarchical Clustering of Feature Correlations', 
             fontsize=14, fontweight='bold', y=0.98)

clustermap_path = output_path / "correlation_clustermap.png"
plt.savefig(clustermap_path, dpi=300, bbox_inches='tight')
print(f"✓ Clustermap saved: {clustermap_path}")
plt.close()

# Save correlation matrix as CSV
print("\n[6/6] Saving results...")
corr_csv_path = output_path / "correlation_matrix.csv"
correlation_matrix.to_csv(corr_csv_path)
print(f"✓ Correlation matrix CSV saved: {corr_csv_path}")

# Generate insights report
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS RESULTS")
print("=" * 80)

# Top positive correlations
print("\n📊 TOP 10 POSITIVE CORRELATIONS (excluding diagonal):")
print("-" * 80)
correlation_pairs = correlation_matrix.where(
    np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
)
top_positive = correlation_pairs.stack().sort_values(ascending=False).head(10)
for i, ((feat1, feat2), corr_value) in enumerate(top_positive.items(), 1):
    print(f"{i:2d}. {feat1:25s} ↔ {feat2:25s} : {corr_value:+.3f}")

# Top negative correlations
print("\n📉 TOP 10 NEGATIVE CORRELATIONS:")
print("-" * 80)
top_negative = correlation_pairs.stack().sort_values(ascending=True).head(10)
for i, ((feat1, feat2), corr_value) in enumerate(top_negative.items(), 1):
    print(f"{i:2d}. {feat1:25s} ↔ {feat2:25s} : {corr_value:+.3f}")

# Correlations with target
if target_col:
    print(f"\n🎯 CORRELATIONS WITH READMISSION TARGET ({target_col}):")
    print("-" * 80)
    target_correlations = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
    for i, (feature, corr_value) in enumerate(target_correlations.items(), 1):
        direction = "+" if correlation_matrix[target_col][feature] > 0 else "-"
        print(f"{i:2d}. {feature:30s} : {direction}{corr_value:.3f}")

# Multicollinearity check
print("\n⚠️  POTENTIAL MULTICOLLINEARITY (|correlation| > 0.7):")
print("-" * 80)
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

if high_corr:
    for feat1, feat2, corr in high_corr:
        print(f"  • {feat1:25s} ↔ {feat2:25s} : {corr:+.3f}")
else:
    print("  ✓ No highly correlated feature pairs detected (threshold = 0.7)")

# Summary statistics
print("\n📈 CORRELATION SUMMARY STATISTICS:")
print("-" * 80)
corr_values = correlation_pairs.stack().values
print(f"  Mean absolute correlation    : {np.abs(corr_values).mean():.3f}")
print(f"  Median absolute correlation  : {np.median(np.abs(corr_values)):.3f}")
print(f"  Max correlation (non-diag)   : {corr_values.max():.3f}")
print(f"  Min correlation              : {corr_values.min():.3f}")
print(f"  Std of correlations          : {corr_values.std():.3f}")

# Save insights to text file
insights_path = output_path / "correlation_insights.txt"
with open(insights_path, 'w') as f:
    f.write("CORRELATION ANALYSIS INSIGHTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write(f"Features analyzed: {len(available_features)}\n\n")
    
    f.write("TOP POSITIVE CORRELATIONS:\n")
    f.write("-" * 80 + "\n")
    for (feat1, feat2), corr in top_positive.items():
        f.write(f"{feat1} ↔ {feat2}: {corr:+.3f}\n")
    
    if target_col:
        f.write(f"\nCORRELATIONS WITH TARGET ({target_col}):\n")
        f.write("-" * 80 + "\n")
        for feature, corr in target_correlations.items():
            direction = "+" if correlation_matrix[target_col][feature] > 0 else "-"
            f.write(f"{feature}: {direction}{corr:.3f}\n")

print(f"\n✓ Insights saved: {insights_path}")

print("\n" + "=" * 80)
print("✅ CORRELATION ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nOutput files saved to: {output_path}/")
print("  1. feature_correlation_heatmap.png")
print("  2. target_correlation_barplot.png")
print("  3. correlation_clustermap.png")
print("  4. correlation_matrix.csv")
print("  5. correlation_insights.txt")
print("\n" + "=" * 80)