"""
Simple data visualization script for the diabetes readmission dataset.
Creates several plots (histograms, bar plots, boxplots, scatter) and saves
them to patient-readmission-prediction/output/.

Usage:
    python src/visualize_data.py

The script assumes the data files are in patient-readmission-prediction/data/
and writes PNG files to patient-readmission-prediction/output/.
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output'


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_DIR / 'diabetic_data.csv')
    ids_map = pd.read_csv(DATA_DIR / 'IDS_mapping.csv')
    return df, ids_map


def plot_readmission_counts(df):
    plt.figure(figsize=(6,4))
    order = df['readmitted'].value_counts().index
    sns.countplot(data=df, x='readmitted', order=order, palette='pastel')
    plt.title('Readmission Counts')
    plt.ylabel('Count')
    plt.xlabel('Readmitted')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'readmission_counts.png')
    plt.close()


def plot_age_distribution(df):
    plt.figure(figsize=(8,4))
    # age is like [0-10), [10-20) etc - keep as categorical order
    order = sorted(df['age'].unique(), key=lambda x: int(x.strip('[]').split('-')[0]))
    sns.countplot(data=df, x='age', order=order, palette='mako')
    plt.xticks(rotation=45)
    plt.title('Age Group Distribution')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'age_group_distribution.png')
    plt.close()


def plot_time_in_hospital_hist(df):
    plt.figure(figsize=(6,4))
    sns.histplot(df['time_in_hospital'].dropna(), bins=range(0, df['time_in_hospital'].max()+2), kde=False, color='C0')
    plt.title('Time in Hospital (days)')
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_in_hospital_hist.png')
    plt.close()


def plot_num_medications_by_readmit(df):
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='readmitted', y='num_medications', palette='Set2')
    plt.title('Number of Medications by Readmission Status')
    plt.xlabel('Readmitted')
    plt.ylabel('Num Medications')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'num_medications_by_readmit_box.png')
    plt.close()


def plot_lab_procedures_vs_medications(df):
    plt.figure(figsize=(7,6))
    sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=42), x='num_lab_procedures', y='num_medications', hue='readmitted', alpha=0.6)
    plt.title('Lab Procedures vs Number of Medications (sampled)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'labproc_vs_medications_scatter.png')
    plt.close()


def plot_diagnosis_top_codes(df):
    # show top diagnosis codes for diag_1
    top = df['diag_1'].value_counts().nlargest(20)
    plt.figure(figsize=(8,6))
    sns.barplot(x=top.values, y=top.index, palette='viridis')
    plt.title('Top 20 Primary Diagnosis Codes (diag_1)')
    plt.xlabel('Count')
    plt.ylabel('diag_1')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top20_diag1.png')
    plt.close()


def run_all():
    ensure_output_dir()
    df, ids_map = load_data()
    # Basic plots
    plot_readmission_counts(df)
    plot_age_distribution(df)
    plot_time_in_hospital_hist(df)
    plot_num_medications_by_readmit(df)
    plot_lab_procedures_vs_medications(df)
    plot_diagnosis_top_codes(df)
    print('Saved plots to', OUTPUT_DIR)


if __name__ == '__main__':
    run_all()
