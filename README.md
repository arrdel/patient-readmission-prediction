# 🏥 Patient Readmission Prediction

This project builds and evaluates machine learning models to predict **30-day hospital readmissions** using a real-world diabetes dataset. The pipeline covers data preprocessing, feature engineering, model training (with SMOTE to address class imbalance)and threshold tuning.

---

## 📌 Dataset

- **Source:** [UCI Diabetes 130-US Hospitals Dataset (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Records:** ~100,000 hospital admissions
- **Target:** Whether the patient was readmitted within 30 days
- **Features:** Demographics, diagnoses, medications, lab results, and visit types


## ML Pipeline Overview

- **Data Cleaning & Encoding**
- **Feature Engineering** (custom features like total visits, insulin usage, etc.)
- **Modeling** with:
  - Logistic Regression
  - MLP Classifier
  - Random Forest / Gradient Boosting
  - XGBoost (with and without feature selection)
- **Imbalance Handling** via SMOTE
- **Threshold Tuning** using ROC and Precision-Recall curves
<!-- - **Deployment** via a real-time prediction app (Streamlit) -->

---

## 🧩 Project Structure

```text
patient-readmission-prediction/
├── data/                          # CSVs, cleaned data
├── models/                        # Saved trained models (.pkl) (.gitignore)
├── output/                        # Visualizations for report
│   ├── feature_selection
│   ├── train
├── src/                           # Training scripts
│   ├── 3_train_model.py
│   ├── 3_train_model_smote.py
│   ├── 3_train_logistic_mlp_smote.py
│   ├── 3_train_compare_models_engineered.py
│   ├── 3_train_xgboost_smote.py
│   ├── 3_train_xgboost_reduced.py
│   ├── 4_threshold_tuning.py
│   └── app.py                          # Streamlit app for user prediction
│   └── visualize_data.py               # Visualization creation script
├── utils/   
│   ├── 1_load_and_explore.py
│   ├── 2_clean_preprocess.py
│   ├── 2.5_feature_selection.py
│   ├── 2.6_feature_engineering.py      # dataloaders and data processing pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/arrdel/patient-readmission-prediction.git
cd patient-readmission-prediction
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

### 3. Prepare Data (Local Version)

Run:

```bash
python utils/2_clean_preprocess.py
python utils/2.6_feature_engineering.py
```

### 4. Train Models (Locally)

```bash
python src/3_train_model_smote.py
python src/3_train_xgboost_smote.py
python src/3_train_logistic_mlp_smote.py
```

### 5. Tune Threshold

```bash
python src/4_threshold_tuning.py
```


## 📄 License

This project is licensed under the **MIT License**.  
Original dataset available from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
