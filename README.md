# Heart Disease Prediction

> **Jedha Bootcamp — Data Essentials, Final Project (2026)** — [View Certificate (PDF)](Certificate/Jedha-DataEssentials.pdf)

<p align="center">
  <img src="Certificate/Jedha-DataEssentials.png" alt="Jedha Bootcamp Data Essentials Certificate" width="700">
</p>

A machine learning project to predict cardiovascular disease from clinical patient data. Built as the final project for Jedha Bootcamp — Data Essentials.

## The Problem

Cardiovascular diseases are the **#1 cause of death globally**, killing ~17.9 million people each year (WHO). Early detection can save lives, but traditional diagnosis relies on expensive tests and specialist availability. Can we build a model that helps doctors screen patients faster?

## Dataset

918 patients with 12 clinical features from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease):

- **Demographics:** Age, Sex
- **Symptoms:** Chest pain type, exercise-induced angina
- **Clinical measures:** Resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, ST slope, oldpeak
- **Target:** HeartDisease (0 = healthy, 1 = disease)

## Approach

### 1. Data Cleaning
- Detected **172 zero values** in Cholesterol and 1 in RestingBP (physiologically impossible)
- Replaced with NaN and imputed using **median strategy** via `SimpleImputer`

### 2. Exploratory Data Analysis (EDA)
- Target distribution analysis (balanced: 55% disease / 45% healthy)
- Sex and age breakdown vs heart disease prevalence
- Boxplots of numerical features by disease status
- Chest pain type analysis (asymptomatic = highest risk)
- Correlation heatmap

### 3. Preprocessing Pipeline
- `ColumnTransformer` with:
  - **Numerical:** SimpleImputer (median) + StandardScaler
  - **Categorical:** OneHotEncoder (drop='first')
- 80/20 stratified train/test split

### 4. Models Trained

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | ~86% | 0.92 |
| Random Forest | ~87% | 0.93 |
| Decision Tree | ~82% | 0.82 |

### 5. Evaluation
- Confusion matrices for all 3 models
- ROC curves comparison
- Feature importance analysis (coefficients + tree-based)
- Focus on **minimizing false negatives** (missing a sick patient is worse than a false alarm)

## Key Findings

- **Top predictors:** ST_Slope, ChestPainType (Asymptomatic), Oldpeak, MaxHR, Sex
- **Random Forest** performs best overall with 87% accuracy and 0.93 AUC
- **Logistic Regression** is nearly as good and more interpretable — better for clinical deployment
- The model could serve as a **first-pass screening tool** for general practitioners

## Tech Stack

- Python 3, Jupyter Notebook
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (LogisticRegression, RandomForest, DecisionTree, ColumnTransformer, Pipeline)

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
jupyter notebook heart_disease_prediction.ipynb
```

## Project Context

- **Program:** Jedha Bootcamp — Data Essentials (Final Project)
- **Team:** Flavien, Gabriel, Saïd
- **Presentation:** 10-minute oral defense with jury
- **Compliance:** GDPR considerations documented in the notebook

## What I Learned

- Building an **end-to-end ML pipeline** from raw data to model evaluation
- The importance of **data cleaning** before modeling (zero values in medical data)
- How to choose between models based on **business context** (false negatives vs false positives)
- Working as a team with Git and shared notebooks
