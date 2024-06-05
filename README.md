# TensorFlow and Machine Learning — Liver Disease Prediction

Clinical data classification for liver disease prediction using the Indian Liver Patient Dataset (ILPD). Built during an industrial internship at VIT Chennai.

## Dataset

**Indian Liver Patient Dataset (ILPD)**
- 583 patient records, 10 clinical features + 1 binary target (liver disease: yes/no)
- Features: Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, Alamine Aminotransferase, Aspartate Aminotransferase, Total Proteins, Albumin, Albumin/Globulin Ratio
- Class distribution: 416 positive (71.4%), 167 negative (28.6%) — imbalanced
- Gender split: 441 Male (75.6%), 142 Female (24.4%)

## Preprocessing

- 4 missing values in Albumin/Globulin Ratio → filled with column mean
- High-correlation features removed: Direct Bilirubin and Aspartate Aminotransferase dropped
- Final feature set: 9 features after encoding
- SMOTE applied to balance the class imbalance before training
- StandardScaler normalisation
- Train/test split: 90% test, 10% train (unusual but intentional — tests generalisation)

## Results

| Model | Test Accuracy | AUC-ROC |
|---|---|---|
| **Logistic Regression** | **73.83%** | **0.738** |
| Random Forest (130 trees) | 71.03% | — |
| Decision Tree | 60.35% | 0.6036 |
| KNN (k=2) | — | — |

**Logistic Regression** was the best performer overall:
- Precision: 0.82, Recall: 0.62, F1: 0.70
- Confusion matrix: TN=319, FP=50, FN=146, TP=234
- CV Score (5-fold): 0.66
- Config: C=0.0001, solver=liblinear, max_iter=1000

Decision Tree hit 100% training accuracy (severe overfitting) but only 60.35% test accuracy — a good illustration of why cross-validation matters.

The trained Logistic Regression model is saved as `executables/model_reg.pkl` and served via `executables/model_deploy.py`.

## Repository Contents

**Core**

| File | Description |
|---|---|
| `executables/ml_liver (7).ipynb` | Full pipeline — EDA, preprocessing, SMOTE, model training, evaluation |
| `executables/model_deploy.py` | TensorFlow Serving deployment script |
| `executables/model_reg.pkl` | Serialised Logistic Regression model |
| `executables/HOME_m.html` | Web interface for model predictions |
| `executables/index1.html` | Alternative web interface |

**Reports**

| File | Description |
|---|---|
| `Industrial Internship Report 21BCE1950.pdf` | Full internship report |
| `ML_DA2_Prediction of heart and liver disease ml.pdf` | ML data analysis report |

## Stack

Python · scikit-learn · pandas · NumPy · matplotlib · seaborn · joblib · SMOTE (imblearn) · Jupyter Notebook