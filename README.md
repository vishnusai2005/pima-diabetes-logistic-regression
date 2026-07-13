# pima-diabetes-logistic-regression
<div align="center">

# 🩺 Diabetes Risk Prediction — Logistic Regression

### Predicting diabetes risk from routine diagnostic measurements using a fully interpretable, clinically-aligned ML pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()

</div>

---

## 📖 Overview

Diabetes affects over 500 million people worldwide, and early detection is one of the highest-leverage interventions in preventive healthcare — it's far cheaper and safer to flag risk early than to treat complications later. This project builds a **Logistic Regression classifier** that predicts whether a patient is likely diabetic using **8 routine diagnostic measurements** (glucose level, BMI, age, blood pressure, etc.) — the kind of data already collected in a standard clinical checkup.

The focus of this project isn't just "fit a model and report accuracy." It walks through the full applied ML workflow a real healthcare-analytics task demands: catching silent data quality issues (biologically impossible zero values), handling class imbalance, drawing insight from correlation structure, and evaluating the model with metrics that actually matter for a medical screening tool (recall, ROC-AUC) — not just raw accuracy.

**Why Logistic Regression?** It's the natural first model for binary medical diagnosis tasks: every coefficient is directly interpretable as the change in diabetes log-odds per unit increase in a feature, which matters enormously in healthcare, where a "black box" prediction is rarely acceptable to a clinician.

---

## 🎯 Problem Statement

> Given a patient's diagnostic measurements, predict whether they have diabetes (`Outcome = 1`) or not (`Outcome = 0`) — a **binary classification** problem.

---

## 📂 Dataset

This project uses the **Pima Indians Diabetes Database**, a well-established benchmark dataset originally compiled by the National Institute of Diabetes and Digestive and Kidney Diseases. All patients are female, of Pima Indian heritage, and at least 21 years old.

| Property | Value |
|---|---|
| Samples | 768 patients |
| Features | 8 diagnostic measurements |
| Target | `Outcome` (0 = No Diabetes, 1 = Diabetes) |
| Missing values (explicit) | None — but see [Data Quality](#-data-quality--cleaning) below |

**Feature dictionary:**

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hr oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | A function scoring likelihood of diabetes based on family history |
| `Age` | Age in years |

**Class distribution:**

<img src="assets/class_distribution.png" width="480"/>

500 non-diabetic vs. 268 diabetic cases (**65% / 35% split**) — a **moderate class imbalance** that directly informed the choice of evaluation metrics (see [Results](#-results)).

---

## 🔍 Data Quality & Cleaning

A closer look at the raw data revealed a classic real-world data quality trap: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` all contained values of **`0`** — which is **medically impossible** for a living patient. These aren't true zeros; they're missing values that were silently encoded as `0`.

**Fix applied:**
1. Replaced `0` with `NaN` in the five affected columns.
2. Imputed using a **distribution-aware strategy**:
   - **Median** for `Glucose`, `Insulin`, `SkinThickness` — right-skewed distributions where the median is more robust to outliers.
   - **Mean** for `BMI`, `BloodPressure` — approximately normal distributions.

This step alone materially changes model quality — training on ~50% silently-corrupted `Insulin` values (374 of 768 rows were `0`) without catching this would have produced a confidently wrong model.

---

## 🔬 Exploratory Data Analysis

- **Univariate analysis** — boxplots and histograms across all 8 features to inspect spread, skew, and outliers.
- **Bivariate analysis** — strip plots of each feature against `Outcome` to visually surface separation between classes.
- **Correlation analysis** — a full feature correlation heatmap to quantify linear relationships with the target and detect multicollinearity risk.

<img src="assets/correlation_heatmap.png" width="600"/>

**Top predictors of diabetes outcome (by correlation with target):**

| Rank | Feature | Correlation with `Outcome` |
|---|---|---|
| 1 | `Glucose` | **0.47** |
| 2 | `BMI` | 0.29 |
| 3 | `Age` | 0.24 |
| 4 | `Pregnancies` | 0.22 |
| 5 | `DiabetesPedigreeFunction` | 0.17 |

This aligns with clinical intuition — glucose tolerance is the single strongest signal, which is expected since diabetes is fundamentally a disorder of blood glucose regulation.

---

## ⚙️ Methodology

```
Raw Data (768 × 9)
      │
      ▼
Zero-Value Detection → NaN → Median/Mean Imputation
      │
      ▼
Exploratory Data Analysis (Univariate + Bivariate + Correlation)
      │
      ▼
Train/Test Split (80/20, stratified by random_state=42)
      │
      ▼
Logistic Regression (liblinear solver)
      │
      ▼
Evaluation: Accuracy · Precision · Recall · F1 · ROC-AUC · Log Loss
```

**Model:** `sklearn.linear_model.LogisticRegression(solver='liblinear')` — chosen for its efficiency on small-to-medium datasets and native support for L1/L2 regularization.

---

## 📊 Results

Evaluated on a held-out test set (154 patients, unseen during training):

| Metric | Score |
|---|---|
| **Accuracy** | 77.9% |
| **Precision** | 73.3% |
| **Recall (Sensitivity)** | 60.0% |
| **F1-Score** | 0.66 |
| **ROC-AUC** | **0.815** |
| **Log Loss** | 0.497 |

<img src="assets/confusion_matrix.png" width="380"/>  <img src="assets/roc_curve.png" width="420"/>

**Reading the results:**
- An **ROC-AUC of 0.815** means the model correctly ranks a random diabetic patient above a random non-diabetic patient ~81.5% of the time — solidly above the 0.50 random baseline, indicating strong discriminative power for a linear model on clinical data.
- **Recall (60%)** is lower than precision (73%) — the model misses more true diabetic cases than it falsely flags. This is a direct consequence of class imbalance (268 vs. 500) and is explicitly called out below as the top target for improvement, since in a screening context, **false negatives are more costly than false positives**.

---

## 💡 Key Insights

- **Glucose is king.** It's the single most predictive feature by a wide margin — consistent with the biological mechanism of diabetes.
- **Data quality > model choice.** The silent zero-encoding issue affected up to 49% of rows in `Insulin` alone; catching and fixing this mattered more to final performance than any hyperparameter tuning would have.
- **Accuracy alone is misleading here.** With a 65/35 class split, a model that *always* predicts "no diabetes" would score ~65% accuracy while being clinically useless. ROC-AUC and recall are the metrics that actually matter for this task.
- **Interpretability is a feature, not a limitation.** Every coefficient in the model has a direct, explainable relationship to diabetes log-odds — critical for any healthcare application where a clinician needs to trust *why* a model made a prediction.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling & Evaluation | scikit-learn |
| Environment | Jupyter Notebook |

---

## 📁 Project Structure

```
diabetes-risk-prediction/
├── Logistic_Regression_diabetics.ipynb   # Full analysis & modeling notebook
├── diabetes.csv                          # Pima Indians Diabetes dataset
├── assets/                               # Charts used in this README
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch the notebook
jupyter notebook Logistic_Regression_diabetics.ipynb
```

---

## 🔮 Future Enhancements

- [ ] Address class imbalance with `class_weight='balanced'` or SMOTE to improve recall on the diabetic class
- [ ] Replace the single train/test split with **stratified K-Fold cross-validation** for a more robust performance estimate
- [ ] Tune regularization strength (`C`) and penalty type via `GridSearchCV`
- [ ] Benchmark against Random Forest, XGBoost, and SVM to quantify the interpretability–performance trade-off
- [ ] Add SHAP-based explainability plots for per-patient prediction breakdowns
- [ ] Package the trained model behind a lightweight FastAPI endpoint for real-time risk scoring

---

## 👤 Author

**Vydhyam Vishnusai**
B.Tech CSE (AI & ML), Mohan Babu University, Tirupati

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat-square&logo=github)](https://github.com/<your-username>)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vishnusai-vydhyam/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
