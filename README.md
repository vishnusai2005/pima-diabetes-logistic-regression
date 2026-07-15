# pima-diabetes-logistic-regression
# 🩺 Diabetes Risk Prediction — Logistic Regression & KNN Regression

### Comparing an interpretable linear classifier against a tuned, distance-based regressor for diabetes risk scoring

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)](.)

---

## 📖 Overview

Diabetes affects over 500 million people worldwide, and early detection is one of the highest-leverage interventions in preventive healthcare — flagging risk early is far cheaper and safer than treating complications later. This project attacks that problem from **two different modeling angles** on the same clinical dataset:

1. **Logistic Regression** — a fully interpretable linear classifier that outputs a hard diabetic / non-diabetic decision, with every coefficient readable as a change in diabetes log-odds.
2. **KNN Regression (GridSearchCV-tuned)** — instead of classifying, this reframes `Outcome` as a continuous risk score and predicts it using a distance-weighted K-Nearest-Neighbors regressor, hyperparameter-tuned across 80 configurations with 5-fold cross-validation.

The goal isn't just "fit a model, report accuracy." The project walks through the full applied ML workflow a real healthcare-analytics task demands: catching silent data-quality issues (biologically impossible zero values), exploring correlation structure, tuning a non-parametric model properly, and — critically — **comparing two very different modeling philosophies on the same problem** to show where each one wins and where it doesn't.

---

## 🎯 Problem Statement

> Given a patient's diagnostic measurements, predict whether they have diabetes (`Outcome = 1`) or not (`Outcome = 0`).

This project explores that question two ways: as **binary classification** (Logistic Regression) and as **continuous risk-score regression** (KNN Regressor, thresholded at 0.5 for comparison).

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

![Class Distribution](./assets/class_distribution.png)

500 non-diabetic vs. 268 diabetic cases (**65% / 35% split**) — a moderate class imbalance that directly shapes which evaluation metrics actually matter here (accuracy alone would be misleading — see [Results](#-model-1-logistic-regression-classifier)).

---

## 🔍 Data Quality & Cleaning

A closer look at the raw data revealed a classic real-world data-quality trap: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` all contained values of **`0`** — medically impossible for a living patient. These are missing values silently encoded as `0`, not true measurements.

**Zero-value counts found in the raw data:**

| Feature | Zero (missing) count | % of dataset |
|---|---|---|
| `Insulin` | 374 | 48.7% |
| `SkinThickness` | 227 | 29.6% |
| `BloodPressure` | 35 | 4.6% |
| `BMI` | 11 | 1.4% |
| `Glucose` | 5 | 0.7% |

**Fix applied:**
1. Replaced `0` with `NaN` in the five affected columns.
2. Imputed using a distribution-aware strategy:
   - **Median** for `Glucose`, `Insulin`, `SkinThickness` — right-skewed distributions where the median is more robust to outliers.
   - **Mean** for `BMI`, `BloodPressure` — approximately normal distributions.

Training on a column that's silently ~49% corrupted (`Insulin`) without catching this would produce a confidently wrong model — this step mattered more to final quality than any hyperparameter tuning done afterward.

---

## 🔬 Exploratory Data Analysis

- **Univariate analysis** — boxplots and histograms across all 8 features to inspect spread, skew, and outliers.
- **Bivariate analysis** — strip plots of each feature against `Outcome` to visually surface separation between classes.
- **Correlation analysis** — a full feature correlation heatmap to quantify linear relationships with the target and flag multicollinearity risk.

![Correlation Heatmap](./assets/correlation_heatmap.png)

**Top predictors of diabetes outcome (by absolute correlation with target):**

| Rank | Feature | Correlation with `Outcome` |
|---|---|---|
| 1 | `Glucose` | **0.49** |
| 2 | `BMI` | 0.31 |
| 3 | `Age` | 0.24 |
| 4 | `Pregnancies` | 0.22 |
| 5 | `SkinThickness` | 0.21 |

This aligns with clinical intuition — glucose tolerance is the single strongest signal, which is expected since diabetes is fundamentally a disorder of blood glucose regulation.

---

## ⚙️ Methodology

```
Raw Data (768 × 9)
      │
      ▼
Zero-Value Detection → NaN → Median / Mean Imputation
      │
      ▼
Exploratory Data Analysis (Univariate + Bivariate + Correlation)
      │
      ▼
Train / Test Split (80 / 20, random_state = 42)
      │
      ├──────────────────────────┬───────────────────────────────┐
      ▼                          ▼                                 
Logistic Regression        KNN Regressor + GridSearchCV            
(liblinear solver)         (80 candidates × 5-fold CV = 400 fits)   
      │                          │                                 
      ▼                          ▼                                 
Classification Metrics     Regression Metrics (MSE / RMSE / R²)     
(Accuracy, Precision,      + thresholded-classifier comparison      
 Recall, F1, ROC-AUC)                                               
```

---

## 📊 Model 1: Logistic Regression (Classifier)

**Model:** `sklearn.linear_model.LogisticRegression(solver='liblinear')` — chosen for its efficiency on small-to-medium datasets, native L1/L2 support, and — most importantly for a healthcare context — full interpretability.

Evaluated on a held-out test set (154 patients, unseen during training):

| Metric | Score |
|---|---|
| **Accuracy** | 77.9% |
| **Precision** | 73.3% |
| **Recall (Sensitivity)** | 60.0% |
| **F1-Score** | 0.66 |
| **ROC-AUC** | **0.815** |
| **Log Loss** | 0.497 |

<p float="left">
  <img src="./assets/confusion_matrix.png" width="45%" />
  <img src="./assets/roc_curve.png" width="45%" />
</p>

**Reading the results:**
- An **ROC-AUC of 0.815** means the model correctly ranks a random diabetic patient above a random non-diabetic patient ~81.5% of the time — strong discriminative power for a linear model on clinical data.
- **Recall (60%)** trails precision (73%) — the model misses more true diabetic cases than it falsely flags. In a screening context, false negatives are more costly than false positives, so this is called out explicitly under [Future Enhancements](#-future-enhancements).
- With a 65/35 class split, a model that *always* predicts "no diabetes" scores 65.1% accuracy while being clinically useless — which is exactly why ROC-AUC and recall carry more weight here than raw accuracy.

**Interpretability:** every coefficient has a direct, explainable relationship to diabetes log-odds. `DiabetesPedigreeFunction` (0.397) and `Pregnancies`/`BMI` (~0.071 each) carry the largest positive weights — critical for a healthcare application where a clinician needs to trust *why* a model made a prediction, not just *that* it made one.

---

## 📈 Model 2: KNN Regression (GridSearchCV-Tuned)

Rather than treating `Outcome` purely as a class label, this branch of the project explores it as a **continuous risk score**, using `KNeighborsRegressor` to predict a value between 0 and 1 based on the (weighted) average outcome of a patient's nearest neighbors in feature space — no parametric decision boundary, no assumption of linearity.

**Hyperparameter search**, tuned via `GridSearchCV` (5-fold CV, scored on negative MSE):

| Hyperparameter | Values searched |
|---|---|
| `n_neighbors` | 1 – 20 |
| `weights` | `uniform`, `distance` |
| `p` (Minkowski power) | 1 (Manhattan), 2 (Euclidean) |

That's **80 candidate configurations × 5 folds = 400 model fits** to find the best-performing neighbor count, distance metric, and weighting scheme.

![KNN GridSearchCV Tuning](./assets/knn_grid_search.png)

**Best configuration found:** `n_neighbors=20`, `weights='distance'`, `p=1` (Manhattan distance)

| Metric | Score |
|---|---|
| **Best 5-Fold CV MSE** | 0.162 |
| **Test MSE** | 0.184 |
| **Test RMSE** | 0.429 |
| **Test R²** | 0.197 |
| Default KNN (k=5, uniform, Euclidean) Test MSE | 0.217 |

Tuning cut test-set MSE by roughly **15%** versus an untuned default KNN — and the search consistently favored **distance-weighting** and **more neighbors (k=20)**, meaning the signal in this dataset benefits from smoothing over a wider, similarity-weighted neighborhood rather than a small, evenly-weighted one.

![KNN Predicted vs Actual](./assets/knn_predicted_vs_actual.png)

**Repurposed as a classifier** (thresholding the regressor's output at 0.5), for direct comparison with Model 1:

| Metric | KNN Regression (as classifier) | Logistic Regression |
|---|---|---|
| Accuracy | 73.4% | **77.9%** |
| ROC-AUC | 0.773 | **0.815** |

---

## 🥊 Model Comparison — Which One Actually Wins?

| | Logistic Regression | KNN Regression (tuned) |
|---|---|---|
| **Task framing** | Binary classification | Continuous risk regression |
| **Best configuration** | `liblinear` solver | k=20, Manhattan distance, distance-weighted |
| **Tuning effort** | Single fit, default hyperparameters | GridSearchCV, 400 fits across 80 configs |
| **Test performance** | 77.9% accuracy / 0.815 AUC | MSE 0.184 / R² 0.197 (0.773 AUC as classifier) |
| **Interpretability** | High — every coefficient is explainable | Low — predictions come from neighbor geometry, not explainable weights |
| **Sensitive to feature scale?** | No | **Yes** — and this run does *not* scale features first (see note below) |

**Takeaway:** despite 400x more tuning effort, the tuned KNN regressor still trails the untuned Logistic Regression on ranking quality (AUC) and raw accuracy once thresholded. That's a genuinely useful result — it shows that more hyperparameter search doesn't automatically beat a well-suited, simpler model, especially on a small (768-row), mostly-linear-signal tabular dataset like this one.

> **Methodological note:** `KNeighborsRegressor` is distance-based, and this run does **not** scale the features first. Since `Insulin` (0–846) and `Glucose` (0–199) live on a completely different scale from `DiabetesPedigreeFunction` (0.08–2.42), the raw Euclidean/Manhattan distances used by KNN are almost certainly dominated by the large-magnitude features. Feature scaling (`StandardScaler`) before KNN is the single highest-leverage next step — see [Future Enhancements](#-future-enhancements).

---

## 💡 Key Insights

- **Glucose is king.** It's the single most predictive feature by a wide margin — consistent with the biological mechanism of diabetes.
- **Data quality > model choice.** The silent zero-encoding issue affected up to 49% of rows in `Insulin` alone; catching and fixing this mattered more to final performance than any amount of hyperparameter tuning.
- **A simple, well-matched model beat a heavily-tuned one.** Logistic Regression — with zero hyperparameter search — outperformed a 400-fit GridSearchCV-tuned KNN Regressor on both accuracy and ROC-AUC. Tuning effort isn't a substitute for choosing a model whose assumptions actually fit the data.
- **Unscaled KNN is leaving performance on the table.** Because features weren't standardized before the distance-based KNN step, its results likely understate what KNN could actually achieve here — a clear, specific next experiment rather than a vague "try harder."
- **Accuracy alone is misleading on imbalanced data.** With a 65/35 split, an "always predict no-diabetes" model would score 65.1% accuracy while being clinically useless — which is why ROC-AUC and recall are reported alongside it throughout.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Modeling | scikit-learn — `LogisticRegression`, `KNeighborsRegressor` |
| Model Selection | `GridSearchCV`, `train_test_split` (5-fold CV, 400 fits) |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss, MSE, RMSE, R² |
| Environment | Jupyter Notebook |

---

## 📁 Project Structure

```
pima-diabetes-logistic-regression/
├── Logistic_Regression_diabetics.ipynb   # Full analysis & modeling notebook
├── diabetes.csv                          # Pima Indians Diabetes dataset
├── assets/                               # Charts used in this README
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── knn_grid_search.png
│   └── knn_predicted_vs_actual.png
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/vishnusai2005/pima-diabetes-logistic-regression.git
cd pima-diabetes-logistic-regression

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch the notebook
jupyter notebook Logistic_Regression_diabetics.ipynb
```

---

## 🔮 Future Enhancements

- [ ] **Scale features before KNN** (`StandardScaler`) — the highest-leverage fix, since raw feature magnitudes currently dominate the distance calculation
- [ ] Address class imbalance with `class_weight='balanced'` or SMOTE to improve recall on the diabetic class
- [ ] Replace the single train/test split with **stratified K-Fold cross-validation** for a more robust performance estimate
- [ ] Tune Logistic Regression's regularization strength (`C`) and penalty type via `GridSearchCV`
- [ ] Add a native `KNeighborsClassifier` run for a direct apples-to-apples comparison against the regression-based approach
- [ ] Benchmark against Random Forest, XGBoost, and SVM to quantify the interpretability–performance trade-off
- [ ] Add SHAP-based explainability plots for per-patient prediction breakdowns
- [ ] Package the trained model behind a lightweight FastAPI endpoint for real-time risk scoring

---

## 👤 Author

**Vydhyam Vishnusai**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat-square&logo=github)](https://github.com/vishnusai2005)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vishnusai-vydhyam/)

---

