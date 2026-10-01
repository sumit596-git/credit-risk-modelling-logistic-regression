# Credit Risk Modelling and Scorecard Development Using Logistic Regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nG2al7Foj9rVb658ay1wG2IQxkocK1w-)

A logistic-regression-based credit-risk model designed to predict the probability of credit-card default and transform those predictions into an interpretable numerical credit score.

---

## Interactive Notebook

You can run and interact with the full project directly in Google Colab:
- **Google Colab Notebook:** [Open `ml_regression_project.ipynb` in Google Colab](https://colab.research.google.com/drive/1nG2al7Foj9rVb658ay1wG2IQxkocK1w-)

---

## Project Overview

This project develops a complete credit-risk modelling pipeline using the UCI Default of Credit Card Clients dataset. It follows a structured statistical and machine learning workflow—covering rigorous data quality checks, exploratory data analysis (EDA), logistic regression modeling, performance evaluation, multicollinearity assessment, and credit-scorecard generation with score band validation.

---

## Project Objectives

- **Data Preparation:** Assess and clean the credit-card customer dataset for reliable statistical modeling.
- **Exploratory Data Analysis (EDA):** Investigate demographic profiles, credit limits, repayment statuses, bill amounts, and payment histories to uncover key default drivers.
- **Predictive Modeling:** Train a logistic regression model to estimate the probability of credit-card default using stratified splitting and feature standardization.
- **Model Evaluation:** Measure performance using classification metrics, confusion matrix analysis, ROC curves, and ROC-AUC.
- **Multicollinearity Diagnostics:** Assess multicollinearity among numerical and repayment predictors using Variance Inflation Factors (VIF).
- **Scorecard Development:** Transform predicted default probabilities into structured numerical credit scores using a log-odds scaling function.
- **Scorecard Validation:** Analyze observed default rates across distinct score bands and compute rank-ordering correlations.

---

## Dataset

The analysis uses the **Default of Credit Card Clients** dataset (`UCI_credit_card_risk_dataset`), containing records for **30,000 credit-card customers** with **23 predictor variables**.

* **Target Variable:** `target` (`0` = Non-default, `1` = Default)
* **Dataset Size:** 30,000 observations
* **Class Distribution:** 23,364 non-defaults and 6,636 defaults (approx. 22.12% portfolio default rate)
* **Data Integrity:** Zero missing values identified; minor categorical code anomalies cleaned during preprocessing.

---

## Methodology

1. **Data Quality Assessment:** Audited missing values, duplicate records, variable distributions, and categorical code validity (handling undocumented levels in `EDUCATION` and `MARRIAGE`).
2. **Exploratory Data Analysis:** Examined bivariate relationships between default risk and predictors such as recent repayment status (`PAY_1`), credit limits (`LIMIT_BAL`), and demographic attributes.
3. **Logistic Regression Modeling:** Partitioned the data into an 80% training set and a 20% test set using stratified sampling to preserve class proportions. Categorical variables were dummy-encoded and features scaled appropriately.
4. **Model Evaluation:** Evaluated model discriminatory power on unseen test data using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
5. **Multicollinearity Assessment:** Checked predictor collinearity, noting high collinearity among consecutive monthly bill statements (`BILL_AMT1` through `BILL_AMT6`).
6. **Credit Scorecard Transformation:** Converted predicted default probabilities ($p$) into a project-specific credit score using a log-odds formula:
   $$\text{Score} = 600 - 100\log\left(\frac{p}{1-p}\right)$$
7. **Score Band Validation:** Segmented scores into discrete bands to compare predicted risk against observed historical default rates.

---

## Challenges and Solutions

Several practical and statistical challenges were encountered during the development of this project. The primary challenges and the corresponding analytical approaches used to resolve them are summarized below:

| Challenge | Approach Used |
| :--- | :--- |
| **Class imbalance between default and non-default customers** | Stratified train-test splitting was used, and precision, recall, F1-score and ROC-AUC were reported alongside accuracy. |
| **Numerically coded categorical and ordinal variables** | Categorical demographic variables were dummy encoded, while repayment-status variables were interpreted according to their ordinal meaning. |
| **Right-skewed financial variables** | Distributions were examined using descriptive statistics and visualisations, and predictors were standardised before logistic regression. |
| **Multicollinearity among monthly bill variables** | VIF was calculated to identify predictors with high multicollinearity, particularly the monthly bill amount variables. |
| **Limited identification of actual defaulters at the 0.50 threshold** | Default-class precision and recall were examined separately rather than relying only on overall accuracy. |
| **Converting predicted probabilities into a credit score** | A log-odds-based scoring function was used to transform predicted default probabilities into an interpretable numerical score. |
| **Validating the scorecard** | Score bands were compared with observed default rates, and Spearman correlation was used to assess the monotonic relationship between score and default outcome. |

---

## Model Results

Evaluated on the 20% unseen test dataset:

| Metric | Result |
| :--- | :---: |
| **Accuracy** | 81.22% |
| **Precision (Default)** | 71.71% |
| **Recall (Default)** | 24.68% |
| **F1-score (Default)** | 36.72% |
| **ROC-AUC** | **0.721** |

*The ROC-AUC of 0.721 demonstrates moderate discriminatory power. The lower default-class recall at the standard 0.50 threshold indicates that balancing threshold tuning or cost-sensitive learning could be explored in future iterations.*

---

## Key Modeling Findings

* **Repayment Behavior:** Recent repayment status (`PAY_1`) emerged as the strongest positive predictor of default risk, with delays of 2 or more months showing dramatically higher default rates.
* **Credit Limits:** `LIMIT_BAL` demonstrated a negative association with default odds, indicating lower risk profiles among higher-limit cardholders.
* **Multicollinearity:** Monthly bill amounts (`BILL_AMT1`–`BILL_AMT6`) exhibited substantial pairwise collinearity due to measuring related financial balances across consecutive months.

---

## Credit Scorecard Results

The generated credit scores span from a minimum of **114** to a maximum of **2,570**, with a median of **741** and a mean of **746.21** (middle 50% between 703 and 800).

### Score Band Validation

| Score Band | Customers | Defaults | Observed Default Rate |
| :--- | ---: | ---: | ---: |
| **<500** | 54 | 42 | 77.78% |
| **500–599** | 401 | 284 | 70.82% |
| **600–699** | 957 | 375 | 39.18% |
| **700–799** | 3,076 | 451 | 14.66% |
| **800–899** | 1,228 | 147 | 11.97% |
| **900–999** | 232 | 21 | 9.05% |
| **1000+** | 52 | 5 | 9.62% |

* **Rank-Order Performance:** The Spearman correlation between the credit score and observed default outcome was **-0.3176 ($p < 0.001$)**, confirming that higher scores reliably correspond to lower observed default rates.

---

## Limitations

* Evaluated using a single train-test split rather than cross-validation.
* Default-class recall is constrained at the default 0.50 decision threshold.
* The scoring scale is project-specific and does not map directly to commercial credit bureau scoring models (e.g., FICO).
* Implements a streamlined pipeline rather than full production Weight of Evidence (WoE) and Information Value (IV) binning procedures.

---

## Technologies and Libraries

* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **Core Libraries:** Pandas, NumPy, Scikit-learn, Statsmodels, SciPy, Matplotlib, Seaborn

---

## Project Structure

```text
credit-risk-modelling-logistic-regression/
│
├── ml_regression_project.ipynb
├── credit_risk_data_iitb.csv
└── README.md
