# 💳 Credit Risk Modelling & Scorecard Development

### Logistic Regression • Credit Risk • Statistical Modelling • Scorecard Development

> A statistically driven credit-risk modelling project that predicts credit-card default probability using logistic regression and transforms model outputs into an interpretable numerical credit score.


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nG2al7Foj9rVb658ay1wG2IQxkocK1w-)

---

## 📌 Project Overview

Credit risk modelling is fundamentally a **probability and ranking problem**: given a customer's observed financial and repayment characteristics, how likely are they to default?

This project develops an end-to-end **logistic-regression-based credit-risk model** using the **UCI Default of Credit Card Clients dataset**.

The project goes beyond simply fitting a classifier. It combines:

- Statistical data-quality assessment
- Exploratory data analysis
- Logistic regression
- Predicted default probabilities
- 5-fold stratified cross-validation
- ROC-AUC evaluation
- Coefficient and odds-ratio interpretation
- Multicollinearity analysis using VIF
- Credit-score transformation
- Score-band validation

The final outcome is an interpretable modelling pipeline connecting **default probability → risk ranking → credit score**.

---

## 🎯 Objectives

The project aims to:

1. Prepare and assess the credit-card dataset for statistical modelling.
2. Explore demographic, financial, billing, payment, and repayment-status variables.
3. Develop a logistic regression model for predicting credit-card default.
4. Evaluate predictive performance using both a held-out test set and 5-fold stratified cross-validation.
5. Interpret model coefficients through odds ratios.
6. Diagnose multicollinearity among predictors.
7. Transform predicted probabilities into an interpretable credit score.
8. Validate the scorecard using observed default rates across score bands.

---

## 📊 Dataset

The analysis uses the **Default of Credit Card Clients** dataset originally provided through the UCI Machine Learning Repository.

| Property | Description |
|---|---|
| Observations | 30,000 |
| Predictors | 23 |
| Response | `target` |
| `target = 0` | Non-default |
| `target = 1` | Default |
| Non-default observations | 23,364 |
| Default observations | 6,636 |
| Default rate | 22.12% |

### Main Variable Groups

**Demographic**
- `SEX`
- `EDUCATION`
- `MARRIAGE`
- `AGE`

**Credit information**
- `LIMIT_BAL`

**Repayment status**
- `PAY_1` – `PAY_6`

**Monthly bill amounts**
- `BILL_AMT1` – `BILL_AMT6`

**Monthly payment amounts**
- `PAY_AMT1` – `PAY_AMT6`

The response variable represents whether the customer defaulted on their credit-card payment in the subsequent month.

---

## 🔬 Methodology

The modelling pipeline follows a structured statistical workflow.

### 1. Data Preparation

- Loaded and cleaned the dataset.
- Corrected column formatting and data types.
- Removed the customer identifier from the predictor set.
- Renamed `PAY_0` to `PAY_1` for consistent sequential naming.
- Dummy-encoded categorical demographic variables.
- Retained repayment-status variables according to their ordered coding.
- Standardised predictors using the training data.

### 2. Data Quality Assessment

The dataset was examined for:

- Missing observations
- Duplicate observation patterns
- Target-class distribution
- Variable ranges
- Categorical and ordinal coding validity

No missing values were identified.

### 3. Exploratory Data Analysis

EDA was used to investigate:

- Distribution of demographic variables
- Credit-limit characteristics
- Repayment behaviour
- Monthly billing and payment patterns
- Default rates across important predictors
- Relationships between customer characteristics and default

### 4. Logistic Regression

A binary logistic regression model was fitted to estimate:


P(Y=1|X)


The model provides an estimated probability of default for each customer.

### 5. 5-Fold Stratified Cross-Validation

To assess the stability of predictive performance, **5-fold stratified cross-validation** was performed on the training data.

Stratification preserves the approximate default/non-default proportion across folds.

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The mean and standard deviation across folds were reported.

### 6. Model Evaluation

Performance was assessed using:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC curve
- ROC-AUC

Because default is the minority class, accuracy was not treated as the sole measure of model performance.

### 7. Model Interpretation

The fitted logistic regression coefficients were extracted and converted into odds ratios:

An odds ratio greater than 1 indicates higher estimated odds of default associated with an increase in the corresponding predictor, while an odds ratio below 1 indicates lower estimated odds, conditional on the other variables in the model.

This allows the direction and magnitude of associations between predictors and default odds to be interpreted.

### 8. Multicollinearity

Variance Inflation Factors (VIF) were examined to identify potential multicollinearity among predictors, particularly the monthly bill variables.

### 9. Credit Scorecard

Predicted probabilities of default were transformed into numerical credit scores using a log-odds-based scoring relationship.

The resulting scores provide an interpretable risk-ranking mechanism in which lower predicted default risk corresponds to higher credit scores.

### 10. Scorecard Validation

Customers were grouped into score bands and the observed default rate was examined across the bands.

A monotonic relationship between score and observed default behaviour was also assessed using Spearman correlation.

---

## 📈 Model Performance

### 5-Fold Cross-Validation

| Metric | Mean CV Score | Std. Dev. |
|---|---:|---:|
| Accuracy | **0.8111** | 0.0044 |
| Precision | **0.7157** | 0.0222 |
| Recall | **0.2425** | 0.0166 |
| F1-score | **0.3621** | 0.0210 |
| ROC-AUC | **0.7249** | 0.0034 |

The relatively small standard deviation of ROC-AUC indicates that the model's discriminatory performance is reasonably stable across the five validation folds.

The relatively low recall indicates that the standard 0.50 classification threshold identifies only a portion of the actual default cases. This highlights the importance of considering the classification threshold and the business cost associated with different types of classification errors.

### Held-Out Test Set

| Metric | Test Result |
|---|---:|
| Accuracy | **0.81** |
| Precision | **0.72** |
| Recall | **0.25** |
| F1-score | **0.37** |
| ROC-AUC | **0.72** |

The test-set ROC-AUC of **0.72** is close to the cross-validation mean of **0.73**, providing consistent evidence of the model's discriminatory performance across validation and held-out data.

> **Important:** Cross-validation is used here to assess performance stability; it is not treated as a method for increasing the model's predictive performance.

---

## 🔎 Key Findings

### Repayment Behaviour

Repayment-status variables, particularly recent repayment behaviour such as `PAY_1`, show strong associations with default risk.

Customers exhibiting more severe recent repayment delays have substantially higher observed default rates.

### Credit Limit

`LIMIT_BAL` shows a negative association with default odds in the fitted model, indicating lower model-estimated default odds for customers with higher credit limits, conditional on the other variables in the model.

### Multicollinearity

Monthly bill variables exhibit substantial correlation because they represent related financial balances measured over consecutive months.

This is reflected in elevated VIF values for some bill-related predictors.

### Classification Threshold

At the conventional 0.50 threshold, the model has relatively low recall for the default class.

This demonstrates an important distinction between:

**probability estimation and binary classification.**

The logistic model produces a continuous probability of default, while converting that probability into a binary decision requires selecting a threshold.

---

## 💳 Credit Scorecard

The logistic model's predicted default probabilities are transformed into a numerical credit score using the relationship between **probability of default and log-odds**.

Conceptually:

```text
Customer characteristics
          ↓
Logistic Regression
          ↓
Probability of Default
          ↓
Log-Odds Transformation
          ↓
Credit Score
          ↓
Risk Ranking / Score Bands
