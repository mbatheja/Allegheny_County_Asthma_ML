# Asthma in Allegheny County, PA: A Machine Learning Project

This repository contains the machine learning pipeline developed for the **Machine Learning Foundations with Python** course at Carnegie Mellon University. This project utilizes environmental, sociodemographic, and healthcare utilization data to classify asthma risk levels in Allegheny County census tracts.

---

## Repository Structure

Based on the project's current architecture, the files are organized as follows:

```text
├── Aequitas Bias Reports/     # Detailed fairness and bias audits
├── Archive/                  # Legacy notebooks (variable_threshold, etc.)
├── our-data/                 # Cleaned and processed CSV datasets
├── AsthmaUtilization_Cleaning.ipynb
├── Emissions_Cleaning.ipynb
├── Feature selection.ipynb
├── Models.ipynb
├── README.md
└── requirements.txt

### Main Code Files
1. **`Emissions_Cleaning.ipynb`**: Preliminary cleaning of EPA and local emissions data.
2. **`AsthmaUtilization_Cleaning.ipynb`**: Core data wrangling hub, joining utilization and census data.
3. **`Feature selection.ipynb`**: Explores relevant features using EDA, PCA, VIF, and variance thresholding to isolate the final set of 19 features.
4. **`Models.ipynb`**: Modeling suite featuring Logistic Regression, Random Forest, and XGBoost with class imbalance handling.

---

## Executive Summary
Asthma affects approximately **8% of the US population**, but its impact is not felt equally. In Allegheny County, pediatric asthma rates exceed national averages. Our objective was to develop a binary classification task to identify census tracts with a **High Asthma Diagnosis Rate (>10%)**. 

With a target class distribution of **25.9% (High Risk)**, we leveraged advanced sampling and ensemble modeling to provide a comprehensive picture of asthma management needs. The **XGBoost model** emerged as the top performer, achieving a **0.95 AUC-ROC** and a recall rate of **0.89**, demonstrating high efficacy in identifying vulnerable communities.

---

## Key Project Features & Methodology

This project utilized a multi-stage analytical pipeline to move from 130+ raw variables down to a refined, 19-variable predictive model.

### 1. Dimensionality Reduction (PCA)
To handle the high dimensionality of environmental and sociodemographic data, we employed **Principal Component Analysis (PCA)**. This helped us identify the "latent" structures in Allegheny County’s data—such as "Urban Industrial Stress"—without overfitting the model to specific, redundant noise.


### 2. Multi-collinearity Mitigation (VIF)
Public health data is highly collinear (e.g., poverty levels correlating with housing age). We calculated the **Variance Inflation Factor (VIF)** to ensure model stability. We iteratively removed features with high VIF scores to ensure that each feature in the final set provided independent, unique information.

### 3. Missingness & Variance Thresholding
* **Missingness Filter**: Removed columns where more than **40% of the data was missing or zero**, ensuring a strong signal-to-noise ratio.
* **Variance Thresholding**: Eliminated "quasi-constant" features that rarely changed across tracts to focus the model on variables that actually drive health disparities.

### 4. Advanced Sampling for Class Imbalance
Since the "High Risk" class is the minority, we implemented:
* **SMOTE (Synthetic Minority Over-sampling Technique)**: To teach the model subtle patterns in high-risk areas.
* **Random Undersampling (RUS)**: To balance the training set, specifically improving **Recall** to ensure no high-risk community is left behind.

---

## Installation and Requirements
Ensure you have Python 3.9+ installed.

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/asthma-allegheny-ml.git](https://github.com/your-username/asthma-allegheny-ml.git)
cd asthma-allegheny-ml
```
###2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn ipython
```
## Code Implementation Logic
The modeling logic in Models.ipynb follows a rigorous pipeline:

<li> Temporal Cross-Validation: Data is sorted by YearOfContactDate using TimeSeriesSplit (4 folds) to prevent temporal data leakage and mimic real-world predictive deployment. </li>

<li> Preprocessing: Features are scaled using StandardScaler within a ColumnTransformer. </li>

<li> Hyperparameter Tuning: GridSearchCV was used to optimize PR-AUC (Precision-Recall Area Under Curve). For XGBoost, this involved tuning gamma, learning_rate, max_depth, and reg_alpha. </li>

<li> Threshold Optimization: We explored custom classification thresholds (e.g., 0.3 and 0.4) to prioritize Recall over Accuracy, as the public health cost of missing a high-risk tract is significant. </li>

## Feature Importance
Our models consistently identified the following as the most influential drivers of asthma risk:

<li> NumberED_VisitsAge0to17Per100: The strongest proxy for uncontrolled asthma. </li>

<li>Carbon Monoxide: A primary environmental predictor. </li>

<li> Age0to17PopEst: Highlighting the demographic vulnerability of the pediatric population. </li>

## Policy Recommendations
<li> Localized Funding: Direct resources to the PA Asthma Control Program for identified "High Risk" tracts. </li>
<li> Proactive Regulation: Use pollutants like Carbon Monoxide as triggers for proactive emissions inspections. </li>
<li>Equity Task Force: Specifically address racial disparities highlighted by the positive correlation between the Black/African American population and high diagnosis rates. </li>
