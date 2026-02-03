# Asthma in Allegheny County, PA: A Machine Learning Project

This repository contains the machine learning pipeline developed for the **Machine Learning Foundations with Python** course at Carnegie Mellon University. This project utilizes environmental, sociodemographic, and healthcare utilization data to classify asthma risk levels in Allegheny County census tracts.

---

## Repository Structure
* **Top-level directory**: Primary Jupyter Notebooks for the ML pipeline.
* **`our-data/`**: Raw and processed datasets (CSV format).
* **`Aequitas Bias Reports/`**: Detailed fairness and bias audits.

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
