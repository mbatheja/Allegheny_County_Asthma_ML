# Asthma in Allegheny County, PA: A Machine Learning Project

This repository contains the code for a ML project completed as part of the *Machine Learning Foundations with Python* course at Carnegie Mellon University. For the accompanying report, see [here](https://docs.google.com/document/d/1hkgx5kfAXBgw1ZC6xaPD2KtcRmtCuz5S7xuiRFFBuqY/edit?usp=sharing).

The repository is structured with code files kept in the top-level directory and datasets kept in the second-level `our-data/` directory.

The main code files are:

* `Emissions_Cleaning.ipynb`: cleans the emissions data preliminarily before being joined with the other data files
* `AsthmaUtilization_Cleaning.ipynb`: handles the majority of the data cleaning, wrangling, and preprocessing
* `Feature selection.ipynb`: explores relevant features using EDA, PCA and variance thresholding to isolate the final set of 18 fetures
* `Models.ipynb`: contains the ML models used to predict asthma risk in Allegheny County census tracts

Archive consists of additional code files that the team played with while developing the code but are not relevant to the main project:

* `Gridsearch_and_Smote.ipynb`
* `variable_threshold.ipynb`
* `classification.ipynb`

Similarly, the `our-data/` folder contains all datasets that were explored and produced in the development of this project. See the report or `AsthmaUtilization_Cleaning.ipynb` to understand the final files that were used.

---

### Executive Summary

Asthma is a common chronic illness in the United States, affecting approximately 8% of the total population. Despite relatively stable prevalence rates throughout the 21st century, both direct and indirect costs of asthma have skyrocketed and will continue to do so. These costs are alleviated via proper asthma management, including avoiding triggers and adhering to prescription regimens, but many obstacles make following these guidelines more difficult than it should be. Often, these obstacles lie outside the control of the individual such as pollution and medication costs, highlighting the need for larger-scale policy initiatives.

The current project builds on prior research to predict areas of higher asthma risk using machine learning models. We focus on Allegheny County, where the pediatric asthma rate is higher than the national average, and compile utilization, environmental, and sociodemographic data to classify census tracts as having high (>10%) or low asthma prevalence rates. By examining several factors at once, we provide a more comprehensive picture of asthma management needs across the county.

We leveraged common machine learning techniques, including variance thresholding, temporal cross-validation, and hyperparameter grid search to develop three predictive models. Logistic regression exhibited subpar performance (recall = 0.62), which we attributed to the restrictive linear nature of the model. The random forest model initially had mediocre recall (0.71), but random undersampling improved this metric to 0.92, though precision dropped to 0.46. The XGBoost model performed well, achieving a recall rate of 0.89 while maintaining a precision rate of 0.90 (AUC = 0.95). Throughout the project, we paid particular attention to instances of bias and fairness in our models, with specific cases highlighted in the report.

Our results inform three policy recommendations for Allegheny County: (1) additional funding should be directed into local initiatives via the PA Asthma Control Program; (2) emissions inspections and regulatory enforcement should take place in a proactive, rather than reactive, manner; and (3) a task force should be assembled to focus specifically on racial disparities in asthma exacerbators and to devise solutions to alleviate these inequities. Shifting a part of the asthma management burden onto policymakers will ultimately increase efficiency and mitigate costs associated with untreated or poorly treated asthma.

Limitations include a lack of temporal breadth and granularity of our data, which can be attributed to the limited availability of healthcare utilization data in the public domain. Most importantly, this project spanned only four weeks, whereas typical ML policy endeavors require much more time to ensure all programmatic and technical elements are present and well-validated. Thus, the results presented should be taken with an enormous grain of salt. Future directions are described and aim to address many of these shortcomings. 
