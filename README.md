# Telco Churn Boosting Experiment

## Project Overview

This project documents my early journey into machine learning by
experimenting with several **boosting algorithms** on the **Telco
Customer Churn** dataset.

The goal of the project is to understand how different boosting models
perform, how to tune them, and what insights can be gained from
comparing them.

## Project Structure

    ├── data/
    │   └── telco_churn.csv
    ├── notebooks/
    │   ├── 01_data_preprocessing.ipynb
    │   ├── 02_modeling.ipynb
    │   └── 03_model_interpretation.ipynb
    ├── src/
    │   ├── preprocessing.py
    │   ├── modeling.py
    │   └── utils.py
    ├── readme.md
    └── requirements.txt

------------------------------------------------------------------------

## Methods Used

The experiment focuses on several boosting algorithms: 
- **LightGBM**
- **XGBoost**
- **CatBoost**

Each model is trained and compared based on common classification
metrics.

------------------------------------------------------------------------

## Key Steps

1.  Data Cleaning & Preprocessing
2.  Exploratory Data Analysis
3.  Training Boosting Models
4.  Model Comparison
5.  Extracting Feature Importance & Interpretation

------------------------------------------------------------------------

## Results Summary

There is no overfitting model between CV, train, and test on all boosting models. XGBoost performs better than the rest of models, but still lack of ideal precision score. This model still has several limitations, particularly the large number of false positives it produces in the test data. If we reflect this on a real business scenario, it would imply a significant amount of budget spent on campaigns, promotions, or retention activities to keep customers who were never intending to churn in the first place. However, the model performs reasonably well in identifying customers who actually intend to churn, correctly predicting them as churn cases at a rate of around 76%.

------------------------------------------------------------------------

## Contact

If you have any feedback towards this project, please feel free to contact me on bagaskoroah@gmail.com\
Thank you! :)
