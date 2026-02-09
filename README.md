# Clinical Risk Prediction with Machine Learning (diabetes-risk-classification)
Builds end-to-end pipeline to clean and preprocess clinical data, engineer features, and train classification models. Compares logistic regression and random forest models using cross-validation and ROC-AUC. Automates evaluation and visualization workflows to ensure reproducibility and interpretability of results.

## Dataset
Pima Indians Diabetes Database (UCI ML Repository).
Automatically downloaded via kagglehub when the script runs.

## Objective
Develop reproducible predictive models that classify diabetes diagnosis from structured health indicators such as glucose, BMI, insulin, and age.

## Methods
- Data preprocessing and feature scaling
- Logistic Regression
- Random Forest
- Cross-validation
- ROC-AUC evaluation
- Automated model benchmarking

## Results
Both models achieved strong ROC-AUC performance, with Random Forest outperforming baseline logistic regression.

## Tech Stack
Python, pandas, scikit-learn, matplotlib

## How to Run
pip install -r requirements.txt  
python src/model_pipeline.py

## Author
Priya Jani

