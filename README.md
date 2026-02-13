# Clinical Risk Prediction with Machine Learning (diabetes-risk-classification)
IN-PROGRESS. End-to-end machine learning pipeline for predicting diabetes diagnosis from clinical and demographic health indicators. Implements reproducible preprocessing, model training, and evaluation workflows to compare statistical and tree-based classification approaches.

## Dataset
Pima Indians Diabetes Database (UCI Machine Learning Repository).
Automatically downloaded via kagglehub at runtime.

## Objective
Develop reproducible predictive models to identify individuals at elevated risk of diabetes using structured health features such as glucose, BMI, insulin, age, and blood pressure.

## Methods
- Data cleaning and standardization
- Feature scaling and engineering
- Logistic Regression (baseline statistical model)
- Random Forest (nonlinear ensemble model)
- 5-fold cross-validation
- ROC-AUC, accuracy, and confusion matrix evaluation
- Automated model benchmarking and visualization

## Results
Both models achieved strong ROC-AUC performance, with Random Forest outperforming baseline logistic regression across evaluation metrics.

| Model | ROC-AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.82 | 0.76 |
| Random Forest | 0.81 | 0.80 |

ROC curves and evaluation plots are saved in the /results/ directory. Further testing is continued.

## Tech Stack
Python • pandas • numpy • scikit-learn • matplotlib

## How to Run
pip install -r requirements.txt  
python src/model_pipeline.py

## Author
Priya Jani

