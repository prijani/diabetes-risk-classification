"""
Clinical Risk Prediction using Machine Learning
Predict diabetes diagnosis from clinical biomarkers using sklearn.

Author: Priya Jani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay

RANDOM_STATE = 42

def load_data():
    # automatically download dataset
    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    
    csv_path = os.path.join(path, "diabetes.csv")
    df = pd.read_csv(csv_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y

def build_models():
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": forest
    }

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"\n{name}")
    print("ROC-AUC:", round(auc, 3))
    print(classification_report(y_test, model.predict(X_test)))

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(name)

    # Determine results folder path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    results_dir = os.path.join(script_dir, "..", "results")  
    os.makedirs(results_dir, exist_ok=True)

    plt.savefig(os.path.join(results_dir, f"{name}_roc_curve.png"))
    plt.close()

    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print("Cross-val AUC:", round(cv.mean(), 3))

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = build_models()

    for name, model in models.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
