"""
Phishing Website Detection - Main Training, Cross-Validation, Threshold Tuning, PCA & Test Evaluation
Author: Aditya Bhatt (URN: 2481184)
Course: Applied Machine Learning (Semester 5)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


def load_and_split_data(csv_path: str = "data/uci_phishing.csv"):
    """Load UCI Phishing dataset and create a stratified 80/20 train-test split."""
    df = pd.read_csv(csv_path)
    target_col = "class" if "class" in df.columns else "Result"

    X = df.drop(columns=[target_col])
    # Map -1 (phishing) -> 1, +1 (legitimate) -> 0
    y = df[target_col].map({-1: 1, 1: 0}) if set(df[target_col].unique()) == {-1, 1} else df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(feature_cols):
    """Leak-free preprocessing pipeline fitted strictly on training folds."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, list(feature_cols))]
    )


def run_cv_and_tuning(X_train, y_train):
    """Evaluate Dummy baselines and tune the five syllabus models using 5-fold StratifiedKFold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    preprocessor = build_preprocessor(X_train.columns)

    scoring = {
        "recall": make_scorer(recall_score, pos_label=1),
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    # Dummy baselines
    for strat in ["most_frequent", "stratified"]:
        dummy_pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", DummyClassifier(strategy=strat, random_state=RANDOM_SEED)),
        ])
        res = cross_validate(dummy_pipe, X_train, y_train, cv=cv, scoring=scoring)
        print(
            f"Dummy ({strat}): Recall={res['test_recall'].mean():.4f} +/- {res['test_recall'].std():.4f}, "
            f"Accuracy={res['test_accuracy'].mean():.4f}"
        )

    # Syllabus models + GridSearchCV parameter grids
    model_specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {
                "clf__max_depth": [8, 12, 18, None],
                "clf__min_samples_split": [2, 10],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            {
                "clf__n_estimators": [100, 150],
                "clf__max_depth": [15, None],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
            {"clf__C": [1.0, 5.0, 10.0], "clf__gamma": ["scale", 0.05, 0.1]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [3, 5, 7, 9],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
    }

    best_estimators = {}
    for name, (estimator, param_grid) in model_specs.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=make_scorer(recall_score, pos_label=1),
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best_estimators[name] = grid.best_estimator_

        cv_res = cross_validate(grid.best_estimator_, X_train, y_train, cv=cv, scoring=scoring)
        print(
            f"{name} (Tuned): Recall={cv_res['test_recall'].mean():.4f} +/- {cv_res['test_recall'].std():.4f} | "
            f"Precision={cv_res['test_precision'].mean():.4f} +/- {cv_res['test_precision'].std():.4f} | "
            f"ROC-AUC={cv_res['test_roc_auc'].mean():.4f} | Params={grid.best_params_}"
        )

    return best_estimators, cv


def select_decision_threshold(rf_pipeline, X_train, y_train, cv):
    """Out-of-fold threshold sweep using 10:1 asymmetric cost (Cost = 10*FN + FP)."""
    oof_probs = cross_val_predict(
        rf_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    print("\n--- Out-of-Fold Threshold Sweep (10:1 Cost Ratio) ---")
    for thresh in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        preds = (oof_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
        cost = 10 * fn + fp
        rec = recall_score(y_train, preds)
        prec = precision_score(y_train, preds)
        f1 = f1_score(y_train, preds)
        print(
            f"Threshold={thresh:.2f} | Recall={rec:.4f} | Precision={prec:.4f} | "
            f"F1={f1:.4f} | FN={fn} | FP={fp} | Cost={cost}"
        )

    return 0.35


def run_pca_experiment(X_train, y_train, cv):
    """Retain 95% variance via PCA inside a leak-free Pipeline and compare Random Forest CV recall."""
    pca_pipe = Pipeline([
        ("prep", build_preprocessor(X_train.columns)),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=RANDOM_SEED)),
    ])
    pca_pipe.fit(X_train, y_train)
    n_comp = pca_pipe.named_steps["pca"].n_components_
    var_ratio = pca_pipe.named_steps["pca"].explained_variance_ratio_.sum()

    rec_scores = cross_val_predict(pca_pipe, X_train, y_train, cv=cv, method="predict")
    print(f"\nPCA (95% variance): {n_comp} components ({var_ratio:.2%} variance), OOF Recall={recall_score(y_train, rec_scores):.4f}")
    return pca_pipe


def evaluate_once_on_test(best_rf, X_train, y_train, X_test, y_test, threshold: float = 0.35):
    """Single final evaluation on the untouched held-out test set."""
    best_rf.fit(X_train, y_train)
    test_probs = best_rf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    print("\n--- Final Held-Out Test Set Evaluation (Threshold = 0.35) ---")
    print(f"Test Recall:    {recall_score(y_test, test_preds):.4f}")
    print(f"Test Precision: {precision_score(y_test, test_preds):.4f}")
    print(f"Test F1-Score:  {f1_score(y_test, test_preds):.4f}")
    print(f"Test Accuracy:  {accuracy_score(y_test, test_preds):.4f}")
    print(f"Test ROC-AUC:   {roc_auc_score(y_test, test_probs):.4f}")
    print(f"Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(classification_report(y_test, test_preds, target_names=["Legitimate (0)", "Phishing (1)"]))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data("data/uci_phishing.csv")
    best_models, cv = run_cv_and_tuning(X_train, y_train)
    chosen_threshold = select_decision_threshold(best_models["Random Forest"], X_train, y_train, cv)
    run_pca_experiment(X_train, y_train, cv)
    evaluate_once_on_test(best_models["Random Forest"], X_train, y_train, X_test, y_test, threshold=chosen_threshold)
