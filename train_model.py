"""Reproducible PhishGuard training, model comparison and evaluation."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             classification_report, RocCurveDisplay)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "uci_phishing.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 42
THRESHOLD = 0.35
TARGET_CANDIDATES = ["class", "Result", "result"]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python download_dataset.py")
    df = pd.read_csv(DATA_PATH)
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(f"Could not find target column. Expected one of {TARGET_CANDIDATES}; found {list(df.columns)}")
    df = df.drop(columns=[c for c in ["Index", "index"] if c in df.columns])
    y_raw = df.pop(target)
    # Report convention: phishing (-1) -> positive class 1; legitimate (+1) -> 0.
    if set(pd.Series(y_raw).dropna().unique()).issubset({-1, 1}):
        y = y_raw.map({-1: 1, 1: 0}).astype(int)
    else:
        y = y_raw.astype(int)
    X = df.apply(pd.to_numeric, errors="coerce")
    if X.shape[1] != 30:
        raise ValueError(f"Expected 30 feature columns after removing index/target; found {X.shape[1]}.")
    return X, y


def make_pipeline(model):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    prep = ColumnTransformer([("num", numeric, list(range(30)))])
    return Pipeline([("preprocessor", prep), ("classifier", model)])


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    specs = {
        "Dummy (most frequent)": (DummyClassifier(strategy="most_frequent"), {}),
        "Dummy (stratified)": (DummyClassifier(strategy="stratified", random_state=SEED), {}),
        "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=SEED),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
            {"classifier__max_depth": [8, 12, 18, None], "classifier__min_samples_split": [2, 10],
             "classifier__criterion": ["gini", "entropy"]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"classifier__n_estimators": [100, 150], "classifier__max_depth": [15, None],
             "classifier__min_samples_leaf": [1, 2]}),
        "SVM (RBF)": (SVC(kernel="rbf", probability=True, random_state=SEED),
            {"classifier__C": [1.0, 5.0, 10.0], "classifier__gamma": ["scale", 0.05, 0.1]}),
        "KNN": (KNeighborsClassifier(),
            {"classifier__n_neighbors": [3, 5, 7, 9], "classifier__weights": ["uniform", "distance"],
             "classifier__p": [1, 2]}),
    }
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1",
               "accuracy": "accuracy", "roc_auc": "roc_auc"}
    fitted, cv_rows, best_params = {}, [], {}
    print(f"Dataset: {len(X)} rows | {X.shape[1]} features | phishing={int(y.sum())} | legitimate={int((1-y).sum())}")
    print("Training models with 5-fold stratified CV. This may take a few minutes...\n")

    for name, (estimator, grid) in specs.items():
        pipe = make_pipeline(estimator)
        if grid:
            search = GridSearchCV(pipe, grid, scoring="recall", cv=cv, n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params[name] = search.best_params_
        else:
            best = pipe.fit(X_train, y_train)
            best_params[name] = {"strategy": estimator.strategy}
        fitted[name] = best
        scores = cross_validate(best, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"]))
        cv_rows.append(row)
        print(f"{name:24s} recall={row['recall_mean']:.4f} ± {row['recall_std']:.4f}")

    # Select threshold from training-only out-of-fold Random Forest probabilities.
    rf = fitted["Random Forest"]
    oof_prob = cross_val_predict(rf, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold_rows = []
    for t in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        pred = (oof_prob >= t).astype(int)
        fn = int(((y_train == 1) & (pred == 0)).sum())
        fp = int(((y_train == 0) & (pred == 1)).sum())
        threshold_rows.append({"threshold": t, "recall": recall_score(y_train, pred),
                               "precision": precision_score(y_train, pred, zero_division=0),
                               "f1": f1_score(y_train, pred), "FN": fn, "FP": fp,
                               "cost_10_FN_1_FP": 10 * fn + fp})
    pd.DataFrame(threshold_rows).to_csv(REPORT_DIR / "threshold_sweep.csv", index=False)

    # PCA experiment: fit scaler/PCA on training data only, then evaluate RF with 23 components.
    pca_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("pca", PCA(n_components=0.95, random_state=SEED)),
                         ("classifier", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1))])
    pca_scores = cross_validate(pca_pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    pca_pipe.fit(X_train, y_train)
    pca_components = int(pca_pipe.named_steps["pca"].n_components_)

    # Final untouched test-set evaluation happens once, after model selection.
    test_rows = []
    for name, model in fitted.items():
        if name.startswith("Dummy"):
            pred = model.predict(X_test)
            prob = None
        else:
            prob = model.predict_proba(X_test)[:, 1]
            pred = (prob >= THRESHOLD).astype(int) if name == "Random Forest" else (prob >= 0.5).astype(int)
        test_rows.append({"model": name, "recall": recall_score(y_test, pred),
                          "precision": precision_score(y_test, pred, zero_division=0),
                          "f1": f1_score(y_test, pred), "accuracy": accuracy_score(y_test, pred),
                          "roc_auc": roc_auc_score(y_test, prob) if prob is not None else 0.5,
                          "confusion_matrix_TP_FN_FP_TN": [int(confusion_matrix(y_test, pred).ravel()[3]), int(confusion_matrix(y_test, pred).ravel()[2]), int(confusion_matrix(y_test, pred).ravel()[1]), int(confusion_matrix(y_test, pred).ravel()[0]) ]})
    # Add PCA result using the same untouched test set, once.
    pca_prob = pca_pipe.predict_proba(X_test)[:, 1]
    pca_pred = (pca_prob >= 0.5).astype(int)
    test_rows.append({"model": f"Random Forest + PCA ({pca_components} components)",
                      "recall": recall_score(y_test, pca_pred), "precision": precision_score(y_test, pca_pred),
                      "f1": f1_score(y_test, pca_pred), "accuracy": accuracy_score(y_test, pca_pred),
                      "roc_auc": roc_auc_score(y_test, pca_prob),
                      "confusion_matrix_TP_FN_FP_TN": [int(confusion_matrix(y_test, pca_pred).ravel()[3]), int(confusion_matrix(y_test, pca_pred).ravel()[2]), int(confusion_matrix(y_test, pca_pred).ravel()[1]), int(confusion_matrix(y_test, pca_pred).ravel()[0]) ]})

    # Final saved model is Random Forest at the report's chosen 0.35 threshold.
    joblib.dump(rf, MODEL_DIR / "phishguard_random_forest.joblib")
    metadata = {"threshold": THRESHOLD, "positive_class": "phishing", "random_seed": SEED,
                "feature_columns": list(X.columns), "best_parameters": best_params,
                "pca_components_for_95_percent_variance": pca_components,
                "pca_variance_retained": float(pca_pipe.named_steps["pca"].explained_variance_ratio_.sum())}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(cv_rows).to_csv(REPORT_DIR / "cross_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_json(REPORT_DIR / "test_results.json", orient="records", indent=2)
    (REPORT_DIR / "classification_report.txt").write_text(classification_report(y_test, (rf.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int),
        target_names=["Legitimate", "Phishing"]))

    # Figures for the project demonstration.
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = pd.Series(y).map({0: "Legitimate", 1: "Phishing"}).value_counts()
    counts.plot(kind="bar", ax=ax, title="UCI Phishing Dataset – Class Balance", ylabel="Websites")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "class_balance.png", dpi=160); plt.close(fig)

    pred = (rf.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Legitimate", "Phishing"], ax=ax, cmap="Blues")
    ax.set_title("Random Forest – Test Confusion Matrix (threshold 0.35)")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)

    importances = pd.Series(rf.named_steps["classifier"].feature_importances_, index=X.columns).nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5)); importances.plot(kind="barh", ax=ax, title="Top 10 Random Forest Feature Importances")
    ax.set_xlabel("Gini importance"); fig.tight_layout(); fig.savefig(REPORT_DIR / "feature_importance.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")
    ax.set_title("ROC Curve – Held-out Test Set"); fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curve.png", dpi=160); plt.close(fig)

    print("\nTraining complete.")
    print(f"Saved model: {MODEL_DIR / 'phishguard_random_forest.joblib'}")
    print(f"Selected threshold: {THRESHOLD}")
    print("Final test results saved in reports/test_results.json")
    final = next(r for r in test_rows if r["model"] == "Random Forest")
    print(f"Random Forest test recall: {final['recall']:.4f} | precision: {final['precision']:.4f} | ROC-AUC: {final['roc_auc']:.4f}")
    print("Note: results are calculated from this downloaded dataset and run; they may not exactly match report values.")


if __name__ == "__main__":
    main()
