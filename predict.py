"""Predict from a CSV containing rows of the same 30 UCI features used for training.
This deliberately does not claim to classify an arbitrary raw URL: 21 UCI features
require webpage/domain observations that cannot safely be inferred from URL text alone.
"""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "phishguard_random_forest.joblib"
META = ROOT / "models" / "metadata.json"


def main():
    parser = argparse.ArgumentParser(description="Predict phishing risk from UCI feature rows.")
    parser.add_argument("csv", help="CSV with the 30 feature columns; target column is not needed")
    args = parser.parse_args()
    if not MODEL.exists() or not META.exists():
        raise SystemExit("Model not found. Run: python train_model.py")
    metadata = json.loads(META.read_text())
    frame = pd.read_csv(args.csv)
    # Remove optional index column; reorder columns exactly as training data.
    frame = frame.drop(columns=[c for c in ["Index", "index", "class", "Result", "result"] if c in frame.columns])
    expected = metadata["feature_columns"]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise SystemExit(f"Input is missing {len(missing)} required feature columns: {missing}")
    frame = frame[expected]
    model = joblib.load(MODEL)
    scores = model.predict_proba(frame)[:, 1]
    threshold = metadata["threshold"]
    for i, score in enumerate(scores, start=1):
        label = "POTENTIAL PHISHING" if score >= threshold else "LIKELY LEGITIMATE"
        print(f"Row {i}: {label} | phishing score: {score:.1%} | threshold: {threshold}")
    print("\nEducational result only. Do not treat this model as a guarantee of website safety.")


if __name__ == "__main__":
    main()
