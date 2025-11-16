# src/predict.py
"""
Usage:
    python src/predict.py models/logreg_tfidf.joblib data/Nike_2025-01-01_2025-01-07.cleaned.vader.csv

Output:
    same CSV with appended model predictions saved as .predicted.csv
"""
import sys
import pandas as pd
import joblib
import os

def predict(model_path, infile):
    model = joblib.load(model_path)
    df = pd.read_csv(infile)
    if 'clean_text' not in df.columns:
        raise ValueError("Input CSV must have a 'clean_text' column")
    X = df['clean_text'].fillna('').astype(str)
    preds = model.predict(X)
    df['ml_sentiment'] = preds
    out = infile.replace('.csv', '.predicted.csv')
    df.to_csv(out, index=False)
    print(f"Saved predictions to {out}")
    return out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/predict.py models/logreg_tfidf.joblib data/file.cleaned.vader.csv")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])
