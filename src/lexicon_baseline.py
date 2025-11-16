# src/lexicon_baseline.py
"""
Usage:
    python src/lexicon_baseline.py data/Nike_2025-01-01_2025-01-07.cleaned.csv

Output:
    data/Nike_2025-01-01_2025-01-07.cleaned.vader.csv
"""
import sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def label_from_compound(c):
    if c >= 0.05:
        return "Positive"
    if c <= -0.05:
        return "Negative"
    return "Neutral"

def apply_vader(infile):
    df = pd.read_csv(infile)
    if 'clean_text' not in df.columns:
        raise ValueError("Input CSV must have a 'clean_text' column (run preprocess.py first)")
    sia = SentimentIntensityAnalyzer()
    scores = df['clean_text'].astype(str).apply(lambda t: sia.polarity_scores(t)['compound'])
    df['vader_compound'] = scores
    df['vader_sentiment'] = df['vader_compound'].apply(label_from_compound)
    out = infile.replace('.cleaned.csv', '.cleaned.vader.csv')
    df.to_csv(out, index=False)
    print(f"Saved VADER results to {out}")
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/lexicon_baseline.py data/yourfile.cleaned.csv")
        sys.exit(1)
    apply_vader(sys.argv[1])
