# src/preprocess.py
"""
Usage:
    For brand-specific CSV:
        python src/preprocess.py data/Nike_2025-01-01_2025-01-07.csv

    For Sentiment140:
        python src/preprocess.py data/tweets.csv

Output:
    data/<filename>.cleaned.csv
"""

import sys
import os
import re
import pandas as pd
import emoji
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK dependencies silently
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

# -----------------------------------
# 🔹 Generic text cleaning functions
# -----------------------------------

def remove_urls_mentions(text: str) -> str:
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text

def demojize_text(text: str) -> str:
    # Convert emoji to textual description
    return emoji.demojize(text, language='en')

def clean_text(text: str) -> str:
    """Full cleaning pipeline: URLs, mentions, emojis, punctuation, stopwords."""
    if not isinstance(text, str):
        return ""
    text = remove_urls_mentions(text)
    text = demojize_text(text)
    text = re.sub(r'#', '', text)                 # remove hashtag symbols
    text = re.sub(r'[^A-Za-z0-9\s:]', ' ', text) # keep alphanumeric + spaces
    text = text.lower()
    tokens = [t for t in text.split() if t not in STOP and len(t) > 1]
    tokens = [LEM.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# -----------------------------------
# 🔹 Dataset-specific logic
# -----------------------------------

def process_brand_file(infile: str):
    """For brand CSVs with a 'content' column."""
    df = pd.read_csv(infile)
    if 'content' not in df.columns:
        raise ValueError("Input CSV must have a 'content' column")
    df['clean_text'] = df['content'].fillna('').astype(str).apply(clean_text)
    out = infile.replace('.csv', '.cleaned.csv')
    df.to_csv(out, index=False)
    print(f"✅ Saved cleaned brand data to {out}")
    return out

def process_sentiment140(infile: str):
    """For Sentiment140 dataset format."""
    df = pd.read_csv(infile, encoding='latin-1', header=None)
    df = df.rename(columns={0: "target", 5: "text"})

    # Map numeric target → textual sentiment
    df["sentiment"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"})

    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    out = "data/clean_tweets.csv"
    df[["clean_text", "sentiment"]].to_csv(out, index=False)
    print(f"✅ Cleaned Sentiment140 data saved to {out}")
    return out

# -----------------------------------
# 🔹 Main entrypoint
# -----------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/preprocess.py <input_csv>")
        sys.exit(1)
    infile = sys.argv[1]

    # Automatically detect dataset type
    if os.path.basename(infile) == "tweets.csv":
        process_sentiment140(infile)
    else:
        process_brand_file(infile)

if __name__ == "__main__":
    main()
