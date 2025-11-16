# src/train_ml.py
"""
Train a TF-IDF + Logistic Regression model and compare it with VADER sentiment analysis.

Usage:
    python src/train_ml.py data/clean_tweets.csv data/sentiment_model.pkl

The input CSV must have columns: clean_text,label  (label in {positive, negative, neutral})
"""

import sys
import os
import joblib
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ---------- VADER Lexicon Sentiment Function ----------
def vader_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# ---------- Training Function ----------
def train(infile, out_model_path):
    print(f"📘 Loading data from {infile} ...")
    df = pd.read_csv(infile)

    # Ensure required columns
    if 'clean_text' not in df.columns:
        raise ValueError("Input CSV must have a 'clean_text' column")
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})

    X = df['clean_text'].astype(str)
    y = df['label'].astype(str)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Train ML Model ----------
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    print("\n🚀 Training TF-IDF + Logistic Regression model...")
    pipe.fit(X_train, y_train)

    # Evaluate ML model
    y_pred_ml = pipe.predict(X_test)
    acc_ml = accuracy_score(y_test, y_pred_ml) * 100
    print(f"\n✅ ML Model Accuracy: {acc_ml:.3f}")
    print("\nClassification Report (ML):\n", classification_report(y_test, y_pred_ml))
    print("Confusion Matrix (ML):\n", confusion_matrix(y_test, y_pred_ml))

    # ---------- Apply VADER on same test set ----------
    print("\n🧠 Running VADER (Lexicon-based) Sentiment...")
    sid = SentimentIntensityAnalyzer()
    y_pred_vader = [vader_sentiment(t) for t in X_test]
    acc_vader = accuracy_score(y_test, y_pred_vader)
    print(f"\n✅ VADER Accuracy: {acc_vader:.3f}")
    print("\nClassification Report (VADER):\n", classification_report(y_test, y_pred_vader))
    print("Confusion Matrix (VADER):\n", confusion_matrix(y_test, y_pred_vader))

    # ---------- Comparison Summary ----------
    comparison = pd.DataFrame({
        "Text": X_test,
        "Actual": y_test,
        "ML_Predicted": y_pred_ml,
        "VADER_Predicted": y_pred_vader
    })

    comparison_out = os.path.join(os.path.dirname(out_model_path), "comparison_results.csv")
    comparison.to_csv(comparison_out, index=False)
    print(f"\n📊 Comparison saved to: {comparison_out}")

    # ---------- Save Models ----------
    os.makedirs(os.path.dirname(out_model_path) or '.', exist_ok=True)

    # Save full pipeline
    joblib.dump(pipe, out_model_path)
    print(f"\n💾 Full ML pipeline (model + vectorizer) saved to {out_model_path}")

    # Save separate components for Flask
    model_path = out_model_path.replace('.pkl', '_only_model.pkl')
    vectorizer_path = out_model_path.replace('sentiment_model.pkl', 'vectorizer.pkl')

    model = pipe.named_steps['clf']
    vectorizer = pipe.named_steps['tfidf']

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"💾 Model saved to {model_path}")
    print(f"💾 Vectorizer saved to {vectorizer_path}")

    # ---------- Save VADER for reuse ----------
    vader_info = {"type": "VADER", "description": "Lexicon-based Sentiment Analyzer"}
    vader_path = os.path.join(os.path.dirname(out_model_path), "vader_info.pkl")
    joblib.dump(vader_info, vader_path)
    print(f"💾 VADER info saved to {vader_path}")

    print("\n✅ Training complete! Ready for Flask dashboard comparison.")


# ---------- Main ----------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/train_ml.py data/clean_tweets.csv data/sentiment_model.pkl")
        sys.exit(1)

    infile = sys.argv[1]
    out_model = sys.argv[2]
    train(infile, out_model)
