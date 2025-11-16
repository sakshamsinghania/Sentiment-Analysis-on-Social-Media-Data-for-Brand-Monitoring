from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Try importing snscrape (optional)
try:
    import snscrape.modules.twitter as sntwitter
    SNS_ENABLED = True
    print("✅ snscrape loaded successfully.")
except Exception as e:
    print("⚠️ snscrape import failed, falling back to local dataset:", e)
    SNS_ENABLED = False

# Initialize Flask app
app = Flask(__name__)

# Load trained model (Pipeline with TF-IDF + Logistic Regression)
MODEL_PATH = os.path.join("data", "sentiment_model.pkl")
model = joblib.load(MODEL_PATH)

# ---------- Tweet Collection Function ----------
def collect_tweets(brand, limit=10):
    """Collect tweets using snscrape or fallback to local CSV."""
    tweets = []

    if SNS_ENABLED:
        print(f"📡 Fetching tweets for '{brand}'...")
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{brand} lang:en').get_items()):
            if i >= limit:
                break
            tweets.append(tweet.content)

    if not tweets:
        print("⚠️ No tweets found, switching to fallback dataset.")

    if not tweets:  # fallback: use local cleaned CSV sample
        df = pd.read_csv("data/clean_tweets.csv").sample(limit)
        tweets = df["clean_text"].tolist()

    return tweets

# ---------- Sentiment Prediction ----------
def predict_sentiment(texts):
    # The model pipeline already contains TF-IDF, so feed raw text
    preds = model.predict(texts)
    results = list(zip(texts, preds))
    return results

# ---------- Flask Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    brand = None
    summary = None

    if request.method == "POST":
        brand = request.form["brand"]
        tweets = collect_tweets(brand, limit=10)
        results = predict_sentiment(tweets)

        # Calculate sentiment summary
        labels = [label for _, label in results]
        total = len(labels)
        summary = {
            'positive': round(labels.count('positive') / total * 100, 1),
            'neutral': round(labels.count('neutral') / total * 100, 1),
            'negative': round(labels.count('negative') / total * 100, 1)
        }

    return render_template("index.html", results=results, brand=brand, summary=summary)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
