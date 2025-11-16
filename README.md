# Sentiment Analysis on Social Media Data for Brand Monitoring

A practical pipeline and dashboard for brand sentiment monitoring using Twitter data. It combines a machine‑learning model (TF‑IDF + Logistic Regression) with a lexicon baseline (VADER), and provides a simple Flask UI to explore live or sample results.

## Features
- Data collection via `snscrape` for brand queries (optional)
- Text preprocessing with emoji handling, stopwords, and lemmatization
- ML training: TF‑IDF + Logistic Regression with evaluation metrics
- Lexicon baseline using VADER for comparison
- Batch prediction script to append model outputs to CSVs
- Flask dashboard to input a brand and view sentiment summary
- Large datasets tracked via Git LFS

## Repository Structure
- `app/flask_app.py` — Flask app serving a simple dashboard
- `src/collect_tweets.py` — Collect tweets via `snscrape`
- `src/preprocess.py` — Clean raw text to `clean_text`
- `src/lexicon_baseline.py` — Apply VADER and save scores/labels
- `src/train_ml.py` — Train TF‑IDF + Logistic Regression, save models, comparisons
- `src/predict.py` — Append ML predictions to a cleaned CSV
- `data/` — Datasets and model artifacts (tracked with Git LFS)
- `models/` — Pretrained artifacts (e.g., `logreg_tfidf.joblib`)
- `requirements.txt` — Python dependencies

## Prerequisites
- Python 3.10+
- Git LFS (required to fetch large CSV/model files)
- Recommended: virtual environment

Install Git LFS:
```bash
git lfs install
```

## Setup
```bash
# Clone (LFS-enabled)
git clone https://github.com/sakshamsinghania/Sentiment-Analysis-on-Social-Media-Data-for-Brand-Monitoring.git
cd Sentiment-Analysis-on-Social-Media-Data-for-Brand-Monitoring/brand-sentiment

# Create & activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# If missing, also install Flask and kagglehub (used by app/download script)
pip install Flask kagglehub
```

## Data Acquisition
You can either download the Sentiment140 dataset or collect brand‑specific tweets.

- Download Sentiment140 via Kaggle (requires `kagglehub`):
```bash
python src/download_dataset.py  # saves to data/tweets.csv
```

- Collect tweets for a brand using `snscrape`:
```bash
python src/collect_tweets.py "Nike" 2025-01-01 2025-01-07 1000
# output: data/Nike_2025-01-01_2025-01-07.csv
```

## Preprocessing
Generate cleaned text suitable for training and evaluation:
```bash
# For Sentiment140
python src/preprocess.py data/tweets.csv  # outputs data/clean_tweets.csv

# For brand CSVs with a 'content' column
python src/preprocess.py data/Nike_2025-01-01_2025-01-07.csv  # outputs ...cleaned.csv
```

## Baseline (VADER)
Apply VADER sentiment to cleaned brand data:
```bash
python src/lexicon_baseline.py data/Nike_2025-01-01_2025-01-07.cleaned.csv
# outputs ...cleaned.vader.csv
```

## Training (TF‑IDF + Logistic Regression)
Train the ML model and generate evaluation artifacts:
```bash
python src/train_ml.py data/clean_tweets.csv data/sentiment_model.pkl
# saves: data/sentiment_model.pkl, models/vectorizer.pkl, data/vader_info.pkl
# also writes: data/comparison_results.csv
```

## Batch Prediction
Append ML predictions to a VADER‑annotated CSV:
```bash
python src/predict.py models/logreg_tfidf.joblib data/Nike_2025-01-01_2025-01-07.cleaned.vader.csv
# outputs ...predicted.csv with 'ml_sentiment'
```

## Flask Dashboard
Run the dashboard and explore sentiment interactively:
```bash
python app/flask_app.py
# open http://localhost:5000/
```
Notes:
- If `snscrape` import fails, the app falls back to `data/clean_tweets.csv`.
- The app loads the trained pipeline from `data/sentiment_model.pkl`.

## Git LFS Notes
Large CSVs and model artifacts are tracked with Git LFS.
- Ensure `git lfs install` before cloning
- If files appear as tiny pointer text, run:
```bash
git lfs fetch --all
git lfs checkout
```

## Troubleshooting
- Missing packages: install `Flask` and `kagglehub` if not present.
- NLTK resources: first run downloads required corpora (stopwords, wordnet, vader_lexicon). If blocked, run:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('vader_lexicon')"
```
- Snscrape rate/availability: collection may intermittently fail; re‑run or use the Sentiment140 dataset.

---

### Code Pointers
- Flask app entry and model load: `brand-sentiment/app/flask_app.py:15–21`
- Brand tweet fetch fallback: `brand-sentiment/app/flask_app.py:27–41`
- Preprocessing pipeline: `brand-sentiment/src/preprocess.py:45–56`
- Sentiment140 cleaning & output: `brand-sentiment/src/preprocess.py:73–85`
- VADER baseline labeling: `brand-sentiment/src/lexicon_baseline.py:13–28`
- Model training and artifact saving: `brand-sentiment/src/train_ml.py:56–117`
- Batch prediction: `brand-sentiment/src/predict.py:14–25`