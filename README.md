# **Sentiment Analysis on Social Media Data for Brand Monitoring**

This project implements a **complete end-to-end sentiment analysis system** for monitoring public opinion about brands using Twitter data. It combines:

* **A Machine-Learning model (TF–IDF + Logistic Regression)** trained on the Sentiment140 dataset
* **A lexicon-based sentiment analyzer (VADER)** optimized for social-media text
* **A Flask-based interactive dashboard** that fetches tweets, analyzes sentiment, and visualizes results

The system includes data scraping, preprocessing, ML training, lexicon analysis, prediction, and real-time visualization—fully aligned with the methodology described in the report.

---

# **Features**

### ✅ **Dual Sentiment Analysis Pipeline**

* **Machine Learning Model**

  * TF–IDF vectorization
  * Logistic Regression classifier
  * Trained on Sentiment140 (1.6M tweets)
  * Evaluation: accuracy, precision, recall, F1-score, confusion matrix

* **VADER Lexicon-based Model**

  * Handles emojis, capitalization, intensifiers, elongated words
  * Ideal for short, informal tweets

### ✅ **Real-Time Tweet Collection & Fallback**

* Fetches brand-related tweets via **SNScrape**
* Automatic **fallback to local cleaned dataset** if scraping fails

### ✅ **Data Preprocessing**

* Cleaning: URLs, mentions, hashtags, emojis, punctuation
* Normalization and lowercasing
* Optional tokenization and stopword removal
* Generates cleaned CSVs for ML and lexicon models

### ✅ **Interactive Flask Dashboard**

* Input any brand name
* View:

  * Raw and cleaned tweets
  * ML + VADER sentiment predictions
  * Positive/Negative/Neutral distribution
  * Summary statistics and sentiment tables

### ✅ **Modular & Extensible Architecture**

* Separate scripts for scraping, preprocessing, ML training, VADER, prediction, and dashboard
* Supports offline mode and scalable deployment

---

# **Repository Structure**

```
app/
  flask_app.py          – Flask backend & dashboard

src/
  collect_tweets.py     – Brand tweet scraping (SNScrape)
  preprocess.py         – Text cleaning and normalization
  train_ml.py           – TF-IDF + Logistic Regression training pipeline
  lexicon_baseline.py   – VADER sentiment scoring
  predict.py            – Batch predictions using ML model
  download_dataset.py   – Downloads Sentiment140 dataset

models/
  sentiment_model.pkl   – Final ML pipeline (vectorizer + classifier)

data/
  clean_tweets.csv      – Preprocessed Sentiment140 dataset
  <brand>.csv           – Raw scraped tweets
  <brand>.cleaned.csv   – Preprocessed brand tweets
  <brand>.vader.csv     – VADER output
  <brand>.predicted.csv – ML + VADER combined output

requirements.txt
```

Large datasets & model artifacts are tracked using **Git LFS**.

---

# **Prerequisites**

* Python **3.10+**
* Git **LFS**
* Recommended: a virtual environment

### Install Git LFS

```bash
git lfs install
```

---

# **Setup**

```bash
git clone https://github.com/sakshamsinghania/Sentiment-Analysis-on-Social-Media-Data-for-Brand-Monitoring.git
cd Sentiment-Analysis-on-Social-Media-Data-for-Brand-Monitoring/brand-sentiment

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install Flask kagglehub
```

---

# **Data Acquisition**

### **Option 1 — Download Sentiment140**

```bash
python src/download_dataset.py
```

### **Option 2 — Scrape Tweets for a Brand**

```bash
python src/collect_tweets.py "Nike" 2025-01-01 2025-01-07 1000
```

---

# **Preprocessing**

Cleans and normalizes text:

```bash
python src/preprocess.py data/tweets.csv
python src/preprocess.py data/Nike_2025-01-01_2025-01-07.csv
```

Outputs `.cleaned.csv`.

---

# **Lexicon-Based Sentiment Analysis (VADER)**

```bash
python src/lexicon_baseline.py data/Nike_2025-01-01_2025-01-07.cleaned.csv
```

Outputs `.vader.csv`.

---

# **Machine Learning Model Training**

Trains TF–IDF + Logistic Regression:

```bash
python src/train_ml.py data/clean_tweets.csv data/sentiment_model.pkl
```

Outputs:

* `sentiment_model.pkl` (vectorizer + classifier)
* Evaluation metrics
* Confusion matrix
* Comparison with VADER

---

# **Batch Prediction on Brand Tweets**

```bash
python src/predict.py models/sentiment_model.pkl data/<brand>.vader.csv
```

Outputs `<brand>.predicted.csv` with ML labels added.

---

# **Flask Dashboard**

Start dashboard:

```bash
python app/flask_app.py
```

Open: **[http://localhost:5000/](http://localhost:5000/)**

Dashboard includes:

* Brand search
* ML + VADER sentiment for tweets
* Sentiment distribution visualization
* Raw & cleaned tweet previews
* Fallback to local dataset if scraping fails

---

# **Git LFS Usage**

```bash
git lfs fetch --all
git lfs checkout
```

---

# **Troubleshooting**

### Missing NLTK Corpora

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### SNScrape Errors

Scraping may fail due to rate limits or connection issues.
The dashboard automatically switches to `clean_tweets.csv`.

---

# **System Overview (from Report)**

### **1. Data Collection**

SNScrape → fallback to Sentiment140

### **2. Preprocessing**

Clean → normalize → tokenize → save cleaned text

### **3. ML Model**

TF–IDF + Logistic Regression
Accuracy ≈ **77%**
Balanced precision/recall for positive & negative classes

### **4. VADER Analysis**

Handles emojis, intensifiers, casing
Accuracy: **70–75%** expected on noisy text

### **5. Dashboard**

Interactive sentiment visualization + tweet-level predictions

### **6. Future Scope**

* Aspect-based sentiment analysis
* Multi-emotion classification
* Deep learning (BERT/LSTM)
* Real-time streaming APIs
* Multilingual support

