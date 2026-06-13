# 🛡️ ReviewShield AI
### AI-Powered Review Intelligence System

> Detect fake reviews, analyze sentiment, and generate authenticity insights using Machine Learning and NLP.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model (first time only)
python model.py

# 3. Run the app
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Fake Detection** | Logistic Regression classifier detects fake vs genuine reviews |
| 💬 **Sentiment Analysis** | Positive / Neutral / Negative classification |
| 🚨 **Spam Detection** | Detects ALL CAPS, promo phrases, excessive punctuation |
| 🏆 **Authenticity Score** | 0–100 composite score per review |
| 📂 **Bulk Analysis** | Upload CSV for batch processing |
| 📊 **Dashboard** | Charts: trends, sentiment distribution, score histograms |
| ⬇️ **Export** | Download results as CSV |

---

## 📁 Project Structure

```
reviewshield-ai/
├── app.py                  # Main Streamlit app
├── model.py                # ML training & prediction
├── preprocess.py           # Text cleaning pipeline
├── sentiment.py            # Sentiment analysis
├── spam_detector.py        # Spam pattern detection
├── requirements.txt        # Dependencies
├── dataset/
│   └── reviews_dataset.csv # Training data
├── models/
│   ├── fake_review_model.pkl
│   └── vectorizer.pkl
└── utils/
    └── helpers.py          # Scoring utilities
```

---

## 🧠 ML Pipeline

```
Raw Review Text
      ↓
Text Preprocessing (lowercase, remove punct, stopwords, stemming)
      ↓
TF-IDF Vectorization (5000 features, bigrams)
      ↓
Logistic Regression Classifier
      ↓
Prediction + Confidence Score
      ↓
Sentiment Analysis + Spam Detection
      ↓
Authenticity Score (0–100)
      ↓
Dashboard & Analytics
```

---

## 📊 Resume Description

> **ReviewShield AI – AI-Powered Review Intelligence System**
> Developed an NLP-based fake review detection system using Python and machine learning.
> Implemented TF-IDF vectorization, Logistic Regression classifier, sentiment analysis, spam pattern detection, and authenticity scoring. Built interactive Streamlit dashboards, bulk CSV analysis, and downloadable report generation for review analytics.

---

## 🛠️ Tech Stack

- **Python 3** — Core language
- **scikit-learn** — ML model + TF-IDF
- **Streamlit** — Modern web UI
- **pandas / numpy** — Data processing
- **matplotlib** — Visualizations
- **joblib** — Model persistence

---

*Built for Amazon internship project · ReviewShield AI v1.0*
