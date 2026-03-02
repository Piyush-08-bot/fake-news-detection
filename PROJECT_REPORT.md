<div align="center">

---

# 🎓 Rishihood University
## Department of Computer Science

<br>

# Fake News Detection System
### *Using Natural Language Processing and Machine Learning*

<br>

**Submitted in partial fulfilment of the requirements for**
Bachelor of Technology in Computer Science & Artificial Intelligence
*(Course: Generative AI)*

<br>

---

### 👥 Submitted By

| Name | Enrollment No. |
|:---|:---|
| Vikash Kumar | 2401010503 |
| Piyush Raj | 2401010328 |
| Mohammed Yaseen | 2401010281 |
| Sankalp M Tellur | 2401010416 |

<br>

**Submitted To:** Kartik Gupta
**Department:** Computer Science
**Academic Year:** 2025–2026 &nbsp;|&nbsp; **Submitted:** 2 March 2026

<br>

---

</div>

<div align="left">

🔗 **Live Demo:** [fake-news-detection-ml-ai.streamlit.app](https://fake-news-detection-ml-ai.streamlit.app/)<br>
📁 **GitHub (Piyush):** [github.com/Piyush-08-bot/fake-news-detection](https://github.com/Piyush-08-bot/fake-news-detection)

</div>

---

## 📋 Table of Contents

| # | Section |
|:---:|:---|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Data Description](#2-data-description) |
| 3 | [Exploratory Data Analysis](#3-exploratory-data-analysis) |
| 4 | [Methodology](#4-methodology) |
| 5 | [Model Training Workflow](#5-model-training-workflow) |
| 6 | [Evaluation](#6-evaluation) |
| 7 | [Optimisation](#7-optimisation) |
| 8 | [Application Architecture](#8-application-architecture) |
| 9 | [Conclusion](#9-conclusion) |
| 10 | [Team Contribution](#10-team-contribution) |

---

## 1. Problem Statement

Fake news is not a new problem, but the internet has made it far harder to control. A single misleading article can reach millions of people within hours through social media, WhatsApp forwards, and news aggregators. By the time a correction is published, the damage is already done.

The core challenge we wanted to tackle is simple:

> **"Can a machine learning system read a news article and determine whether it is Real or Fake — purely based on the words used in it?"**

We intentionally avoided relying on external fact-checking databases or APIs. The system should work from the article text alone, looking for language patterns that distinguish credible journalism from fabricated content.

**Our goals were to:**
- ✅ Build a pipeline that processes raw article text and classifies it as *Real* or *Fake*
- ✅ Show a confidence score along with the prediction
- ✅ Deliver this through a clean, interactive web dashboard anyone can use

---

## 2. Data Description

### 2.1 Dataset Source

We used the **ISOT Fake News Dataset**, created by the Information Security and Object Technology (ISOT) Lab at the University of Victoria.

> 🔗 **Dataset Link:** [ISOT Fake News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| Source | Description |
|:---|:---|
| **Real News** | Scraped from **Reuters.com** — a globally trusted wire news service |
| **Fake News** | Collected from sites flagged by **PolitiFact** and other fact-checkers |
| **Time Period** | 2016–2017 (heavily focused on the US Presidential Election) |

### 2.2 Dataset Size

| Class | Count | Share |
|:---|:---:|:---:|
| ✅ Real Articles | 21,417 | **55.4%** |
| ❌ Fake Articles | 17,229 | **44.6%** |
| **Total** | **38,646** | **100%** |

> **Why this balance matters:** A near-balanced dataset means the model cannot just guess "Real" every time and score well. It is forced to actually learn distinguishing patterns.

### 2.3 Features Used

The dataset had columns for `title`, `text`, `subject`, and `date`. We only trained on:

| Column | Used? | Reason |
|:---|:---:|:---|
| `text` | ✅ Yes | The full article body — our primary input |
| `label` | ✅ Yes | 0 = Fake, 1 = Real |
| `subject` | ❌ No | **Data leakage** — it perfectly separates classes without learning real language |
| `title` / `date` | ❌ No | Metadata, not article language |

---

## 3. Exploratory Data Analysis

Before building any model, we explored the data to understand its structure and patterns.

### 3.1 Class Balance

The dataset is **55% Real / 45% Fake** — close enough to balanced that no special resampling (like SMOTE) was needed.

```
Real  ████████████████████████░░░░░░░░  55.4%
Fake  ░░░░░░░░░░░░░░░░░░░░████████████  44.6%
```

### 3.2 Article Lengths

We measured article lengths in characters and found a notable difference:

- 📰 **Real articles** — consistent length (~2,500–3,500 chars). Wire journalism follows editorial standards.
- 🚨 **Fake articles** — highly variable. Ranges from ultra-short clickbait posts to very long conspiracy pieces.

> **Insight:** Article length alone is not enough to classify news, but it confirms that fake and real content have different structural habits.

### 3.3 Top Words in the Dataset

After cleaning the text, the most frequent words across both classes:

| Rank | Word | Notes |
|:---:|:---|:---|
| 1 | `said` | Journalism attributes quotes to people |
| 2 | `trump` | Election-era dataset |
| 3 | `president` | Heavy political focus |
| 4 | `us` | US-centric news |
| **6** | **`reuters`** | ⚠️ Almost exclusively in Real articles — a data bias |

> ⚠️ **Important Note:** The word *"reuters"* being this frequent is a **dataset bias**. Real articles in this dataset are mostly Reuters wire stories. The model will learn to associate this word with Real news, which may not generalise well to other real news sources.

### 3.4 Word Signals by Class

When we examined the model's learned weights, we found two distinct vocabulary clusters:

| Signal | Words |
|:---|:---|
| 🟢 **Real News words** | `reuters`, `washington`, `official`, `statement`, `government` |
| 🔴 **Fake News words** | `video`, `image`, `share`, `watch`, emotional & sensational terms |

This makes intuitive sense — credible journalism uses attribution language, while fake articles use engagement-bait vocabulary.

---

## 4. Methodology

### 4.1 Text Preprocessing Pipeline

Raw news text cannot go directly into a model. Every article passed through this cleaning pipeline, defined in `utils.py`:

```
Raw Text
  │
  ├─ Step 1 → Lowercase everything            ("Trump" = "trump")
  ├─ Step 2 → Remove punctuation & numbers    (keep only letters)
  ├─ Step 3 → Tokenise                        (split into words)
  ├─ Step 4 → Remove stopwords                ("the", "is", "a" → gone)
  ├─ Step 5 → Lemmatise                       ("running" → "run")
  └─ Step 6 → Rejoin into cleaned string
```

> **Why Lemmatisation over Stemming?** Stemming can produce non-words (e.g., "runn"). Lemmatisation always returns a valid base English word, giving TF-IDF a cleaner vocabulary to work with.

### 4.2 Feature Extraction — TF-IDF

After preprocessing, we convert each article into numbers using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

**The core idea:**
- A word appearing a lot in *one article* but rarely across *all articles* → **High TF-IDF score** (important word)
- A word appearing in *every article* → **Low TF-IDF score** (generic, unhelpful)

**Configuration used:**

| Parameter | Value | Why |
|:---|:---:|:---|
| `max_features` | 5,000 | Caps vocabulary to reduce noise |
| `sublinear_tf` | True | Log-scale dampens very common terms |
| `ngram_range` | (1,1) | Unigrams only — computationally efficient |

### 4.3 Classification — Logistic Regression vs Decision Tree

We trained and compared two models:

| Factor | ✅ Logistic Regression | Decision Tree |
|:---|:---:|:---:|
| Test Accuracy | **98.50%** | 99.71% |
| Overfitting Risk | **Low** | Higher |
| Confidence Scores | **Smooth & reliable** | Jumpy |
| Interpretability | **High** (word coefficients) | Moderate |
| **Chosen for deployment?** | **✅ Yes** | ❌ No |

> 🏆 **Why we chose Logistic Regression:** The Decision Tree's slightly higher accuracy (99.71%) came from memorising specific article phrases — a classic overfitting sign. Logistic Regression learns *general* linguistic patterns that hold up on new, unseen articles.

---

## 5. Model Training Workflow

Training was carried out in `News_Credibility_Training.ipynb`. The full pipeline:

**① Load Data**
Load `True.csv` and `Fake.csv`, assign labels (`Real=1`, `Fake=0`), and merge into one DataFrame.

**② Clean Data**
Remove duplicates, drop null rows, and drop `title`, `subject`, `date` columns to prevent data leakage.

**③ Preprocess Text**
Run every article through `preprocess_text()` from `utils.py`. Store cleaned text in a new column.

**④ Split Data**
80% training / 20% testing with **stratified sampling** — both splits maintain the same class ratio.

**⑤ Train Models**
Fit `TF-IDF + Logistic Regression` and `TF-IDF + Decision Tree` pipelines on the training set.

**⑥ Evaluate**
Predict on the test set. Compute Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.

**⑦ Save Models**
Serialise trained pipelines with `joblib` to the `models/` directory for fast dashboard loading.

```python
# Production pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, sublinear_tf=True)),
    ('classifier', LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)
```

---

## 6. Evaluation

> All results are computed on the **held-out 20% test set — 7,730 articles the model had never seen during training.**

### 6.1 Performance Metrics

| Metric | Score | What it means |
|:---|:---:|:---|
| 🎯 **Accuracy** | **98.50%** | Overall correct predictions |
| 🔍 **Precision** | 97.95% | Of articles predicted Real, how many actually were |
| 📡 **Recall** | 99.34% | Of all Fake articles, how many were correctly caught |
| ⚖️ **F1 Score** | **98.64%** | Balanced measure of Precision and Recall |
| 📈 **ROC-AUC** | **0.9987** | Near-perfect discrimination ability |

### 6.2 Confusion Matrix

|  | Predicted Fake | Predicted Real |
|:---|:---:|:---:|
| **Actual Fake** | ✅ **3,411** (correct) | ❌ 88 (missed) |
| **Actual Real** | ❌ 28 (false alarm) | ✅ **4,203** (correct) |

> 🚨 **The most dangerous error** is a *False Negative* — a fake article that slips through as Real. We had only **28 such cases** out of 7,730 articles, giving us a **Recall of 99.34%**. That's a very strong result.

### 6.3 ROC-AUC

An **AUC of 0.9987** means the model has a 99.87% chance of correctly ranking a Real article above a Fake one. Perfect would be 1.0; random guessing would be 0.5.

---

## 7. Optimisation

We made several conscious decisions to improve model quality and system performance:

### 7.1 Model Quality

| Decision | Impact |
|:---|:---|
| **Lemmatisation over Stemming** | Cleaner vocabulary → better TF-IDF features |
| **Dropped leakage columns** | Forces model to learn real language patterns, not metadata |
| **`sublinear_tf=True`** | Prevents dominant common words from drowning out rare but meaningful ones |
| **`max_features=5000`** | Balanced capacity vs. overfitting — tested higher values with no improvement |
| **LR over Decision Tree** | More stable, smoother probabilities, generalises better |

### 7.2 System Performance

| Decision | Impact |
|:---|:---|
| **`@st.cache_resource` on model loading** | Model loads once at startup, not on every user click — much faster |
| **`joblib` serialisation** | Optimised for large NumPy arrays (TF-IDF weight matrices), faster than `pickle` |

---

## 8. Application Architecture

### 8.1 How It Works

The app is a full-stack Python web application built with Streamlit:

```
User (Browser)
      │
      ▼
┌─────────────────────────────────┐
│  Streamlit Dashboard (app.py)   │
│  ┌──────────┐  ┌─────────────┐  │
│  │Text Input│  │  URL Input  │  │
│  └────┬─────┘  └──────┬──────┘  │
└───────┼────────────────┼────────┘
        │           newspaper3k
        │           extracts text
        └─────────────┬───────────
                      ▼
               preprocess_text()       ← utils.py
                      │
                      ▼
              models/model.pkl
          (TF-IDF → Logistic Regression)
                      │
                 ┌────┴────┐
           Prediction   Probabilities
              (0/1)    [P(Fake), P(Real)]
                      │
                      ▼
            Charts, Verdict, Report
```

### 8.2 Dashboard Pages

| Page | What It Shows |
|:---|:---|
| 📰 **Analysis** | Enter text or URL → verdict, confidence bar, charts, auto-report |
| 📊 **Model Performance** | Accuracy metrics, confusion matrix, dataset distribution, feature importance |
| 🔬 **Classifier Deep Dive** | ROC curve, article length distribution |
| 🧠 **NLP Insights** | Top dataset vocabulary, LR vs Decision Tree comparison |

### 8.3 Key Files

| File | Role |
|:---|:---|
| `app.py` | Full dashboard — all 4 pages, charts, navigation, CSS theme |
| `utils.py` | `preprocess_text()` — the NLP cleaning pipeline |
| `models/model.pkl` | Trained LR pipeline (production) |
| `models/vectorizer.pkl` | Standalone TF-IDF (for per-article word inspection) |
| `News_Credibility_Training.ipynb` | End-to-end training notebook |
| `requirements.txt` | All Python dependencies |
| `.streamlit/config.toml` | Light theme configuration |

### 8.4 Tech Stack

| Layer | Technology |
|:---|:---|
| ML / NLP | `scikit-learn`, `NLTK` |
| Web Framework | `Streamlit` |
| Visualisation | `Plotly`, `Matplotlib`, `Seaborn` |
| URL Scraping | `newspaper3k` |
| Model I/O | `joblib` |
| UI Components | `streamlit-shadcn-ui` |
| Hosting | Streamlit Community Cloud |

---

## 9. Conclusion

We started this project asking a straightforward question — can a computer tell fake news from real news just by reading it? The answer, it turns out, is yes — and with very high confidence.

Using TF-IDF vectorisation combined with Logistic Regression, we achieved **98.50% accuracy** and an **AUC of 0.9987** on 7,730 unseen test articles. The final system is live and interactive — anyone can paste an article or drop a URL and get an instant verdict with a confidence score and visual breakdown.

### Limitations We Acknowledge
- The dataset is **biased toward 2016 US political news**. The model may not generalise as well to modern or non-political articles.
- Because real articles are from Reuters, the word *"reuters"* became an artificially strong predictor.
- The system detects **linguistic patterns, not factual accuracy**. A well-written but factually wrong article could still score as "Real."

### Future Improvements
- 🤖 Fine-tune **BERT or RoBERTa** for deeper semantic understanding
- 🌍 Add **multilingual support** for non-English news
- 🔍 Integrate with a **fact-checking API** for a second layer of verification
- 🔄 Set up a **continuous retraining loop** as new misinformation patterns emerge

Overall, this project gave us solid hands-on experience with the full ML pipeline — from raw data all the way to a deployed, production application solving a real-world problem.

---

## 10. Team Contribution

| Name | Enrollment No. | Contributions |
|:---|:---:|:---|
| **Vikash Kumar** | 2401010503 | Dataset collection, text preprocessing pipeline (`utils.py`), model training notebook, model evaluation |
| **Piyush Raj** | 2401010328 | Backend application logic, integrating the trained model with the Streamlit dashboard (`app.py`), inference pipeline |
| **Mohammed Yaseen** | 2401010281 | Exploratory Data Analysis, model evaluation and comparison, writing the project report |
| **Sankalp M Tellur** | 2401010416 | Frontend UI design and development, GitHub repository setup, deployment to Streamlit Cloud, testing |

*All members collaborated on system design, integration testing, and final presentation of the application.*

---

*Submitted as part of the academic curriculum. All reported metrics are computed on held-out test data not used during model training.*
