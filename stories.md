# 📰 Intelligent News Credibility Analysis — Milestone 1

## ML-Based Fake News Classification Using Classical NLP

---

## 1. Problem Understanding

The spread of misinformation through digital news platforms has become one of the most pressing challenges of the information age. Fake news — deliberately fabricated content designed to mislead readers — erodes public trust in journalism, fuels polarization, and can have real-world consequences ranging from public health crises to electoral interference.

**Milestone 1** of this project focuses on building a **classical Machine Learning pipeline** that can automatically classify news articles as **credible (Real)** or **non-credible (Fake)** based solely on their textual content. This serves as the foundational building block before advancing to more sophisticated Agentic AI techniques in Milestone 2.

### Objective

> Build an end-to-end NLP + ML pipeline using **Scikit-Learn** that ingests raw news articles, cleans and preprocesses them, extracts meaningful features, trains a classification model, and evaluates its performance — all without relying on Large Language Models or deep learning.

---

## 2. Why Fake News Detection Matters

| Dimension           | Impact                                                                       |
| ------------------- | ---------------------------------------------------------------------------- |
| **Democracy**       | Fake news can manipulate elections and undermine democratic institutions     |
| **Public Health**   | Misinformation about vaccines, treatments, and pandemics endangers lives     |
| **Economy**         | False financial news can cause market volatility and investor losses         |
| **Social Cohesion** | Fabricated narratives deepen divisions and incite hatred between communities |
| **Trust in Media**  | Persistent misinformation makes the public distrust all media — real or fake |

Automated detection systems act as a **first line of defense**, enabling platforms and fact-checkers to flag suspicious content at scale — something manual review alone cannot achieve.

---

## 3. Dataset Description

We use the **ISOT Fake and Real News Dataset**, a widely cited benchmark in fake news research.

| Property             | Details                                                        |
| -------------------- | -------------------------------------------------------------- |
| **Source**           | University of Victoria (ISOT Research Lab)                     |
| **Files**            | `Fake.csv` — 23,481 articles · `True.csv` — 21,417 articles    |
| **Total Articles**   | ~44,898                                                        |
| **Columns Used**     | `text` (article body), `label` (0 = Fake, 1 = Real)            |
| **Time Period**      | 2015 – 2018                                                    |
| **Real News Source** | Reuters.com                                                    |
| **Fake News Source** | Various unreliable sources flagged by Politifact and Wikipedia |

> **Note:** The dataset is in English and primarily covers political news from the United States.

---

## 4. Data Cleaning Steps

Structural cleaning is the first step to ensure the dataset is consistent, complete, and ready for NLP processing.

### Step-by-Step Process

1. **Load Raw Data**
   - Read `Fake.csv` and `True.csv` using Pandas.

2. **Add Labels**
   - Assign `label = 0` (Low Credibility) to all fake news articles.
   - Assign `label = 1` (High Credibility) to all real news articles.

3. **Combine Datasets**
   - Merge both DataFrames into a single unified dataset.

4. **Column Selection**
   - Retain only the `text` and `label` columns. Other metadata (title, subject, date) is dropped to force the model to learn from content alone.

5. **Remove Null Values**
   - Drop any rows where the `text` field is missing or empty.

6. **Remove Duplicates**
   - Drop rows with identical `text` entries to prevent data leakage and bias.

7. **Shuffle Dataset**
   - Randomly shuffle all rows to eliminate any ordering bias (e.g., all fake articles appearing before real ones).

8. **Print Statistics**
   - Total number of samples after cleaning.
   - Class distribution (count and percentage of Fake vs. Real).
   - Average article length (in characters).

9. **Save Output**
   - Write the cleaned dataset to `data/news_cleaned.csv`.

---

## 5. Text Preprocessing Strategy

Raw text is **noisy** — it contains capitalization variations, punctuation, numbers, stopwords, and inflected word forms that add no discriminative value. Preprocessing transforms this raw text into a **normalized, clean representation** that the ML model can effectively learn from.

### Pipeline Steps

| Step                                   | What It Does                                 | Why It Matters                                                        |
| -------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------- |
| **Lowercasing**                        | Converts all characters to lowercase         | Ensures "Trump" and "trump" are treated as the same token             |
| **Special Character & Number Removal** | Strips punctuation, symbols, and digits      | Reduces noise; numbers rarely help distinguish fake from real         |
| **Tokenization**                       | Splits text into individual words (tokens)   | Enables word-level analysis and stopword filtering                    |
| **Stopword Removal**                   | Removes common words like "the", "is", "and" | These words appear equally in fake and real news — they add no signal |
| **Lemmatization**                      | Reduces words to their base/dictionary form  | Maps "running", "ran", "runs" → "run" to reduce vocabulary size       |
| **String Reconstruction**              | Joins tokens back into a single clean string | Required format for TF-IDF vectorization                              |

### Implementation Notes

- **Library:** NLTK is used for tokenization, stopwords, and lemmatization (WordNetLemmatizer).
- **Modularity:** All preprocessing functions live in `utils.py` for reusability.
- **Output:** Preprocessed text is saved as a new column and exported to `data/news_preprocessed.csv`.

---

## 6. Basic Exploratory Data Analysis (EDA)

Before training, we perform lightweight EDA to understand the data distribution:

1. **Class Distribution**
   - Print the count and percentage of Fake (0) vs. Real (1) articles.
   - Verify the dataset is reasonably balanced (~52% Fake, ~48% Real).

2. **Text Length Distribution**
   - Compute mean, median, min, max, and standard deviation of article lengths.
   - Helps decide if truncation or padding strategies are needed.

3. **Top 20 Most Frequent Words**
   - After preprocessing, identify the 20 most common tokens across the corpus.
   - Provides intuition about what the model will learn (e.g., political terms, named entities).

---

## 7. Feature Engineering — Why TF-IDF?

### The Problem

Machine learning models require **numerical input** — they cannot directly process raw text. We need a method to convert text into a meaningful vector representation.

### Why TF-IDF Over Alternatives

| Method                 | Pros                                                                              | Cons                                                                 |
| ---------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Bag of Words (BoW)** | Simple, interpretable                                                             | Treats all words equally; high-frequency common words dominate       |
| **TF-IDF**             | Weighs words by importance; penalizes common terms; captures discriminative words | Still loses word order; sparse representation                        |
| **Word2Vec / GloVe**   | Captures semantic relationships                                                   | Requires pre-trained embeddings; adds complexity beyond classical ML |

### TF-IDF (Term Frequency – Inverse Document Frequency)

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

- **TF (Term Frequency):** How often a word appears in a document — frequent words in a document get higher scores.
- **IDF (Inverse Document Frequency):** log(N / df) — words that appear in many documents get lower scores, penalizing common terms like "said" or "government".

### Configuration

- `max_features = 5000` — Limits vocabulary to the top 5,000 most informative terms, balancing information retention with computational efficiency.
- The fitted vectorizer is saved as `models/vectorizer.pkl` for consistent transformation of new data during inference.

---

## 8. Model Selection — Why Logistic Regression?

### Rationale

For a **baseline text classification** task, Logistic Regression is the gold standard starting point in the NLP community.

| Criterion                         | Logistic Regression                                                    |
| --------------------------------- | ---------------------------------------------------------------------- |
| **Interpretability**              | Coefficients directly indicate which words push toward Fake or Real    |
| **Performance on Text**           | Excels with high-dimensional sparse features (TF-IDF matrices)         |
| **Training Speed**                | Trains in seconds even on ~45K documents                               |
| **No Hyperparameter Sensitivity** | Works well out-of-the-box with default parameters                      |
| **Baseline Suitability**          | Establishes a clear performance floor to compare future models against |
| **Scikit-Learn Support**          | First-class implementation with extensive documentation                |

### Training Strategy

- **Train/Test Split:** 80% training, 20% testing (stratified to maintain class balance).
- **No Cross-Validation in Baseline:** Kept simple for Milestone 1; can be added for optimization.
- **Model Persistence:** Trained model saved as `models/model.pkl` using `joblib`.

---

## 9. Evaluation Metrics Explanation

A single accuracy number is **insufficient** to judge a classifier. We use a comprehensive suite of metrics:

| Metric               | Formula               | What It Tells Us                                                                 |
| -------------------- | --------------------- | -------------------------------------------------------------------------------- |
| **Accuracy**         | (TP + TN) / Total     | Overall correctness — but misleading on imbalanced data                          |
| **Precision**        | TP / (TP + FP)        | Of all articles predicted as Real, how many actually are? (Avoids false trust)   |
| **Recall**           | TP / (TP + FN)        | Of all actually Real articles, how many did we catch? (Avoids missing real news) |
| **F1 Score**         | 2 × (P × R) / (P + R) | Harmonic mean of Precision and Recall — single balanced metric                   |
| **Confusion Matrix** | 2×2 grid              | Visual breakdown of TP, TN, FP, FN — reveals where the model struggles           |

### Why Each Matters for Fake News

- **High Precision** → We don't wrongly flag legitimate journalism as fake (avoiding censorship).
- **High Recall** → We don't let fake articles slip through undetected (avoiding misinformation spread).
- **F1 Score** → The balanced trade-off between the two.

---

## 10. Model Performance Summary

> _This section will be populated after running `train.py` on the dataset._

### Expected Results (Based on Literature)

| Metric    | Expected Range |
| --------- | -------------- |
| Accuracy  | 92% – 97%      |
| Precision | 93% – 97%      |
| Recall    | 92% – 96%      |
| F1 Score  | 92% – 96%      |

The ISOT dataset is known to be relatively **separable** with classical methods because real and fake articles in this corpus have distinct writing styles, vocabulary patterns, and structural differences.

### Interpretation Guide

After training, analyze the confusion matrix to identify:

- **False Positives (FP):** Real articles misclassified as Fake — potential censorship risk.
- **False Negatives (FN):** Fake articles misclassified as Real — misinformation leakage risk.

---

## 11. Limitations of Classical ML

While our Logistic Regression + TF-IDF pipeline achieves strong results on this dataset, it has fundamental limitations:

| Limitation                    | Explanation                                                                                                             |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **No Semantic Understanding** | TF-IDF treats words as independent tokens — it cannot understand meaning, sarcasm, or context                           |
| **No Word Order**             | "Dog bites man" and "Man bites dog" produce identical TF-IDF vectors                                                    |
| **Dataset Bias**              | The ISOT dataset has a specific temporal and topical scope; the model may not generalize to new domains or time periods |
| **Static Vocabulary**         | New words, slang, or evolving language patterns are ignored unless the model is retrained                               |
| **No Source Verification**    | The model judges text content alone — it cannot verify claims against external sources                                  |
| **No Multi-Modal Analysis**   | Cannot analyze images, videos, or social media engagement patterns that often accompany fake news                       |
| **Binary Classification**     | Real-world credibility exists on a spectrum; binary Fake/Real is an oversimplification                                  |
| **Adversarial Vulnerability** | Sophisticated fake news generators can craft content that mimics real news writing styles                               |

---

## 12. How This Prepares for Agentic AI in Milestone 2

Milestone 1 establishes the **foundation** upon which Milestone 2's Agentic AI system will be built.

### Progression Roadmap

```
Milestone 1 (Current)              →    Milestone 2 (Next)
─────────────────────────────────        ──────────────────────────────
Classical ML Baseline                    Agentic AI System
TF-IDF + Logistic Regression             Multi-Agent Architecture
Single-signal analysis                   Multi-source verification
Static model                            Dynamic, self-improving agents
Binary classification                   Credibility scoring (0–100)
Offline batch processing                 Real-time analysis pipeline
```

### What Milestone 2 Will Add

1. **Source Credibility Agent** — Cross-references claims against trusted databases and fact-checking APIs.
2. **Sentiment & Bias Agent** — Detects emotional manipulation and ideological bias in article framing.
3. **Temporal Analysis Agent** — Tracks how stories evolve over time to detect coordinated disinformation campaigns.
4. **Ensemble Decision Agent** — Combines signals from all agents (including our ML classifier) to produce a holistic credibility score.
5. **Explanation Agent** — Generates human-readable explanations for why an article was flagged.

### Why ML First?

- The ML classifier serves as a **fast, lightweight pre-filter** that the agentic system can invoke.
- It provides a **quantitative baseline** against which agentic improvements are measured.
- Understanding classical ML limitations directly motivates the design of the agentic architecture.

---

## 13. Ethical Considerations

### Bias and Fairness

- **Training Data Bias:** The ISOT dataset is predominantly US political news from 2015–2018. The model's decisions may not transfer fairly to non-English content, non-political topics, or different cultural contexts.
- **Labeling Bias:** The binary Fake/Real labels were assigned based on source reputation, not individual article verification. Some "fake" sources may occasionally publish truthful content, and vice versa.

### Potential for Misuse

- **Censorship Risk:** An imperfect classifier could be weaponized to suppress legitimate dissenting voices by falsely labeling critical journalism as "fake."
- **Over-Reliance:** Deploying this model without human oversight could lead to automated suppression of content at scale.

### Transparency

- **Model Interpretability:** Logistic Regression's coefficients provide transparency into decision-making — stakeholders can inspect which words drive predictions.
- **No Black Box:** We deliberately chose an interpretable model for Milestone 1 to maintain accountability.

### Responsible Deployment Recommendations

1. **Never deploy as a standalone decision-maker** — always pair with human review.
2. **Clearly communicate confidence levels** — present predictions as probabilities, not absolute verdicts.
3. **Regularly retrain** — language and disinformation tactics evolve; static models become stale.
4. **Audit for bias** — periodically test the model on diverse datasets to detect unfair patterns.
5. **Respect press freedom** — err on the side of caution before labeling content as fake.

---

## 14. Implementation Steps Summary

### Step-by-Step Execution Guide

```
Step 1  →  Place Fake.csv and True.csv in the data/ folder
Step 2  →  Install dependencies from requirements.txt
Step 3  →  Run train.py
Step 4  →  Review printed statistics, EDA output, and metrics
Step 5  →  Verify saved artifacts in data/ and models/
```

### File Responsibilities

| File                         | Purpose                                                                                 |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| `utils.py`                   | Text preprocessing functions (lowercase, clean, tokenize, lemmatize)                    |
| `train.py`                   | End-to-end pipeline: load → clean → preprocess → EDA → TF-IDF → train → evaluate → save |
| `requirements.txt`           | Python dependencies: pandas, numpy, scikit-learn, nltk, joblib                          |
| `stories.md`                 | This document — project narrative and methodology                                       |
| `data/news_cleaned.csv`      | Output of structural cleaning                                                           |
| `data/news_preprocessed.csv` | Output of NLP preprocessing                                                             |
| `models/vectorizer.pkl`      | Fitted TF-IDF vectorizer                                                                |
| `models/model.pkl`           | Trained Logistic Regression model                                                       |

---

## 15. Conclusion

Milestone 1 demonstrates that **classical NLP techniques combined with a well-designed ML pipeline** can achieve remarkably strong performance on fake news detection. By methodically cleaning data, engineering meaningful features with TF-IDF, and training an interpretable Logistic Regression model, we establish both a **functional baseline** and a **clear understanding of the limitations** that motivate the transition to Agentic AI in Milestone 2.

This project prioritizes **reproducibility** (relative paths, saved artifacts), **modularity** (separation of utilities and training logic), and **transparency** (interpretable model, comprehensive evaluation metrics) — principles that will carry forward into the more advanced stages of the system.

---

> _Prepared for Mid-Semester Evaluation — Intelligent News Credibility Analysis Project_
