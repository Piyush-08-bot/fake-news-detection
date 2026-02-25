# Project Detail Notes — Fake News Detection Using Machine Learning

### Machine Learning Pipeline

The project uses a Jupyter Notebook for the entire machine learning lifecycle:

- **Notebook**: [News_Credibility_Training.ipynb](News_Credibility_Training.ipynb)

#### Training Steps:

1. Open `News_Credibility_Training.ipynb` in your preferred environment (JupyterLab, VS Code, or Google Colab).
2. Run all cells sequentially.
3. The notebook will:
   - Preprocess the raw data from `data/news_cleaned.csv`.
   - Train a Logistic Regression model (primary) and a Decision Tree.
   - Save the trained models and vectorizers to the `models/` directory.

#### Deployment:

The Streamlit application (`app.py`) automatically loads the model and vectorizer from the `models/` folder. It uses `utils.py` for real-time preprocessing of user input.

---

## 0. DATA PREPARATION (Done on Google Colab)

> **Note:** This step was performed separately on Google Colab before the main training pipeline was run locally. The output of this step is `news_cleaned.csv`, which is what `train.py` loads.

**What was done on Colab (step by step):**

### Step 1: Load the Raw CSV Files

```python
import pandas as pd

fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')
```

- The ISOT dataset comes as two separate files: `Fake.csv` (fake articles) and `True.csv` (real articles).
- Initial shape: `Fake.csv` had ~23,000 rows, `True.csv` had ~21,000 rows.
- Both had 5 columns: `title`, `text`, `subject`, `date`, `label` (after labelling).

### Step 2: Assign Labels

```python
fake["label"] = 0   # 0 = Fake
real["label"] = 1   # 1 = Real
```

- Manually added the binary label column to both DataFrames since the raw ISOT files don't include it.
- `0` = Fake News, `1` = Real News.

### Step 3: Merge into One Dataset

```python
df = pd.concat([fake, real], axis=0)
print("Combined shape:", df.shape)
df.head()
```

- Both DataFrames stacked vertically using `pd.concat`.
- Combined shape: **(44,898 rows, 5 columns)**.

### Step 4: Keep Only Relevant Columns

```python
df = df[["text", "label"]]
df.head()
```

- Dropped `title`, `subject`, and `date` — only `text` and `label` are needed for TF-IDF classification.
- Model classifies purely based on article body text.

### Step 5: Drop Null Values

```python
print("Before dropna:", df.shape)
df.dropna(inplace=True)
print("After dropna:", df.shape)
```

- Removed any rows with missing text.
- Result: **44,898 → 44,898** (no nulls found in this dataset).

### Step 6: Remove Duplicate Articles

```python
print("Before removing duplicates:", df.shape)
df.drop_duplicates(subset=["text"], inplace=True)
print("After removing duplicates:", df.shape)
```

- Removed rows where the article body text was identical.
- Result: **44,898 → 38,646 unique articles** (6,252 duplicates removed).
- This is why the final dataset used in training has **38,646 samples**.

### Step 7: Shuffle the Dataset

```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()
```

- Randomly shuffled all rows so that fake and real articles are interleaved.
- `random_state=42` ensures the shuffle is reproducible.
- `reset_index(drop=True)` resets row numbers from 0 after shuffling.
- **Why shuffle?** If the dataset is ordered (all fakes first, all reals last), the train-test split could accidentally put mostly one class in the test set.

### Step 8: Save Cleaned Data and Download

```python
df.to_csv("news_cleaned.csv", index=False)
print("Dataset cleaned and saved successfully")

from google.files import files
files.download("news_cleaned.csv")
```

- Saved the final cleaned, shuffled, deduplicated dataset as `news_cleaned.csv`.
- Downloaded it from Colab to local machine.
- This file was then placed in the `data/` folder of the local project for use by `train.py`.

**Summary of what Colab produced:**

| Step            | Before             | After                                         |
| --------------- | ------------------ | --------------------------------------------- |
| Load & merge    | Two separate files | 44,898 rows                                   |
| Drop nulls      | 44,898             | 44,898 (no change)                            |
| Drop duplicates | 44,898             | **38,646**                                    |
| Final output    | —                  | `news_cleaned.csv` with `text` + `label` only |

---

**What is fake news detection?**

- Fake news refers to intentionally fabricated or misleading information presented as legitimate journalism.
- Fake news detection is the process of automatically classifying a piece of text as either credible or fabricated using computational methods.

**Why is it important?**

- Misinformation spreads faster than corrections on social media.
- It can influence elections, public health behaviour, stock markets, and social tensions.
- Manual fact-checking cannot scale to the billions of articles and posts published every day.
- Automated ML-based detection provides scalable, consistent, low-latency screening.

**How can ML detect fake news?**

- Real news and fake news have measurably different linguistic patterns.
- Real news tends to cite sources, use formal attribution language, and reference verifiable entities.
- Fake news tends to use emotionally charged, vague, opinionated language targeting political figures.
- ML models learn these statistical word patterns from large labelled datasets and generalize them to classify new unseen articles.

---

## 2. DATASET DETAILS

**Dataset name:** ISOT Fake News Dataset  
**Total articles:** 38,646  
**Labels:**

- `0` = Fake News (articles from unreliable sources, misinformation sites)
- `1` = Real News (articles from Reuters, credible mainstream journalism)

**What does the dataset contain?**

- Two separate CSV files originally: one for real news, one for fake news.
- Each row = one news article with its full body text and a label.
- Cleaned and merged into a single `news_cleaned.csv` with `text` and `label` columns.

**Why is this dataset suitable?**

- Large volume (>38k samples) — enough for a machine learning model to generalize.
- Balanced classes (roughly equal real vs fake) — avoids class imbalance bias.
- Text-rich — each article contains full body text, not just headlines.
- Well-labelled ground truth from credible sources (Reuters) and known fake news sites.

---

## 2b. FEATURE SELECTION STRATEGY

**Original dataset columns:**

| Column    | Description                                |
| --------- | ------------------------------------------ |
| `title`   | Headline of the article                    |
| `text`    | Full article body content                  |
| `subject` | News category (Politics, World News, etc.) |
| `date`    | Publication date                           |
| `label`   | Target class (0 = Fake, 1 = Real)          |

**Only `text` and `label` were kept:**

```python
df = df[["text", "label"]]
```

- `text` → Feature (X) — what the model reads
- `label` → Target (y) — what the model learns to predict

**Why the other columns were dropped:**

- **`title`:** Useful, but the full `text` already contains all meaningful signals. Excluded for simplicity in the baseline model. (Can be concatenated in a future improvement.)
- **`subject`:** Removed to prevent **data leakage**. If fake articles disproportionately belong to one subject, the model would learn category shortcuts instead of linguistic patterns, reducing generalization.
- **`date`:** Removed to prevent **temporal bias**. If most fake articles come from 2016–2017, the model might learn time patterns rather than content patterns.

**Core principle applied:** The model must evaluate credibility based solely on article content — not metadata. This ensures the system generalizes to new articles from any year, any category.

**Advantages:**

- Prevents overfitting to metadata
- Reduces noise in feature space
- Keeps deployment simple (just raw text input)
- Aligns with standard NLP classification practices

---

## 3. TEXT PREPROCESSING

**Why preprocess at all?**

- Raw text contains noise, punctuation, inconsistencies, and stopwords.
- ML models work on numbers, not raw text. Preprocessing standardizes text so TF-IDF counts are meaningful.
- Without preprocessing, "Washington" and "washington" would be counted as two different words.

**Step-by-step breakdown of `preprocess_text()` in `utils.py`:**

### Step 1: Lowercasing

- Converts `"Reuters"` → `"reuters"`, `"SAID"` → `"said"`.
- **Why:** Avoids treating the same word as two different features based on capitalization.

### Step 2: Removing Special Characters

- Strips punctuation, symbols, numbers, HTML tags (e.g., `"#!&*"` → `""`).
- **Why:** These characters carry no semantic meaning for classification. Numbers are unhelpful without context.

### Step 3: Tokenization

- Splits the article string into individual word tokens.
- e.g., `"trump said today"` → `["trump", "said", "today"]`
- **Why:** ML models operate word by word. Sentences must be split into individual units first.

### Step 4: Stopword Removal

- Removes common English words that are grammatically necessary but semantically empty.
- Examples: `"the", "is", "a", "and", "of", "it"` are removed.
- Uses NLTK's predefined English stopword list.
- **Why:** Stopwords appear in every document and do not help distinguish real from fake news. Removing them reduces noise and dimensionality.

### Step 5: Lemmatization

- Reduces words to their base dictionary form.
- `"running"` → `"run"`, `"said"` → `"say"`, `"countries"` → `"country"`
- Uses NLTK's `WordNetLemmatizer`.
- **Why:** Treats inflected forms of the same word as one feature. This reduces vocabulary size and improves generalization.

**Final output of `preprocess_text()`:**  
A cleaned lowercase string with only meaningful, normalized words. This goes into TF-IDF vectorization.

---

## 4. NLP THEORY

**What is NLP (Natural Language Processing)?**

- NLP is a branch of AI that deals with teaching computers to understand, process, and analyze human language.

**Why is NLP needed for this project?**

- Raw text is unstructured data. ML models only work with numbers.
- NLP provides the bridge: it converts text into numerical representations that carry meaning.

**How do machines understand text?**

- Machines cannot "read" the way humans do.
- Instead, they count occurrences of specific words and assign numerical weights to them.
- The model learns that certain word patterns (counted as numbers) correlate strongly with one class label (Real or Fake).
- No actual semantic understanding — the model is performing statistical pattern recognition over word frequency vectors.

---

## 5. TF-IDF THEORY

**What is TF-IDF?**  
TF-IDF stands for **Term Frequency – Inverse Document Frequency**. It is a numerical statistic that reflects how important a word is to a document in a collection (corpus).

### Term Frequency (TF)

- Measures how often a word appears in a single article.
- Formula: `TF(word, doc) = count(word in doc) / total words in doc`
- Example: If "reuters" appears 3 times in a 300-word article → TF = 0.01

### Inverse Document Frequency (IDF)

- Measures how rare a word is across all articles.
- Formula: `IDF(word) = log(total docs / docs containing word)`
- If "reuters" appears in only 5 out of 38,646 articles → IDF is high (it's rare and informative).
- If "said" appears in 30,000 articles → IDF is lower.

### TF-IDF Score

- Formula: `TF-IDF = TF × IDF`
- Words that are frequent in one document but rare across all documents get high TF-IDF scores.
- These are the words that make one document unique — the most useful signals for classification.

**Why use TF-IDF?**

- Plain word counting (Bag of Words) treats every word equally. TF-IDF up-weights rare, specific, informative words.
- Words like "the" appear everywhere — TF-IDF suppresses them automatically via low IDF.
- Words like "reuters" appear specifically in real news → TF-IDF captures this as a strong signal.

**Why `max_features=5000`?**

- The vocabulary of 38,646 articles could contain 100,000+ unique words.
- Using all of them would be computationally expensive and would include useless rare words (typos, proper nouns).
- `max_features=5000` keeps only the top 5000 most frequently occurring words by document frequency.
- This reduces noise, speeds up training, and doesn't significantly hurt accuracy.

**How text becomes numbers:**

- Each article becomes a **vector of 5000 numbers**.
- Each number represents the TF-IDF score of one specific word.
- If a word is absent → 0. If present → computed TF-IDF score.

---

## 6. FEATURES

**What is a feature in ML?**

- A feature is one measurable input signal fed into the model.
- For images, features = pixel values. For this project, features = TF-IDF word scores.

**Words as features:**

- Each of the 5000 selected words is exactly one feature.
- Feature 1 might be the word `"reuters"`, Feature 2 might be `"said"`, etc.

**Feature vector example for one article:**

```
[0.0, 0.85, 0.0, 0.0, 0.12, ..., 0.0]  → 5000 numbers total
         ↑               ↑
       "reuters"        "washington"
```

**Why are these specific words important?**

| Word         | Signal | Why                                                                  |
| ------------ | ------ | -------------------------------------------------------------------- |
| `reuters`    | → Real | Main credible news agency. Appears only in real articles.            |
| `said`       | → Real | Journalistic attribution. Real news quotes sources.                  |
| `washington` | → Real | Common in geopolitical real journalism.                              |
| `image`      | → Fake | Fake news often steals images and credits "image via Twitter".       |
| `hillary`    | → Fake | Fake news heavily used political names as emotional engagement bait. |
| `gop`        | → Fake | Partisan label overused in fake political content.                   |

---

## 7. MACHINE LEARNING MODELS

### Logistic Regression

**How it works:**

- Despite the name, Logistic Regression is a **classification** model, not a regression model.
- It takes the 5000 TF-IDF feature values as input.
- It multiplies each feature by a learned **coefficient** (weight): `score = w1*x1 + w2*x2 + ... + w5000*x5000`
- The score is then passed through a **sigmoid function** which squashes it into a probability between 0 and 1.
- `P(Real) = sigmoid(score) = 1 / (1 + e^(-score))`

**Probabilities:**

- If P(Real) > 0.5 → the model predicts **Real News**.
- If P(Real) < 0.5 → the model predicts **Fake News**.

**Coefficients:**

- The model learns a coefficient (weight) for each of the 5000 words.
- A large **positive** coefficient means the word strongly pushes toward **Real News**.
- A large **negative** coefficient means the word strongly pushes toward **Fake News**.
- `"reuters"` → +23 coefficient → extremely strong Real signal.
- `"via"` → -10 coefficient → strong Fake signal.

**Decision boundary:**

- The imaginary line in the feature space where `score = 0`, i.e., where `P = 0.5`.
- Points on one side = Real. Points on the other = Fake.

---

### Decision Tree

**How it works:**

- A Decision Tree learns a series of YES/NO questions (splits) about feature values.
- Starting from the root node, at each node it asks: "Is TF-IDF of word X greater than Y?"
- It splits articles into two branches at each node.
- This continues down the tree until a **leaf node** is reached.
- A leaf node gives the final classification: Fake or Real.

**Splits (nodes):**

- Each internal node = one question about one feature (word).
- The tree learns which question at each step best separates Real from Fake (measured by Gini Impurity or Information Gain).

**Leaves:**

- Terminal nodes. No more questions. Just outputs `0` (Fake) or `1` (Real) based on the majority class of training samples that fell into that leaf.

---

### Key Differences

| Property         | Logistic Regression     | Decision Tree                |
| ---------------- | ----------------------- | ---------------------------- |
| Model type       | Linear                  | Nonlinear                    |
| Outputs          | Probabilities (0 to 1)  | Hard class labels            |
| Interpretability | Coefficients per word   | Tree path visualization      |
| Overfitting risk | Lower                   | Higher (without pruning)     |
| Speed            | Faster                  | Slower on large vocabs       |
| Best for         | Linearly separable text | Complex nonlinear boundaries |

In this project, **Logistic Regression generally outperforms Decision Tree** on text classification because the decision boundary between Real and Fake is well-captured by a linear function over weighted TF-IDF features.

---

## 8. PIPELINE

**What is a Scikit-learn Pipeline?**

- A `Pipeline` chains multiple processing steps into one single object.
- The output of each step automatically becomes the input of the next step.

**This project's pipelines:**

```
Pipeline 1: TF-IDF Vectorizer → Logistic Regression
Pipeline 2: TF-IDF Vectorizer → Decision Tree
```

**How it flows:**

1. Raw preprocessed text string fed in.
2. `TfidfVectorizer` converts it into a 5000-element numerical vector.
3. That vector is directly passed into the classifier.
4. Classifier outputs a prediction (0 or 1).

**Why pipelines are useful:**

- Prevents **data leakage**: TF-IDF learns its vocabulary from training data only, not test data.
- The fitted pipeline can be saved as one `.pkl` file — no need to save the vectorizer and model separately (though we do both for backup).
- Simplifies deployment: at inference time, `pipeline.predict(raw_text)` handles everything end-to-end.

---

## 9. TRAINING PROCESS (Step by Step)

1. **Load Dataset:** Read `data/news_cleaned.csv` into a DataFrame.
2. **Preprocess Text:** Apply `preprocess_text()` to every article — lowercase, clean, tokenize, remove stopwords, lemmatize.
3. **Save Preprocessed Data:** Write to `data/news_preprocessed.csv`.
4. **EDA:** Print text length stats and top 20 most frequent words.
5. **Train-Test Split:** Split cleaned text and labels 80/20.
6. **Train Pipeline LR:** Fit `TfidfVectorizer + LogisticRegression` on training data.
7. **Evaluate LR:** Predict on test set, compute Accuracy, Precision, Recall, F1, Confusion Matrix.
8. **Train Pipeline DT:** Fit `TfidfVectorizer + DecisionTreeClassifier` on training data.
9. **Evaluate DT:** Predict on test set, same metrics.
10. **Save Best Model:** `pipeline_lr` saved as `models/model.pkl`. TF-IDF vectorizer saved as `models/vectorizer.pkl`.

---

## 10. TRAIN-TEST SPLIT

- `train_test_split(X, y, test_size=0.2, random_state=42)`
- **80% of data (30,916 samples)** → used for training. The model learns from these.
- **20% of data (7,730 samples)** → used for testing. The model never sees these during training.

**Why split at all?**

- To measure how well the model **generalizes** to new, unseen data.
- If you test on the same data used for training, the model has technically "memorized" it — this inflates accuracy and is called **overfitting**.
- A held-out test set is an honest evaluation of real-world performance.

**Why 80/20?**

- A standard convention. Gives the model enough samples to learn from (80%) while also providing a statistically significant test set (20%).

**Why `random_state=42`?**

- Ensures the split is reproducible. The same 80% and 20% subsets are produced every time the code is run.

---

## 11. EVALUATION METRICS

### Accuracy

- Percentage of all predictions that are correct.
- Formula: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **Limitation:** Can be misleading on imbalanced datasets (e.g., if 95% of data is Real, always predicting Real gives 95% accuracy but is useless).

### Precision

- Of all articles the model predicted as Real, what fraction actually was Real?
- Formula: `Precision = TP / (TP + FP)`
- High Precision = few false alarms. Important when false positives are costly.

### Recall

- Of all actually Real articles, what fraction did the model correctly identify?
- Formula: `Recall = TP / (TP + FN)`
- High Recall = few missed detections. Important when missing a positive case is costly.

### F1 Score

- The harmonic mean of Precision and Recall.
- Formula: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- Balances both Precision and Recall into a single number. Best metric when you care equally about both.

### Confusion Matrix

```
                  Predicted Fake    Predicted Real
Actual Fake           TN                FP
Actual Real           FN                TP
```

- **TP (True Positive):** Correctly predicted Real.
- **TN (True Negative):** Correctly predicted Fake.
- **FP (False Positive):** Predicted Real but actually Fake (misclassification).
- **FN (False Negative):** Predicted Fake but actually Real (missed real article).

---

## 12. FEATURE IMPORTANCE

**How words influence prediction:**

- After training, Logistic Regression has learned one coefficient (weight) for each of the 5000 TF-IDF words.
- These coefficients directly encode how much each word pushes the prediction toward Real (+) or Fake (-).

**Positive coefficients → lean toward Real News:**

- `reuters (+23)` — strongly signals credible journalism.
- `said (+18)` — formal attribution used in journalism: "The official said..."
- `washington (+6)` — tied to credible geopolitical reporting.
- `tuesday, wednesday, thursday (+5)` — real news is time-stamped precisely.

**Negative coefficients → lean toward Fake News:**

- `via (-10)` — fake news often says "image via Twitter" or "story via Facebook".
- `image (-7)` — heavy reliance on image-based content instead of original text.
- `gop, hillary, obama (-5)` — politically charged bait words heavily overused in misinformation content.
- `even, like (-4)` — casual, informal, opinionated tone vs. formal journalistic language.

**Coefficient values:**

- The magnitude (size) of the coefficient indicates strength of signal.
- A word with coefficient `+23` has far more influence than one with `+2`.
- If the word is absent in an article, its contribution = `0 × coefficient = 0` (no influence).

---

## 13. MODEL SAVING

**What is `joblib`?**

- A Python library specifically designed for efficiently serializing and deserializing large NumPy arrays and Scikit-learn objects.
- Preferred over `pickle` for ML models due to better performance with large numerical arrays.

**Why save models?**

- Training on 38,000 articles takes minutes. You don't want to re-train every time someone uses the app.
- A saved model can be loaded instantly and used for inference in milliseconds.

**Saved files:**

| File                           | Contents                             | Used for                                 |
| ------------------------------ | ------------------------------------ | ---------------------------------------- |
| `models/model.pkl`             | Full `pipeline_lr` (TF-IDF + LR)     | Main prediction in `app.py`              |
| `models/vectorizer.pkl`        | Just the `TfidfVectorizer` component | Inspecting features, visualizing weights |
| `models/pipeline_lr_tfidf.pkl` | Same as model.pkl (backup)           | Redundancy                               |
| `models/pipeline_dt_tfidf.pkl` | Full `pipeline_dt` (TF-IDF + DT)     | Comparison / alternative                 |

**How loading works in `app.py`:**

```python
import joblib
model = joblib.load('models/model.pkl')
prediction = model.predict([preprocessed_article])
```

The entire TF-IDF transformation + classification happens inside one `model.predict()` call.

---

## 14. WORKFLOW

```
Dataset (news_cleaned.csv)
        ↓
  Text Preprocessing (utils.py: preprocess_text)
  [lowercase → clean → tokenize → remove stopwords → lemmatize]
        ↓
  TF-IDF Vectorization (5000 features)
  [text string → vector of 5000 numbers]
        ↓
  Model Training
  [Logistic Regression / Decision Tree pipeline learns coefficients]
        ↓
  Model Evaluation
  [Accuracy, Precision, Recall, F1 on 20% held-out test set]
        ↓
  Best Model Saved (models/model.pkl)
        ↓
  Streamlit App (app.py)
  [User pastes article → same preprocessing → pipeline.predict() → Real or Fake]
```

---

## 15. LIMITATIONS

**1. Only text-based:**

- The model uses only the body text of the article. It has zero knowledge of:
  - Author identity
  - Source domain credibility
  - Publication date
  - Social media engagement / virality
  - Images, video, or multimedia
- A completely text-identical article from Reuters vs a no-name blog would get the same prediction.

**2. Dataset bias:**

- The ISOT dataset contains articles from a specific time period (mainly 2016–2017, US political context).
- The model is heavily biased toward the vocabulary of that era and political climate.
- Words like `"hillary"` appearing as Fake signals reflect the training dataset's context, not universal truth.
- A news article about a non-political topic from 2020 might not be evaluated accurately.

**3. Limited vocabulary:**

- Only 5000 words are in the model's vocabulary.
- Any word not in those 5000 features is completely ignored at prediction time (TF-IDF assigns it 0).
- This means new slang, recently coined terms, or domain-specific words outside the vocabulary have zero influence.

**4. Classical NLP weaknesses:**

- TF-IDF treats each word in isolation. It has no understanding of word order, context, or semantics.
- `"not good"` and `"good"` might have similar TF-IDF representations despite opposite meanings.
- Sarcasm, irony, and subtle misinformation are impossible for this model to detect.

---
