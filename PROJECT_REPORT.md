<div align="center">

---

# 🎓 Rishihood University
## Department of Computer Science

<br>

# Intelligent Fake News Detection & AI Reasoning System
### *Using Natural Language Processing, Machine Learning, and Agentic AI*

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
📁 **GitHub (Vikash):** [github.com/devVIKASHk/fake-news-detection](https://github.com/devVIKASHk/fake-news-detection)<br>
📁 **GitHub (Piyush):** [github.com/Piyush-08-bot/fake-news-detection](https://github.com/Piyush-08-bot/fake-news-detection)

</div>

---

## 📋 Table of Contents

| # | Section |
|:---:|:---|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Data Description](#2-data-description) |
| 3 | [Exploratory Data Analysis](#3-exploratory-data-analysis) |
| 4 | [Phase 1: Classical ML Methodology](#4-phase-1-classical-ml-methodology) |
| 5 | [ML Training & Evaluation](#5-ml-training--evaluation) |
| 6 | [Phase 2: Agentic AI Reasoning Layer](#6-phase-2-agentic-ai-reasoning-layer) |
| 7 | [Application Architecture](#7-application-architecture) |
| 8 | [Conclusion](#8-conclusion) |
| 9 | [Team Contribution](#9-team-contribution) |

---

## 1. Problem Statement

Fake news is rampant across the web, spreading far faster than manual human fact-checking can mitigate. While earlier efforts in NLP have successfully built classification algorithms that detect clickbait patterns, these models suffer from a "black box" problem: **they can tell you if a news article is fake, but they cannot explain *why*, nor can they verify facts.**

The core challenge we tackled was building a two-phase hybrid system:
1. **The Triage Engine**: Can a lightweight machine learning system process article text in milliseconds and flag linguistic anomalies?
2. **The Investigator**: Can an advanced AI Agent parse flagged articles, break them into conceptual claims, verify those claims against the live web, and provide human-readable explanations?

**Our goals were to:**
- ✅ Train a highly accurate classical ML model to detect fake news language patterns.
- ✅ Build a LangGraph state machine to orchestrate LLM reasoning.
- ✅ Connect the agent to the live internet for active fact-checking.
- ✅ Deliver an interactive web dashboard providing extreme transparency to the user.

---

## 2. Data Description (For ML Training)

### 2.1 Dataset Source

We used the **ISOT Fake News Dataset**, created by the Information Security and Object Technology (ISOT) Lab at the University of Victoria to train our Phase 1 model.

| Source | Description |
|:---|:---|
| **Real News** | Scraped from **Reuters.com** — a globally trusted wire news service |
| **Fake News** | Collected from sites flagged by **PolitiFact** and other fact-checkers |

### 2.2 Dataset Size

| Class | Count | Share |
|:---|:---:|:---:|
| ✅ Real Articles | 21,417 | **55.4%** |
| ❌ Fake Articles | 17,229 | **44.6%** |
| **Total** | **38,646** | **100%** |

---

## 3. Exploratory Data Analysis

Before building any model, we explored the text structure. Our primary insight was that **Real articles** showed consistent article lengths (following editorial guidelines), whereas **Fake articles** ranged from ultra-short engaging clickbait to excessively long conspiracy posts.

### Word Signals by Class

Our TF-IDF vectorization revealed two distinct vocabulary clusters:

| Signal | Words |
|:---|:---|
| 🟢 **Real News words** | `washington`, `official`, `statement`, `government` |
| 🔴 **Fake News words** | `video`, `image`, `share`, `watch`, emotional & sensational terms |

Credible journalism uses attribution language neutrally, while fake articles rely heavily on engagement-bait phrasing.

---

## 4. Phase 1: Classical ML Methodology

Raw text was preprocessed sequentially (lowercased, stripped of punctuation, tokenized, and lemmatized).

We converted each article into a dense mathematical vector using **TF-IDF (Term Frequency–Inverse Document Frequency)** capped at `5,000` features to reduce noise. We then trained a **Logistic Regression** classifier over a Decision Tree because the regression model produced smoother, non-overfitted output probabilities and superior generalisation on unseen data.

---

## 5. ML Training & Evaluation

Our Logistic Regression model was trained on 80% of the data and evaluated on a held-out 20% test set (**7,730 unseen articles**).

### Performance Metrics

| Metric | Score | What it means |
|:---|:---:|:---|
| 🎯 **Accuracy** | **98.50%** | Overall correct predictions |
| 🔍 **Precision** | 97.95% | Of articles predicted Real, how many actually were |
| 📡 **Recall** | **99.34%** | Of all Fake articles, how many were correctly caught |
| 📈 **ROC-AUC** | **0.9987** | Near-perfect discrimination ability |

> 🚨 **False Negatives:** We only missed 28 actual fake articles out of 7,730, resulting in the incredibly high 99.34% recall.

---

## 6. Phase 2: Agentic AI Reasoning Layer

While achieving 98.5% accuracy is excellent, classical ML cannot cross-reference facts. To solve this, we implemented an advanced Agentic AI workflow using **LangGraph** and **Groq**.

If an article passes through the ML classifier, its output is handed to the AI Agent pipeline, which operates sequentially:

1. **Tone & Domain Extraction**: The LLM reads the article to classify the overall mood (e.g., *sensationalist*, *objective*) and topic.
2. **Text Highlighting**: The agent specifically hunts for *Clickbait Phrasing*, *Absolute Claims*, and *Unsupported Numerical Data*, extracting direct quotes.
3. **Claim Breakdown**: The LLM decomposes the article into isolated, checkable facts.
4. **Tool Use (Web Verification)**: We integrated the **Tavily Web Search API**. The AI executes live queries against trusted online domains to check the claims, producing a summarised fact-check report independently of its initial knowledge base.
5. **Human Explanation**: The LLM synthesizes the ML probability, its own highlights, and the web evidence into a clean, easy-to-understand verdict.

---

## 7. Application Architecture

To serve this to end-users, we built a **Streamlit** dashboard. Because the AI graph takes several seconds to run and requires memory, we architected a robust `st.session_state` storage system so the user can interact freely without reloading the heavy AI computations.

### Key Interactive Features
- **Visual Analytics**: Interactive pie and bar charts showing the ML feature breakdown.
- **Agent Sandbox**: Users can read the AI's step-by-step breakdown (Claims, Verification).
- **Article Comparison**: A standalone tool where users paste a second article for live side-by-side comparative analysis.
- **Follow-up Chat**: A persistent conversational interface allowing users to query the AI agent *about* the current article without losing context.

### Tech Stack

| Layer | Technology |
|:---|:---|
| Web Framework | `Streamlit`, `Plotly`, `streamlit-shadcn-ui` |
| Analytics (Phase 1) | `scikit-learn`, `joblib`, `NLTK`, `newspaper3k` |
| AI Orchestration (Phase 2) | `LangGraph`, `LangChain` |
| LLM API | `Groq API` (Llama-3/Mixtral) |
| Web Search API | `Tavily Search API` |

---

## 8. Conclusion

We started this project asking a straightforward question — can a computer tell fake news from real news simply by analyzing its text? Using TF-IDF and Logistic Regression, we achieved **98.50% accuracy**.

However, we took the project significantly further by integrating Agentic AI. By chaining LLM reasoning nodes with live website tracking, we created an engine that mimics a human editor — it flags the article, isolates suspicious text, hops onto a search engine to verify citations, and finally provides a cohesive explanation.

### Future Improvements
- 🤖 Fine-tune **BERT or RoBERTa** to replace the classical Logistic Regression model for even better semantic handling.
- 🌍 Add **multilingual support** (translating non-English input before routing to the agent).
- 📰 Include a feature where the agent proactively rewrites the flagged fake article to reflect the truth based on findings.

---

## 9. Team Contribution

| Name | Enrollment No. | Contributions |
|:---|:---:|:---|
| **Vikash Kumar** | 2401010503 | Dataset collection, text preprocessing (`utils.py`), ML model training notebook, foundational AI Agent graph integration & architecture. |
| **Piyush Raj** | 2401010328 | Backend application logic, LangGraph node algorithms (Highlights, Claim Breakdown, Tavily Tool Use), Session State architecture context handling. |
| **Mohammed Yaseen** | 2401010281 | Exploratory Data Analysis, ML model evaluation and comparison, writing and organising the project report. |
| **Sankalp M Tellur** | 2401010416 | Frontend UI design & development (Streamlit, Shadcn), GitHub repository setup, deployment to Streamlit Community Cloud, QA testing. |

*All members collaborated on system design, presentation, and ideation.*

---

*Submitted as part of the academic curriculum. All reported ML metrics are computed on held-out test data not used during model training.*
