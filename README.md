# Fake News Detection System

A modern, interactive machine learning application that analyzes the credibility of news articles using classical NLP techniques and Logistic Regression.

![Project Preview](https://github.com/Piyush-08-bot/fake-news-detection/raw/main/preview.png)

## Live Demo

[fake-news-detection-ml-ai.streamlit.app](https://fake-news-detection-ml-ai.streamlit.app/)

## Overview

This project delivers an end‑to‑end fake‑news classification workflow. It includes a full training notebook and a polished Streamlit dashboard for real‑time analysis.

- Dataset: ISOT Fake News Dataset (~38,646 articles)
- Model: TF‑IDF Vectorization + Logistic Regression
- UI: Streamlit dashboard with visual analytics

## Features

- Real‑time analysis from pasted text or a URL
- Confidence and probability visualizations
- Classifier deep‑dive with ROC, confusion matrix, and feature importance
- NLP insights into top vocabulary patterns
- Interactive training notebook

## Tech Stack

- Core: Python 3.x
- ML/NLP: scikit‑learn, NLTK, pandas, numpy
- Dashboard: Streamlit, Plotly, Seaborn, Matplotlib
- UI Components: streamlit‑shadcn‑ui

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Piyush-08-bot/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. (Optional) Install Jupyter for training and exploration:

```bash
python3 -m pip install ipykernel jupyter
```

## Usage

### Run the Dashboard

```bash
python3 -m streamlit run app.py
```

### Training and Exploration

```bash
jupyter notebook News_Credibility_Training.ipynb
```

## Project Structure

- `app.py` — Streamlit dashboard
- `News_Credibility_Training.ipynb` — model training pipeline
- `utils.py` — text preprocessing utilities
- `models/` — serialized model and vectorizer
- `data/` — dataset storage (not tracked in git)

## Notes

- URL extraction uses `newspaper3k`. Some sites block scraping or return insufficient text.
- This classifier detects linguistic patterns and does not verify factual truth. Use it as a signal, not a final verdict.

## Deployment

Live demo:
[fake-news-detection-ml-ai.streamlit.app](https://fake-news-detection-ml-ai.streamlit.app/)

## Disclaimer

This tool is for educational purposes only. It does not replace professional fact‑checking for real‑world news events.
