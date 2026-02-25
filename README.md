# 📰 Fake News Detection System

A modern, interactive machine learning application designed to detect and analyze the credibility of news articles using classical NLP techniques and Logistic Regression.

![Project Preview](https://github.com/Piyush-08-bot/fake-news-detection/raw/main/preview.png) _(Note: Add your own screenshot here!)_

## 🚀 Overview

This project provides a complete end-to-end pipeline for fake news classification. It combines a rigorous data science workflow in a **Jupyter Notebook** with a premium, user-friendly **Streamlit Dashboard** for real-time analysis.

- **Dataset**: ISOT Fake News Dataset (~40,000 articles)
- **Model**: TF-IDF Vectorization + Logistic Regression (98.5% Accuracy)
- **Deployment**: Streamlit Community Cloud

## ✨ Features

- **Real-time Analysis**: Paste text or a URL to get an instant credibility score.
- **Visual Analytics**: Interactive charts showing confidence levels, word distributions, and probability breakdowns.
- **Classifier Deep Dive**: Interactive ROC curves and feature importance visualizations.
- **NLP Insights**: Breakdown of how specific linguistic patterns influence the model's decision.
- **Interactive Training**: A dedicated Jupyter Notebook for re-training and exploring the data.

## 🛠️ Tech Stack

- **Core**: Python 3.14+
- **ML/NLP**: Scikit-Learn, NLTK, Pandas, NumPy
- **Dashboard**: Streamlit, Plotly, Seaborn, Matplotlib
- **UI Components**: Streamlit-Shadcn-UI

## 📦 Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Piyush-08-bot/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install dependencies**:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. **Install Jupyter (if not already installed)**:
   ```bash
   python3 -m pip install ipykernel jupyter
   ```

## 🖥️ Usage

### 📊 Using the Dashboard

Launch the Streamlit app to start analyzing news articles:

```bash
python3 -m streamlit run app.py
```

### 🧠 Training & Exploration

Open the Jupyter Notebook to explore the training pipeline:

```bash
jupyter notebook News_Credibility_Training.ipynb
```

## 🏗️ Project Structure

- `app.py`: The main Streamlit dashboard.
- `News_Credibility_Training.ipynb`: Interactive ML training pipeline.
- `utils.py`: Text preprocessing and utility functions.
- `models/`: Saved model and vectorizer serialized files.
- `data/`: Dataset storage (ignored by git due to size).

## 🌍 Deployment

This project is live! You can visit it here:
[fake-news-detection-ml-ai.streamlit.app](https://fake-news-detection-ml-ai.streamlit.app/)

---

**Disclaimer**: This tool is for educational purposes and uses linguistic patterns to detect credibility. It does not replace human fact-checking for real-world news events.
