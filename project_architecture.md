# Intelligent Fake News Detection Engine — Architecture & Workflow

Welcome to the **Intelligent Fake News Detection Engine**. This document serves as a comprehensive guide to understanding how the project is built, the technologies powering it, its advanced features, and exactly how data flows through the system.

Whether you are a developer, a data scientist, or a project stakeholder, this guide will provide a theoretical and visual understanding of the entire application.

---

## 1. Project Overview
At its core, this project is a hybrid system designed to combat misinformation. It combines two powerful paradigms:
1. **Classical Machine Learning (ML)**: A fast, lightweight model trained on thousands of articles to spot linguistic patterns commonly associated with fake news.
2. **Agentic AI (LLMs)**: An advanced, reasoning-based AI agent that "reads" the article, checks facts against the live internet, highlights suspicious phrasing, and explains *why* the article is real or fake in plain English.

---

## 2. Tech Stack & Tools Employed

### Frontend & UI
* **Streamlit & Streamlit Shadcn UI**: The core framework for the web dashboard, providing a reactive, modern, and beautiful user interface.
* **Plotly**: Used for rendering interactive, real-time visual analytics (pie charts, bar charts).

### Machine Learning Layer
* **Scikit-Learn**: Powers the classical ML pipeline.
* **TF-IDF Vectorizer**: Converts plain text words into mathematical vectors based on word frequency and uniqueness.
* **Logistic Regression**: The classification algorithm that decides if the vectors look like "Real" or "Fake" news based on prior training.

### Agentic AI Layer
* **LangGraph**: An orchestration framework that manages the complex multi-step "thinking" process of the AI agent.
* **LangChain & Groq LLM**: Powers the reasoning engine via rapid, high-quality Large Language Models (LLMs) to understand tone, claims, and context.
* **Tavily Search API**: Connects the AI to the live internet, allowing it to perform real-time automated fact-checking on credible news sources.
* **Python Dotenv**: Manages secure environment variables / API keys.

---

## 3. Key Features

* **Instant ML Detection**: Sub-second text analysis using trained linguistic patterns.
* **Visual Analytics**: Top 10 influential words, probability charting, and dense text statistics.
* **Tone & Domain Identification**: Automatically categorizes the mood (e.g., sensational, urgent) and topic (e.g., politics, finance).
* **Live Suspicious Text Highlighting**: Extracts clickbait, emotional triggers, and unverified numbers directly from the text into color-coded tags.
* **Fact Breakdown**: Automatically separates an article into distinct claims, assigning a verdict (Likely True / Likely False / Uncertain) to each.
* **Automated Web Verification**: Automatically searches the live internet to cross-reference claims against trusted sources.
* **Article Comparison Tool**: Allows users to paste a second article to see a side-by-side credibility comparison.
* **Interactive AI Chat**: A persistent follow-up box where users can ask the agent direct questions about the analysis.

---

## 4. Visual Architecture Diagram

The flowchart below illustrates the exact sequence of operations from the moment a user submits an article to the final rendered dashboard.

![Visual Architecture Diagram](https://kroki.io/mermaid/svg/eNp9lXtv2jwUh__fpzhStWnTyrZyGVn0bhPlVlZKWwKtXkXV5CQnYGHiyHbasnbffbYxMArMApTLOc_5-VzMRJB8CqPWK9Dr9WsI1ILRbAIJpjSjivJM2lcxI1K2MIVCooCUMuYfYRm99NOxVILP0D_6XK1Xvcjdlh5ooqZ-OX88jjnjwj_6lJ7Uy2QbNmcOlURIUlyjKpFXTj8fRJ1ghXgvUGSCmXK0tKKFpWsa8Wq1tH6QVvNOYq--TVOcM7mixVhPK2saxlXvy5eDNK9y4lUrL3JGV9uMDWyNKpfjWg0Pb7Naq5STV6vSjKYIHcYf7H3jbTg2lehleaF8GOGjAi5gPOzfvfN931apVPoGp6F9dSUwFzxGKU1xP0KTIcn05Z22nTNLtD-yiCa2I5pGO40Jg4s-XNEcdVegNTHr1LKb4ahT6rU6cIOx4oL-IqZf_kKa1bSmrbDPJ1QqGsMQJ8Lo4JkLkuowexxb1rH9ZOILTGhsbHzoNM7bZqfDdqP_e-OBmUtU27g9N3mW0gSzGCGIuUB4_xfkGTphzzQ3UQh9kk26dsuNHjRMExkdtpv2pMUa2E0QyU0C4ZaLWbqqilkdq7sbDniCPoyzBIVUJEtggA8S_ovEx28w4hnCG2jxOaEZtB-VIPEqBZvQZnUt7czRzuhkyvRXQVDInMaUF3JZ-yW3yWg8iwhVxzAWhnMM7Tk3ZMJ22WeW3XPs5ZZAf3TGLwul-2pF3SSzfU9YQfZL7VncD4c7FUhmCX-wVaZzuWu_vvhhHc-fblDQdOHsv__ecTi3pf0f5TP0XZRbjMC6uR5yiu8pgRG5p2xhwtphfkEZ8Ge4cJBgRvMtyj-09q3WgfPsYobCdFGX8UjPSvsxZyQ7wLhYuu4yB_bF5dOmQ-HrV9vge5JwuUnClVPRQRVPdf2YVpKsamZYEUNoCN2vDOVuIi7XibgOG7pBFpJKXet5zlDhrvwrq_J6e9rMP4Yb5kCZRAT6INB-9uW19RiGbwNyr4UwpjXKgimpj1eY45yLhVMr1Qe5xPyUBvPuzp1h6yjjnnY2s6Rnzj4bWngQLp_ChRmkFpHTiBORWHe6Z357mUI7a1rQWFGmDwHcZCSA0gcNHYUmC0RskqcHlu8ZIGc_Djuc6SOgVORw_aYBzSlRu8YjYww3YWAOA8LMCaBPnGUkKpcNs12gsfW4feGhI2ybmmL8AfCxV4E=)

---

## 5. Step-by-Step Workflow Explanation

If we translate the visual diagram into a theoretical sequence, here is what happens under the hood when you use the Engine:

### Phase 1: Input & Machine Learning
1. **Data Ingestion**: The user pastes raw text or provides a news URL. If a URL is provided, the system scrapes the article text.
2. **Sanitization**: The text is cleaned (removing special characters, standardizing casing) so the model can read it efficiently.
3. **ML Prediction**: The text is passed into the pre-trained Logistic Regression model. The model outputs a prediction (`FAKE` or `REAL`) and a confidence percentage (e.g., `85.4%`). It also exposes which specific vocabulary words caused it to make that decision.

### Phase 2: AI Agentic Pipeline (LangGraph)
Once the fast ML model finishes, its prediction is handed directly to the intelligent AI LangGraph pipeline to generate "explainability". The graph executes nodes sequentially:
1. **Understand News**: The LLM reads the text and assigns it a Tone (Sensational vs Neutral) and a Domain (Politics, Health, etc.).
2. **Highlight Suspicious**: The LLM acts as an editor, scanning the text specifically for clickbait phrasing, extreme absolute claims, and unbacked numerical generalizations, pulling them out as exact quotes.
3. **Reason & Break Down**: The text is conceptually broken down into 3-5 isolated factual claims. 
4. **Tool Use (Crucial Step)**: If the ML model is highly confident a text is FAKE, or moderately confident a text is REAL, the graph routes the agent to use the **Tavily Web Search Tool**. The agent queries the live internet for the claims and generates a verification summary summarizing what trusted sources say.
5. **Synthesis**: The agent looks at the ML Output, Tone, Highlights, Claims, and Web Evidence, and writes a concise, bulleted explanation of *why* the article is credible or misinformation.

### Phase 3: Rendering & Interactivity
1. **Session State Anchoring**: Streamlit takes the entire massive payload of analysis and saves it in the browser session. This ensures that the page layout doesn't break or reset when you interact with other buttons.
2. **Dashboard UI**: The dashboard dynamically renders the badges, accordion drop-downs, and data visualizers.
3. **Follow-ups**: Because the state is saved, a user can write a question in the chat box at the bottom. The system passes the user's question, along with the *saved analysis context*, back to the LLM, allowing it to provide an intelligent, context-aware answer without having to start over.

---

## 6. Codebase Structure overview
* `app.py`: The main entry point. Houses the Streamlit UI configuration, user inputs, ML loading, and UI rendering logic.
* `agent/state.py`: Defines the dictionary/schema that stores variables as they pass through the LangGraph AI pipeline.
* `agent/nodes.py`: Contains the individual Python functions for each step of the AI reasoning process (understanding, breaking down claims, generating explanation).
* `agent/tools.py`: Connects outward to the Tavily API for live internet fact-checking.
* `agent/graph.py`: The orchestrator that wires the `nodes` together conditionally to create the final state machine workflow.
* `models/`: Directory housing the saved `.pkl` weights of the trained ML model.
