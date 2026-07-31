# 📱 ZaloPay Review Severity Classification

**Applying Deep Learning to detect the severity of customer feedback for ZaloPay on Google Play**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PhoBERT](https://img.shields.io/badge/Model-PhoBERT--base--v2-orange.svg)](https://github.com/VinAIResearch/PhoBERT)

🔗 **Live Dashboard:** https://pham-phuong-thy.github.io/zalopay-review-severity-classification/dashboard/zalopay_dashboard.html

---

## 📌 Overview

ZaloPay is one of Vietnam's leading e-wallet platforms, with more than 10 million downloads and roughly 336,000–338,000 user reviews on Google Play. Most review-analysis systems stop at basic sentiment (positive/negative) or star ratings — but two negative reviews can represent very different business risks. *"The app is a bit laggy"* and *"My money was deducted but never refunded"* are both negative, yet one is a minor annoyance and the other is a financial incident that needs immediate attention.

This project goes one step further than sentiment analysis: it classifies each review into **four severity levels**, so that ZaloPay's operations team can prioritize the reviews that matter most.

| Severity | Meaning |
|---|---|
| **0 – Normal** | Neutral / satisfied feedback |
| **1 – Minor** | Small inconveniences, UI/UX suggestions |
| **2 – Serious** | Technical errors affecting usage (login failures, OTP issues, bank-linking errors) |
| **3 – Critical** | Money-related or security incidents (money deducted without refund, account locked, unauthorized transactions) |

The project follows a full **KDD (Knowledge Discovery in Databases) / Web Mining pipeline**: data collection → cleaning & preprocessing → exploratory analysis → model building → evaluation → business dashboard.

---

## 🎯 Objectives

- Build a Deep Learning model (Transformer-based, PhoBERT) to automatically classify ZaloPay reviews into 4 severity levels.
- Benchmark the deep learning approach against a classical Machine Learning baseline (TF-IDF + Logistic Regression).
- Analyze the most common pain points reported by users (transaction errors, bank linking, security, customer support, UI/UX) and how they trend over time.
- Turn model predictions into an interactive dashboard that supports fast, data-driven decision-making for the business.

**Research questions answered by this project:**
1. Can a text-based model accurately predict the severity of a customer review?
2. What are ZaloPay's most critical, recurring pain points?
3. Does a deep learning model (PhoBERT) outperform a traditional ML baseline (TF-IDF + Logistic Regression)?
4. What actionable insights can be derived to improve service quality?
5. Does a Vietnamese-native pretrained language model add real value over classical NLP methods?

---

## 🗂️ Dataset

- **Source:** ~74,636 real user reviews of the ZaloPay app scraped from the Google Play Store.
- **Period:** June 1, 2025 – June 1, 2026.
- **Fields:** review text, star rating (1–5), review date, plus engineered features (review length, etc.).
- **Labeling strategy:** severity labels were derived from star ratings combined with rule-based text signals as a starting point for supervised training; after training, all severity decisions are made purely by the models (not by rating alone).
- **Class imbalance:** the dataset is heavily skewed toward the "Normal" class, so both branches of this project use **resampling** (undersampling the majority class + controlled oversampling of minority classes) before training.

---

## 🧠 Methodology & Models

### 1. Baseline — TF-IDF + Logistic Regression
A classical, lightweight text classification pipeline: TF-IDF vectorization of cleaned/segmented Vietnamese text, followed by a Logistic Regression classifier. Fast to train, easy to deploy, and used as a reference point for the deep learning model.

### 2. Deep Learning — PhoBERT Fine-tuning
The core model of this project is **[`vinai/phobert-base-v2`](https://github.com/VinAIResearch/PhoBERT)**, a RoBERTa-based language model pretrained specifically on Vietnamese text, fully fine-tuned (all 12 Transformer layers, ~135M parameters) with a sequence classification head for the 4 severity classes.

Key implementation details:
- **Word segmentation** with `underthesea` before tokenization, so compound Vietnamese terms (e.g. `liên_kết_ngân_hàng`, `trừ_tiền`) are preserved as single semantic units.
- **Tokenization:** `AutoTokenizer` from `vinai/phobert-base-v2`, `max_length = 128`, padding & truncation.
- **Class imbalance handling:** resampling strategy (max 1,500 samples/class, oversampling capped at 5×) plus a class-weighted loss where needed.
- **Training setup:** batch size 16, learning rate `2e-5`, AdamW optimizer, linear scheduler with warmup, 7 epochs, stratified 80/10/10 train/validation/test split.
- **Model selection criterion:** best checkpoint chosen by **Macro F1-score** on the validation set (not by accuracy alone), to avoid bias toward the majority class.

---

## 📊 Results

| Metric | TF-IDF + Logistic Regression | PhoBERT Fine-tuning | Improvement |
|---|---|---|---|
| Accuracy | 0.79 | 0.87 | +0.08 |
| Macro Avg. Precision | 0.79 | 0.88 | +0.09 |
| Macro Avg. Recall | 0.79 | 0.87 | +0.08 |
| Macro Avg. F1-score | 0.79 | 0.87 | +0.08 |
| Weighted Avg. F1-score | 0.79 | 0.87 | +0.08 |

**PhoBERT outperforms the baseline on every metric**, with the largest practical gains on the classes that matter most for the business:

- **Severity 3 (Critical):** Precision jumps to **0.97** — when PhoBERT flags a review as critical, the operations team can trust it 97% of the time, dramatically reducing false alarms.
- **Severity 0 (Normal):** Recall reaches **0.96**, correctly recognizing short, casual "everything is fine" reviews that a keyword-based model tends to misclassify.
- Training converged smoothly over 7 epochs — training loss dropped from **1.16 → 0.25**, while validation Macro F1 rose from **0.60 → 0.84** (best checkpoint), showing no signs of overfitting.

### Where the models still struggle
Error analysis surfaced three recurring failure patterns, mostly on **short, ambiguous, or sarcastic reviews** (e.g. a one-word review like *"ổn"*/"fine" attached to a 1-star rating, or heavy internet slang hiding a real financial complaint). These are documented in the notebooks and discussed as directions for future work — e.g. combining the star rating as an auxiliary feature alongside the review text.

---

## 📈 Interactive Dashboard

The project includes a self-contained HTML dashboard (no backend required) summarizing:
- Overall customer satisfaction & rating trends over time
- Severity distribution and top complaint categories (transaction errors, bank linking, security, etc.)
- Side-by-side model performance comparison (Accuracy / Precision / Recall / F1 per class)
- Priority list of issues that need immediate attention

🔗 **Live demo:** https://pham-phuong-thy.github.io/zalopay-review-severity-classification/dashboard/zalopay_dashboard.html

---

## 🗃️ Repository Structure

```
zalopay-review-severity-classification/
│
├── dashboard/
│   └── zalopay_dashboard.html        # Interactive dashboard (published via GitHub Pages)
│
├── notebooks/
│   ├── 01_data_processing.ipynb       # Scraping, cleaning, EDA, text preprocessing
│   ├── 02_baseline_tfidf_logreg.ipynb # TF-IDF + Logistic Regression baseline
│   └── 03_phobert_finetuning.ipynb    # PhoBERT fine-tuning, training & error analysis
│
├── pipeline/
│   ├── 01_data_collection.ipynb      
│   ├── 02_text_cleaning.ipynb
│   ├── 03_data_cleaning.ipynb      
│   ├── 04_system_architecture.ipynb
│   ├── 05_tfidf_logistic_regression_pipeline.ipynb
│   └── 06_phobert_classification_pipeline.ipynb  
│
│
├── README.md
└── requirements.txt
```

---

## 🛠️ Tech Stack

- **Language:** Python, Vietnamese NLP (`underthesea` for word segmentation)
- **Classical ML:** `scikit-learn` (TF-IDF, Logistic Regression, `classification_report`, resampling)
- **Deep Learning:** `PyTorch`, `Hugging Face Transformers` (`vinai/phobert-base-v2`, `AutoModelForSequenceClassification`)
- **Visualization:** HTML/CSS/JS dashboard, Matplotlib/Seaborn (EDA, confusion matrices, word clouds)

---

## 🚀 Getting Started

```bash
git clone https://github.com/pham-phuong-thy/zalopay-review-severity-classification.git
cd zalopay-review-severity-classification
pip install -r requirements.txt
```

Open the notebooks in order (`01` → `04`) to reproduce the full pipeline, or open `dashboard/zalopay_dashboard.html` directly in a browser to explore the results.

---

## 🔮 Future Work

- Expand data collection to multiple channels for a more complete view of customer experience.
- Combine automatic labeling with manual/expert-reviewed labels to reduce label noise.
- Experiment with newer language models (XLM-RoBERTa, ModernBERT, Gemma, or LLMs).
- Combine severity classification with Aspect-Based Sentiment Analysis to detect *what* is wrong, not just *how severe* it is.
- Build a real-time alerting pipeline for critical financial/security complaints.
- Integrate the model into a customer support system for automatic triage and response suggestions.


---

## 📚 References

Key references include Nguyen & Nguyen (2020) on PhoBERT, Liu et al. (2019) on RoBERTa, Vaswani et al. (2017) on the Transformer architecture, and Pedregosa et al. (2011) on scikit-learn. Full citation list is available in the academic report under `report/`.
