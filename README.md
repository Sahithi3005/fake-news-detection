# 📰 Fake News Detection System

An explainable fake news classifier combining classical ML (TF-IDF + Logistic Regression/XGBoost)
and deep learning (fine-tuned DistilBERT), with SHAP-based explainability and a live Streamlit demo.

## Why this project is different
Most fake-news-classifier projects stop at "accuracy: 95%, done." This one adds:
- **Model comparison**: classical ML baseline vs. transformer, with honest tradeoff discussion
- **Explainability**: SHAP shows *which words* drove each prediction, not just a black-box label
- **A working demo**: paste any article and get an instant credibility check + explanation

## Project Structure
```
fake_news_project/
├── data/                      # put Fake.csv and True.csv here (see Setup)
├── models/                    # trained models saved here after training
├── eda_outputs/                # generated plots
├── 1_data_preprocessing.py    # clean & merge dataset
├── 2_eda.py                   # exploratory data analysis + plots
├── 3_train_baseline.py        # TF-IDF + Logistic Regression + XGBoost
├── 4_train_transformer.py     # fine-tune DistilBERT
├── 5_explainability.py        # SHAP explanations (script version)
├── 6_app.py                   # Streamlit demo app
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Get the **"Fake and Real News Dataset"** from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `Fake.csv` and `True.csv` inside a `data/` folder next to these scripts.

### 3. Run the pipeline in order
```bash
python 1_data_preprocessing.py     # ~1 min
python 2_eda.py                    # ~1 min, generates plots for your report
python 3_train_baseline.py         # ~2-5 min
python 4_train_transformer.py      # 20-60 min (use Google Colab GPU — see notes in file)
python 5_explainability.py         # ~1 min, generates example SHAP plot
```

### 4. Launch the demo
```bash
streamlit run 6_app.py
```

## Model Performance
*(Fill this in after training — this table is your evidence for interviews)*

| Model                     | Accuracy | F1-Score | Notes                          |
|---------------------------|----------|----------|---------------------------------|
| Logistic Regression + TF-IDF | —     | —        | Fast baseline                  |
| XGBoost + TF-IDF          | —        | —        | Tree-based comparison          |
| DistilBERT (fine-tuned)   | —        | —        | Deep learning, semantic understanding |

## Interview Talking Points

**Why TF-IDF baseline first?**
Shows engineering discipline — establish a benchmark before reaching for deep learning.
Also useful because SHAP explanations on linear models are exact and fast, unlike transformers.

**Why DistilBERT instead of full BERT/RoBERTa?**
40% smaller, ~60% faster inference, retains ~97% of BERT's language understanding —
a deliberate tradeoff for a resource/time-constrained project, which is worth explaining as
a design decision, not a limitation.

**Why SHAP instead of just showing accuracy?**
Fake news models often learn *stylistic* patterns (sensational language, punctuation, source
formatting) rather than *factual* patterns. SHAP lets you show and discuss this honestly —
a much stronger answer than pretending the model "understands truth."

**Known limitations (be upfront about these — it shows maturity):**
- Trained on a specific dataset/time period; may not generalize to breaking news or new topics
- Detects linguistic/stylistic patterns correlated with fake news, not verified factual accuracy
- No real-time source credibility or social-spread signal in this version (mentioned as future work)

## Future Work (mention these as "if I had more time")
- Add source credibility scoring (domain reputation)
- Incorporate social media spread patterns (bot-like propagation detection)
- Stance detection against verified fact-checking databases
- Browser extension for one-click checking while reading news
