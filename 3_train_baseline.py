"""
STEP 3: Baseline Models (TF-IDF + Logistic Regression + XGBoost)
-------------------------------------------------------------------
This is your safety-net model — fast to train, easy to explain in interviews,
and a benchmark to compare your transformer model against.

Run: python 3_train_baseline.py
Output:
  - models/tfidf_vectorizer.joblib
  - models/logistic_regression.joblib
  - models/xgboost_model.joblib
  - prints accuracy / precision / recall / F1 for both models
"""

import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier

DATA_PATH = "data/processed_news.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- Load data ----
df = pd.read_csv(DATA_PATH)
X = df["clean_content"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- TF-IDF vectorization ----
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.joblib")

# ---- Model 1: Logistic Regression ----
print("\n=== Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train_tfidf, y_train)
lr_preds = lr.predict(X_test_tfidf)

print(classification_report(y_test, lr_preds, target_names=["Fake", "Real"]))
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f} | F1: {f1_score(y_test, lr_preds):.4f}")

joblib.dump(lr, f"{MODEL_DIR}/logistic_regression.joblib")

# ---- Model 2: XGBoost ----
print("\n=== XGBoost ===")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train_tfidf, y_train)
xgb_preds = xgb.predict(X_test_tfidf)

print(classification_report(y_test, xgb_preds, target_names=["Fake", "Real"]))
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f} | F1: {f1_score(y_test, xgb_preds):.4f}")

joblib.dump(xgb, f"{MODEL_DIR}/xgboost_model.joblib")

print("\nModels saved to 'models/' folder.")
print("Keep these accuracy/F1 numbers — you'll compare them against the transformer model next.")
