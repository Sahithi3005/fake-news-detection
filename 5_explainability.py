"""
STEP 5: Explainability with SHAP
------------------------------------
This is your differentiator vs. generic fake-news-classifier projects.

We use SHAP on the Logistic Regression + TF-IDF model because:
  - It's fast and stable (transformer SHAP is much slower / more fragile with 1 week left)
  - Linear models + SHAP give clean, easy-to-explain word-level importance
  - You can honestly say in interviews: "I used SHAP's LinearExplainer for
    fast, exact attributions on the TF-IDF model"

Run: python 5_explainability.py
Output: eda_outputs/shap_example_explanation.png + prints top contributing words
"""

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_DIR = "models"
OUT_DIR = "eda_outputs"
DATA_PATH = "data/processed_news.csv"

vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
model = joblib.load(f"{MODEL_DIR}/logistic_regression.joblib")

feature_names = np.array(vectorizer.get_feature_names_out())

# Build a small background sample from real data (SHAP needs this as a reference
# distribution to compute expected values against — using zeros would be inaccurate).
_df = pd.read_csv(DATA_PATH)
_background_texts = _df["clean_content"].astype(str).sample(n=min(100, len(_df)), random_state=42)
_background = vectorizer.transform(_background_texts).toarray()

_explainer = shap.LinearExplainer(model, _background)


def explain_prediction(text: str, top_n: int = 10):
    """
    Returns the model's prediction + the top words that pushed it
    toward FAKE or REAL, using SHAP.
    """
    X_vec = vectorizer.transform([text])
    shap_values = _explainer(X_vec.toarray())

    pred_class = model.predict(X_vec)[0]
    pred_proba = model.predict_proba(X_vec)[0]

    # Get non-zero features (words actually present in the text)
    nonzero_idx = X_vec.nonzero()[1]
    contributions = shap_values.values[0][nonzero_idx]
    words = feature_names[nonzero_idx]

    # Sort by absolute contribution
    order = np.argsort(-np.abs(contributions))[:top_n]
    top_words = [(words[i], float(contributions[i])) for i in order]

    return {
        "prediction": "REAL" if pred_class == 1 else "FAKE",
        "confidence": float(max(pred_proba)),
        "top_contributing_words": top_words,
    }


def plot_explanation(result: dict, save_path: str):
    words = [w for w, _ in result["top_contributing_words"]]
    scores = [s for _, s in result["top_contributing_words"]]
    colors = ["seagreen" if s > 0 else "tomato" for s in scores]

    plt.figure(figsize=(8, 5))
    plt.barh(words[::-1], scores[::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
    plt.xlabel("SHAP value (pushes toward REAL →  ← pushes toward FAKE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    sample_text = (
        "Scientists confirm shocking new evidence that the government has been "
        "secretly hiding the truth from the public for decades according to anonymous sources"
    )

    result = explain_prediction(sample_text)
    print("Prediction:", result["prediction"])
    print("Confidence:", round(result["confidence"], 3))
    print("\nTop contributing words:")
    for word, score in result["top_contributing_words"]:
        direction = "→ REAL" if score > 0 else "→ FAKE"
        print(f"  {word:15s} {score:+.4f}  {direction}")

    plot_explanation(result, f"{OUT_DIR}/shap_example_explanation.png")
    print(f"\nSaved visualization to {OUT_DIR}/shap_example_explanation.png")
