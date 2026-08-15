"""
STEP 6: Streamlit Demo App
-----------------------------
This is what you'll actually show in interviews / your resume link.

Run: streamlit run 6_app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import re
import string

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

MODEL_DIR = "models"
BACKGROUND_PATH = "models/background_sample.csv"


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    model = joblib.load(f"{MODEL_DIR}/logistic_regression.joblib")
    background_df = pd.read_csv(BACKGROUND_PATH)
    background = vectorizer.transform(background_df["clean_content"].astype(str)).toarray()
    explainer = shap.LinearExplainer(model, background)
    feature_names = np.array(vectorizer.get_feature_names_out())
    return vectorizer, model, explainer, feature_names


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_and_explain(text, vectorizer, model, explainer, feature_names, top_n=10):
    cleaned = clean_text(text)
    X_vec = vectorizer.transform([cleaned])

    pred_class = model.predict(X_vec)[0]
    pred_proba = model.predict_proba(X_vec)[0]

    shap_values = explainer(X_vec.toarray())
    nonzero_idx = X_vec.nonzero()[1]
    contributions = shap_values.values[0][nonzero_idx]
    words = feature_names[nonzero_idx]

    order = np.argsort(-np.abs(contributions))[:top_n]
    top_words = [(words[i], float(contributions[i])) for i in order]

    return {
        "prediction": "REAL" if pred_class == 1 else "FAKE",
        "confidence": float(max(pred_proba)),
        "top_words": top_words,
    }


# ---- UI ----
st.title("📰 Fake News Detection System")

vectorizer, model, explainer, feature_names = load_artifacts()

user_input = st.text_area(
    "Paste article text here:",
    height=200,
    placeholder="Paste the full article text (title + body) here..."
)

if st.button("Analyze", type="primary"):
    if not user_input.strip():
        st.warning("Please paste some article text first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_and_explain(user_input, vectorizer, model, explainer, feature_names)

        # Prediction banner
        if result["prediction"] == "REAL":
            st.success(f"✅ Likely REAL — confidence: {result['confidence']:.1%}")
        else:
            st.error(f"⚠️ Likely FAKE — confidence: {result['confidence']:.1%}")

        # Explanation
        st.subheader("Why this prediction?")
        st.caption("Words in green pushed toward REAL, words in red pushed toward FAKE.")

        for word, score in result["top_words"]:
            color = "green" if score > 0 else "red"
            bar_width = min(abs(score) * 100, 100)
            st.markdown(
                f"<div style='display:flex;align-items:center;margin-bottom:4px;'>"
                f"<div style='width:100px'>{word}</div>"
                f"<div style='background:{color};width:{bar_width}px;height:14px;border-radius:3px;'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()
st.caption("This website is developed by Gunapu Sahithi")