"""
STEP 2: Exploratory Data Analysis (EDA)
-----------------------------------------
Generates plots you can drop straight into your project report / presentation.

Run: python 2_eda.py
Output: plots saved inside eda_outputs/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/processed_news.csv"
OUT_DIR = "eda_outputs"

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)

# 1. Class balance
plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df)
plt.xticks([0, 1], ["Fake", "Real"])
plt.title("Class Distribution")
plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Article length distribution
df["word_count"] = df["clean_content"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="word_count", hue="label", bins=50, kde=True, element="step")
plt.title("Article Length Distribution (Fake vs Real)")
plt.xlim(0, 1000)
plt.savefig(f"{OUT_DIR}/word_count_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Most common words per class (simple frequency, no wordcloud dependency needed)
from collections import Counter

def top_words(text_series, n=20):
    words = " ".join(text_series).split()
    return Counter(words).most_common(n)

fake_top = top_words(df[df.label == 0]["clean_content"])
real_top = top_words(df[df.label == 1]["clean_content"])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].barh([w for w, _ in fake_top[::-1]], [c for _, c in fake_top[::-1]], color="tomato")
axes[0].set_title("Top Words — Fake News")
axes[1].barh([w for w, _ in real_top[::-1]], [c for _, c in real_top[::-1]], color="seagreen")
axes[1].set_title("Top Words — Real News")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/top_words_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"EDA plots saved to '{OUT_DIR}/' folder:")
print(" - class_distribution.png")
print(" - word_count_distribution.png")
print(" - top_words_comparison.png")
