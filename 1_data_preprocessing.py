"""
STEP 1: Data Preprocessing
---------------------------
Dataset: Kaggle "Fake and Real News Dataset"
Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
You need two files: Fake.csv and True.csv
Place them inside a folder called `data/` next to this script.

Run: python 1_data_preprocessing.py
Output: data/processed_news.csv (cleaned, combined, labeled dataset)
"""

import pandas as pd
import re
import string
import os

DATA_DIR = "data"
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_news.csv")


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove URLs, punctuation, extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"<.*?>", " ", text)                    # remove HTML tags
    text = re.sub(r"\[.*?\]", " ", text)                  # remove text in brackets
    text = re.sub(r"\d+", " ", text)                      # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace
    return text


def load_and_merge():
    if not (os.path.exists(FAKE_PATH) and os.path.exists(TRUE_PATH)):
        raise FileNotFoundError(
            f"Could not find {FAKE_PATH} or {TRUE_PATH}.\n"
            "Download the dataset from Kaggle and place Fake.csv and True.csv inside the 'data/' folder."
        )

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df["label"] = 0   # 0 = fake
    true_df["label"] = 1   # 1 = real

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for a richer signal
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + ". " + df["text"]).astype(str)

    # Clean
    df["clean_content"] = df["content"].apply(clean_text)

    # Drop empty / duplicate rows
    df = df[df["clean_content"].str.len() > 20]
    df = df.drop_duplicates(subset="clean_content")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df[["title", "text", "clean_content", "label"]]


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_and_merge()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Processed dataset saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print("\nSample row:")
    print(df.iloc[0])
