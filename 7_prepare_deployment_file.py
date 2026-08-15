"""
STEP 7: Prepare a small file for deployment
----------------------------------------------
Your app needs a small "background sample" of articles to power the SHAP
explanations. Right now it's pulling this from the full processed_news.csv
(tens of MB) — this script saves just a tiny 100-row version instead, which
keeps your GitHub repo and deployment small and fast.

Run this ONCE locally before deploying: python 7_prepare_deployment_file.py
Output: models/background_sample.csv (small file, safe to upload to GitHub)
"""

import pandas as pd

df = pd.read_csv("data/processed_news.csv")
background = df["clean_content"].astype(str).sample(n=min(100, len(df)), random_state=42)
background.to_frame().to_csv("models/background_sample.csv", index=False)

print(f"Saved models/background_sample.csv with {len(background)} rows.")
print("This small file is what gets uploaded to GitHub — NOT the full processed_news.csv.")
