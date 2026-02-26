import pandas as pd
import re

df = pd.read_csv("WELFake_Dataset.csv")

print("Original shape:", df.shape)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Clean only title and text
df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)

print("After cleaning shape:", df.shape)

df.to_csv("news_dataset_clean.csv", index=False)

print("Cleaning completed ✅")
