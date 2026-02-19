import pandas as pd
import re

# Load dataset
df = pd.read_csv("fake_news_data.csv")

print("Original shape:", df.shape)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))

print("After cleaning shape:", df.shape)

df.to_csv("news_dataset_clean.csv", index=False)

print("Cleaning completed ✅")
