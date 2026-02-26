import pandas as pd

# Load dataset
df = pd.read_csv("WELFake_Dataset.csv")

# Drop rows where text or label is empty
df = df.dropna(subset=['text','label']).reset_index(drop=True)

# Optional: check first rows
print(df.head())
print("Total rows:", df.shape[0])

# Save cleaned dataset
df.to_csv("news_dataset_clean.csv", index=False)
