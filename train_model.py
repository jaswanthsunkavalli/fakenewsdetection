import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Example small dataset (replace with your dataset)
data = {
    "text": [
        "Government confirms new policy update",
        "Shocking secret revealed by unknown source",
        "Scientists discover new planet",
        "You won a million dollars click here"
    ],
    "label": [1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vectorized, y)

# Save properly in binary mode
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")
