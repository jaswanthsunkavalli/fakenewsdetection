import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

# -----------------------------
# 1️⃣ Load CSV File
# -----------------------------
df = pd.read_csv("news_dataset_clean.csv")
print(df.columns)

# Remove missing values
df = df.dropna()

# Features and Labels
X = df["text"]
y = df["label"]

print("Dataset loaded successfully!")
print("Total samples:", len(df))

# -----------------------------
# 2️⃣ Split Data (80% Train / 20% Test)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ Convert Text → TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Text vectorization completed.")

# -----------------------------
# 4️⃣ Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

print("Model training completed.")

# -----------------------------
# 5️⃣ Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = round(accuracy * 100, 2)

print("\n==============================")
print(" Model Accuracy:", accuracy_percentage, "%")
print("==============================\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6️⃣ Save Model & Vectorizer
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# -----------------------------
# 7️⃣ Save Accuracy to File
# -----------------------------
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy_percentage))

print("\nModel, Vectorizer, and Accuracy saved successfully!")
