from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ----------------------------
# Enable CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ----------------------------
# Load Saved Accuracy
# ----------------------------
try:
    with open("accuracy.txt", "r") as f:
        MODEL_ACCURACY = f.read()
except:
    MODEL_ACCURACY = "Not Available"

# ----------------------------
# Request Format
# ----------------------------
class NewsInput(BaseModel):
    text: str

# ----------------------------
# Prediction API
# ----------------------------
@app.post("/predict")
def predict(news: NewsInput):

    # Convert text to vector
    transformed = vectorizer.transform([news.text])

    # Predict
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0]

    fake_prob = round(probability[0] * 100, 2)
    real_prob = round(probability[1] * 100, 2)

    result = "Real" if prediction == 1 else "Fake"

    # Risk logic
    if fake_prob > 70:
        risk = "High Risk"
    elif fake_prob > 40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "prediction_result": result,
        "risk_level": risk,
        "model_accuracy": MODEL_ACCURACY + " %",
        "graph_data": {
            "Fake (%)": fake_prob,
            "Real (%)": real_prob
        }
    }
