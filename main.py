from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
import os

app = FastAPI()

# ✅ Enable CORS (Required for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load Model & Vectorizer Safely
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    print("Error loading model/vectorizer:", e)
    model = None
    vectorizer = None


# ✅ Request Schema
class NewsInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Fake News Detection API is running 🚀"}


@app.post("/predict")
def predict(news: NewsInput):

    if model is None or vectorizer is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model or vectorizer not loaded properly."}
        )

    try:
        transformed = vectorizer.transform([news.text])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]

        fake_prob = round(float(probability[0]) * 100, 2)
        real_prob = round(float(probability[1]) * 100, 2)

        result = "Real" if prediction == 1 else "Fake"

        # Risk Logic
        if fake_prob > 70:
            risk = "High Risk"
        elif fake_prob > 40:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return {
            "prediction_result": result,
            "risk_level": risk,
            "headline_review": "Headline analyzed successfully.",
            "article_review": "Article content processed using ML model.",
            "graph_data": {
                "Fake (%)": fake_prob,
                "Real (%)": real_prob
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )
