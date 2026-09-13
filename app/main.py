from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import predict_sentiment


app = FastAPI(
    title="NLP Sentiment API",
    description="Sentiment analysis using a fine-tuned DistilBERT model"
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class SentimentResponse(BaseModel):
    sentiment: str
    negative_probability: float
    positive_probability: float


@app.get("/")
def root():
    return {
        "message": "NLP Sentiment API is running"
    }


@app.post("/predict", response_model=SentimentResponse)
def predict(request: TextRequest):
    result = predict_sentiment(request.text)

    return result