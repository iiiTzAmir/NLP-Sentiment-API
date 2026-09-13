import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_ID = "iiTzAmir21/distilbert-imdb-sentiment"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID
)

model.eval()


def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    negative_probability = probabilities[0][0].item()
    positive_probability = probabilities[0][1].item()

    if prediction == 1:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "sentiment": sentiment,
        "negative_probability": negative_probability,
        "positive_probability": positive_probability
    }