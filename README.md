# NLP Sentiment API

A REST API for sentiment analysis using a fine-tuned DistilBERT model trained on the IMDb dataset.

## Features

- Sentiment classification: Positive / Negative
- Fine-tuned DistilBERT model
- FastAPI REST API
- Automatic API documentation with Swagger UI
- Input validation with Pydantic
- Automated tests with Pytest
- Model hosted on Hugging Face

## Tech Stack

- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Pytest

## Project Structure

```text
NLP-Sentiment-API/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── model.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .gitignore
├── requirements.txt
└── README.md
