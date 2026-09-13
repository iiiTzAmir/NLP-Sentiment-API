# NLP Sentiment API

A REST API for sentiment analysis using a fine-tuned DistilBERT model trained on the IMDb dataset.

## Features

* Sentiment classification: Positive / Negative
* Fine-tuned DistilBERT model
* FastAPI REST API
* Automatic API documentation with Swagger UI
* Input validation with Pydantic
* Automated tests with Pytest
* Model hosted on Hugging Face

## Tech Stack

* Python
* FastAPI
* PyTorch
* Hugging Face Transformers
* DistilBERT
* Pytest

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
```

## Model

The sentiment analysis model is based on `distilbert-base-uncased` and was fine-tuned on a subset of the IMDb dataset.

The fine-tuned model is available on Hugging Face:

[DistilBERT IMDb Sentiment Model](https://huggingface.co/iiTzAmir21/distilbert-imdb-sentiment)

### Training Configuration

* Dataset: IMDb
* Training samples: 2,000
* Test samples: 500
* Maximum sequence length: 128
* Batch size: 8
* Learning rate: 2e-5
* Training epochs: 1

### Evaluation Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |   0.83 |
| Precision | 0.8286 |
| Recall    | 0.8252 |
| F1 Score  | 0.8269 |

## Installation

Clone the repository:

```bash
git clone https://github.com/iiiTzAmir/NLP-Sentiment-API.git
cd NLP-Sentiment-API
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Usage

### `GET /`

Returns the API status.

Example response:

```json
{
  "message": "NLP Sentiment API is running"
}
```

### `POST /predict`

Analyzes the sentiment of the provided text.

Request:

```json
{
  "text": "This movie was absolutely amazing."
}
```

Example response:

```json
{
  "sentiment": "positive",
  "negative_probability": 0.0872,
  "positive_probability": 0.9128
}
```

The probability values represent the model's predicted probability for each sentiment class.

## Testing

Run the automated tests with:

```bash
python -m pytest
```

The project currently includes tests for:

* Root endpoint
* Positive sentiment prediction
* Empty input validation

## Limitations

* The model was fine-tuned on a relatively small subset of the IMDb dataset.
* The model is primarily designed for English text.
* Sentiment can be difficult to classify correctly for sarcasm, ambiguous expressions, typos, or mixed sentiment.
* The model is intended as a practical demonstration and is not optimized for production-scale deployment.

## Future Improvements

* Add confusion matrix and detailed error analysis
* Improve model performance with a larger training dataset
* Add more comprehensive API tests
* Add Docker support
* Extend the project to support Persian sentiment analysis
* Improve handling of mixed and context-dependent sentiment

## Author

Amir-Hossein Faam

GitHub: https://github.com/iiiTzAmir
Hugging Face: https://huggingface.co/iiTzAmir21
