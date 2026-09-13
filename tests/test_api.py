from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "NLP Sentiment API is running"


def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "I really loved this movie."}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] == "positive"
    assert 0 <= data["negative_probability"] <= 1
    assert 0 <= data["positive_probability"] <= 1


def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422