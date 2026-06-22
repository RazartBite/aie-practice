from fastapi.testclient import TestClient
from src.service import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_endpoint_exists():
    payload = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88
    }

    response = client.post("/predict", json=payload)

    assert response.status_code in [200, 500]
