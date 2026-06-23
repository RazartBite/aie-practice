from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# Создаем приложение
app = FastAPI(
    title="Churn Prediction API",
    description="API для предсказания оттока клиентов банка",
    version="1.0.0"
)

# Модель данных для запроса
class ClientData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    
    class Config:
        schema_extra = {
            "example": {
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.00,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88
            }
        }

# Загрузка модели (если есть)
model = None
try:
    model_path = Path.cwd() / 'artifacts' / 'baseline_pipeline.joblib'
    if model_path.exists():
        model = joblib.load(model_path)
        print("✅ Модель загружена")
    else:
        print("⚠️ Модель не найдена, используем заглушку")
except Exception as e:
    print(f"⚠️ Ошибка загрузки модели: {e}")

@app.get("/")
def read_root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "churn-prediction"}

@app.post("/predict")
def predict(client: ClientData):
    """
    Предсказание оттока клиента
    """
    # Если модели нет, возвращаем заглушку
    if model is None:
        return {
            "prediction": 0,
            "probability": 0.15,
            "message": "⚠️ Модель не загружена, возвращена заглушка",
            "client_data": client.dict()
        }
    
    try:
        # Преобразуем в DataFrame
        data = pd.DataFrame([client.dict()])
        
        # Предсказание
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "message": "Клиент уйдет" if prediction == 1 else "Клиент останется"
        }
    except Exception as e:
        return {"error": str(e)}