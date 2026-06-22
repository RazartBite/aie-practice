from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    CreditScore: int = Field(..., example=619)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., example=42)
    Tenure: int = Field(..., example=2)
    Balance: float = Field(..., example=0.0)
    NumOfProducts: int = Field(..., example=1)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    EstimatedSalary: float = Field(..., example=101348.88)


class PredictResponse(BaseModel):
    prediction: int
    proba: float
    model_version: str
