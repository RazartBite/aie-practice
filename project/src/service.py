import os
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.schemas import PredictRequest, PredictResponse
from src.predict import Predictor
from src.logging_config import setup_logging


setup_logging()
logger = logging.getLogger("bank_churn_service")

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    model_dir = os.getenv("MODEL_DIR", "artifacts")
    predictor = Predictor(model_dir=model_dir)
    logger.info("Model loaded from %s", model_dir)
    yield


app = FastAPI(
    title="Bank Churn Prediction Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    try:
        response = await call_next(request)
        logger.info("%s %s id=%s status=%s",
                    request.method, request.url.path, request_id, response.status_code)
        return response
    except Exception as e:
        logger.exception("Unhandled error id=%s error=%s", request_id, str(e))
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predictor.predict_one(request.model_dump())
        return result
    except Exception as e:
        logger.exception("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")
