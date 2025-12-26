from __future__ import annotations

from time import perf_counter
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Импорт функций из ядра (core.py)
from eda_cli.core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description="HTTP-сервис для оценки качества данных.",
)

# 1. Модели данных (Pydantic)

class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    max_missing_share: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)
    categorical_cols: int = Field(..., ge=0)

class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    message: str
    latency_ms: float
    flags: Dict[str, Any] | None = None
    dataset_shape: Dict[str, int] | None = None

# 2. Обязательный эндпоинт GET /health

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }

# 3. Обязательный эндпоинт POST /quality

@app.post("/quality", response_model=QualityResponse)
def quality(req: QualityRequest):
    start = perf_counter()
    
    # Простая эвристика
    score = 1.0 - req.max_missing_share
    if req.n_rows < 1000:
        score -= 0.2
    
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    
    latency = (perf_counter() - start) * 1000
    
    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message="Calculated from metadata",
        latency_ms=latency,
        flags={
            "too_few_rows": req.n_rows < 1000,
            "too_many_missing": req.max_missing_share > 0.5
        },
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols}
    )

# 4. Обязательный эндпоинт POST /quality-from-csv

@app.post("/quality-from-csv", response_model=QualityResponse)
async def quality_from_csv(file: UploadFile = File(...)):
    start = perf_counter()
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty CSV")

    # Использование ядра
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)
    
    score = float(flags_all.get("quality_score", 0.0))
    ok_for_model = score >= 0.7
    
    latency = (perf_counter() - start) * 1000
    
    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message="Calculated from CSV",
        latency_ms=latency,
        flags=flags_all,
        dataset_shape={"n_rows": summary.n_rows, "n_cols": summary.n_cols}
    )

# 5. Мой эндпоинт (вариант A)

@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(file: UploadFile = File(...)):
    """
    Возвращает полный набор флагов, включая проверку дубликатов ID.
    Использует доработки HW03 (has_suspicious_id_duplicates).
    """
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Передача df, чтобы сработала твоя эвристика из HW03
    flags = compute_quality_flags(summary, missing_df, df=df)
    
    return {
        "filename": file.filename,
        "n_rows": summary.n_rows,
        "quality_score": flags.get("quality_score"),
        "detailed_flags": flags
    }