from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import (
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description="HTTP-сервис для оценки качества данных (HW04).",
)


# ---------------------------
# Pydantic модели
# ---------------------------

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
    flags: dict[str, bool] | None = None
    dataset_shape: dict[str, int] | None = None


# ---------------------------
# Эндпоинты
# ---------------------------

@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    start = perf_counter()

    score = 1.0
    score -= req.max_missing_share

    if req.n_rows < 1000:
        score -= 0.2

    if req.n_cols > 100:
        score -= 0.1

    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    msg = "Ready for model" if ok_for_model else "Needs improvement"
    latency = (perf_counter() - start) * 1000.0

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=msg,
        latency_ms=latency,
        flags={
            "too_few_rows": req.n_rows < 1000,
            "too_many_missing": req.max_missing_share > 0.5,
        },
        dataset_shape={
            "n_rows": req.n_rows,
            "n_cols": req.n_cols,
        },
    )


@app.post("/quality-from-csv", response_model=QualityResponse, tags=["quality"])
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading CSV: {e}",
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Empty CSV",
        )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    flags_all = compute_quality_flags(
        summary,
        missing_df,
        df=df,
    )

    score = float(flags_all.get("quality_score", 0.0))
    ok_for_model = score >= 0.7
    latency = (perf_counter() - start) * 1000.0

    flags_bool = {
        k: v
        for k, v in flags_all.items()
        if isinstance(v, bool)
    }

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message="CSV processed via EDA core",
        latency_ms=latency,
        flags=flags_bool,
        dataset_shape={
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
        },
    )


@app.post("/quality-flags-from-csv", tags=["custom"])
async def quality_flags_from_csv(
    file: UploadFile = File(...)
):
    """
    Кастомный эндпоинт для HW04.
    Возвращает расширенные флаги качества данных.
    """
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    flags_all = compute_quality_flags(
        summary,
        missing_df,
        df=df,
    )

    return {
        "filename": file.filename,
        "quality_score": flags_all.get("quality_score"),
        "detailed_flags": flags_all,
    }
