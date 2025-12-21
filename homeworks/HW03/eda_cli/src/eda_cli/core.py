from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
from pandas.api import types as ptypes

@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }

def summarize_dataset(df: pd.DataFrame, example_values_per_column: int = 3) -> DatasetSummary:
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []
    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)
        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )
        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val = max_val = mean_val = std_val = None
        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())
        columns.append(
            ColumnSummary(
                name=name, dtype=dtype_str, non_null=non_null, missing=missing,
                missing_share=missing_share, unique=unique, example_values=examples,
                is_numeric=is_numeric, min=min_val, max=max_val, mean=mean_val, std=std_val,
            )
        )
    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)

def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])
    total = df.isna().sum()
    share = total / len(df)
    result = pd.DataFrame({"missing_count": total, "missing_share": share}).sort_values("missing_share", ascending=False)
    return result

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)

def top_categories(df: pd.DataFrame, max_columns: int = 5, top_k: int = 5) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols = []
    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)
    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values, "share": share.values})
        result[name] = table
    return result

def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    rows = []
    for col in summary.columns:
        rows.append({
            "name": col.name, "dtype": col.dtype, "non_null": col.non_null,
            "missing": col.missing, "missing_share": col.missing_share,
            "unique": col.unique, "is_numeric": col.is_numeric,
            "min": col.min, "max": col.max, "mean": col.mean, "std": col.std,
        })
    return pd.DataFrame(rows)

# Правки после проверки работы

def compute_quality_flags(summary: DatasetSummary, missing_df: pd.DataFrame, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    
    # Стандартные проверки
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100
    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # 1. Новая эвристика: Количество константных колонок
    constant_cols = [c.name for c in summary.columns if c.unique <= 1]
    # ВАЖНО: сохраняем именно число (count)
    flags["constant_columns_count"] = len(constant_cols)
    flags["has_constant_columns"] = len(constant_cols) > 0

    # 2. Новая эвристика: Подозрительные дубликаты ID
    suspicious_ids = False
    if df is not None:
        for col in df.columns:
            if "id" in col.lower() or "user" in col.lower():
                if df[col].nunique() < len(df):
                    suspicious_ids = True
                    break
    flags["has_suspicious_id_duplicates"] = suspicious_ids

    # Расчет скора с явным использованием constant_columns_count
    score = 1.0
    score -= max_missing_share
    if summary.n_rows < 100: score -= 0.2
    if summary.n_cols > 100: score -= 0.1
    
    # Штрафуем за наличие константных колонок
    if flags["constant_columns_count"] > 0: 
        score -= 0.15
        
    if flags["has_suspicious_id_duplicates"]: 
        score -= 0.2

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags
