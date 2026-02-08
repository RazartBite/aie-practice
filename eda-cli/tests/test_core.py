from __future__ import annotations
import pandas as pd
from eda_cli.core import (
    compute_quality_flags, correlation_matrix, flatten_summary_for_print,
    missing_table, summarize_dataset, top_categories,
)

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [10, 20, 30, None],
        "height": [140, 150, 160, 170],
        "city": ["A", "B", "A", None],
    })

def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)
    assert summary.n_rows == 4
    assert summary.n_cols == 3

def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)
    summary = summarize_dataset(df)
    # Передаем df, так как мы изменили сигнатуру функции
    flags = compute_quality_flags(summary, missing_df, df=df)
    assert 0.0 <= flags["quality_score"] <= 1.0

def test_new_heuristics():
    # Создаем датасет с константной колонкой и дубликатами ID
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 3], # Дубликаты!
        "constant_col": [5, 5, 5, 5], # Константа!
        "value": [10, 20, 30, 40]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df=df)
    
    # Проверяем наши новые флаги
    assert flags["has_constant_columns"] is True
    assert flags["has_suspicious_id_duplicates"] is True