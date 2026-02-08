from __future__ import annotations
from pathlib import Path
import pandas as pd
import typer
from .core import (
    DatasetSummary, compute_quality_flags, correlation_matrix,
    flatten_summary_for_print, missing_table, summarize_dataset, top_categories,
)
from .viz import (
    plot_correlation_heatmap, plot_missing_matrix,
    plot_histograms_per_column, save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов (HW03 Version)")

def _load_csv(path: Path, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc

@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))

@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    # Новые параметры
    title: str = typer.Option("EDA Report", help="Заголовок отчета"),
    top_k: int = typer.Option(5, help="Количество топ-категорий для анализа"),
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k)

    quality_flags = compute_quality_flags(summary, missing_df, df=df)

    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty: missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty: corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        
        # --- ВОТ ТУТ МЫ ДОБАВЛЯЕМ ВЫВОД constant_columns_count ---
        f.write(f"- Количество константных колонок: **{quality_flags.get('constant_columns_count', 0)}**\n")
        # -----------------------------------------------------------
        
        f.write(f"- Подозрительные дубликаты ID: **{quality_flags.get('has_suspicious_id_duplicates', False)}**\n\n")

        f.write("## Колонки\n\nСм. файл `summary.csv`.\n\n")
        f.write("## Пропуски\n\n")
        if missing_df.empty: f.write("Пропусков нет.\n\n")
        else: f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")
        
        f.write("## Гистограммы\n\nСм. файлы `hist_*.png`.\n")

    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")

if __name__ == "__main__":
    app()