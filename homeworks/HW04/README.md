## Запуск HTTP-сервиса (API)

Для запуска сервиса используйте uvicorn:

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000