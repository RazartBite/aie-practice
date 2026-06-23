# Dockerfile для сервиса предсказания оттока клиентов банка

FROM python:3.12-slim

# Установка рабочей директории
WORKDIR /app

# Копирование requirements.txt
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY src/ ./src/
COPY models/ ./models/

# Установка PYTHONPATH
ENV PYTHONPATH=/app

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
