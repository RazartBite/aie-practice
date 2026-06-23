# Dockerfile для сервиса предсказания оттока клиентов банка

# Базовый образ с Python 3.11
FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего кода проекта
COPY . .

# Создание директории для артефактов (если её нет)
RUN mkdir -p artifacts

# Открытие порта для доступа к сервису
EXPOSE 8000

# Запуск сервиса
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
