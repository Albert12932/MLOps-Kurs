FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Минимальные системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY pyproject.toml poetry.lock* /app/

# Устанавливаем poetry
RUN pip install --no-cache-dir poetry==2.2.1

# В контейнере venv не нужен
RUN poetry config virtualenvs.create false

# Устанавливаем только runtime-зависимости
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Копируем код приложения
COPY . /app

EXPOSE 8000

# Запуск FastAPI инференс-сервиса
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
