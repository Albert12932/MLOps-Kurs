Emotion MLOps

Проект предназначен для обучения и использования моделей, которые распознают эмоции в текстах. Здесь реализованы обучение моделей, ведение экспериментов и запуск API для получения предсказаний.

## Основные возможности

- Обучение моделей baseline и modernbert  
- Логирование экспериментов и метрик с помощью MLflow  
- Запуск API для инференса с использованием FastAPI  
- Автоматическая документация API через Swagger  

## Структура проекта

- **app/api** – код для REST API  
- **app/inference** – функции для предсказаний  
- **app/train** – скрипты и модули для обучения моделей  
- **configs** – конфигурационные файлы  
- **data/raw** – исходные данные  
- **data/preprocessed** – обработанные данные  
- **models** – сохранённые модели  
- **mlruns** – данные MLflow  

## Быстрый старт

1. Установите зависимости через poetry:
   ```
   poetry install
   ```

2. Подготовьте датасет (препроцессинг):
   ```
   python app/train/preprocess.py
   ```

3. Запустите серию экспериментов:
   Если есть папка scripts, выполните:
   ```
   bash scripts/run_all_experiments.sh
   ```
   Иначе выполните команды по очереди, например:
   ```
   python app/train/train_baseline.py
   python app/train/train.py
   ```

4. Запустите MLflow UI:
   ```
   export MLFLOW_TRACKING_URI=file:$(pwd)/mlruns
   mlflow ui
   ```

5. Запустите API локально:
   ```
   uvicorn app.api.main:app --reload
   ```

6. Запустите через Docker Compose:
   ```
   docker-compose up --build
   ```

## Примеры запросов

Проверка статуса сервиса:
```
curl http://localhost:8000/health
```

Получение предсказаний с параметрами top_k и threshold:
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I'm really happy but extremely anxious in the same time!", "top_k": 3, "threshold": 0.5}'
```

## Артефакты

- Файлы MODEL_CARD.md и DATASET_CARD.md с описанием моделей и данных
