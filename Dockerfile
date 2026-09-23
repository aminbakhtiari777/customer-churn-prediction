FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/churn_pipeline.joblib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
