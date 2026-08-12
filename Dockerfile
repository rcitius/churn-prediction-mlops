FROM python:3.12-slim

LABEL org.opencontainers.image.source=https://github.com/rcitius/churn-prediction-mlops

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/
COPY data/ ./data/

# runtime deps only — the dev group is not installed
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "src.predict"]