FROM python:3.12-slim

LABEL org.opencontainers.image.source=https://github.com/rcitius/churn-prediction-mlops

WORKDIR /app

# deps first, so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

CMD ["python", "-m", "src.predict"]