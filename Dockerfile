FROM python:3.12-slim

LABEL org.opencontainers.image.source=https://github.com/rcitius/churn-prediction-mlops

WORKDIR /app

# deps first, so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
COPY data/ ./data/

# make src an installed package, so imports work from any working directory
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "src.predict"]