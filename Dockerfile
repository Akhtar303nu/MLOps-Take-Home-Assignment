FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.train", "--data", "data/WA_Fn-UseC_-Telco-Customer-C.csv"]