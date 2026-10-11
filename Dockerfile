# Serves the trained model. Train first so models/model.joblib exists:
#   python -m fraud.train --openml
#   docker build -t fraud-api .
#   docker run -p 8000:8000 fraud-api
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraud ./fraud
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "fraud.api:app", "--host", "0.0.0.0", "--port", "8000"]
