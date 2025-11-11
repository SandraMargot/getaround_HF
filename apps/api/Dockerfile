FROM python:3.11-slim

WORKDIR /app

# Install required packages
COPY apps/api/requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

# Copy API code
COPY apps/api/ ./

# Copy the MLflow model
COPY mlflow_model ./mlflow_model

# Start the FastAPI server
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
