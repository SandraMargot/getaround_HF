FROM python:3.11-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY app.py ./app.py
COPY mlflow_model ./mlflow_model

# Start FastAPI on the Hugging Face port (7860)
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port 7860"]
