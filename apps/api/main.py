from pathlib import Path
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Auto-detect base directory (project root)
BASE_DIR = Path(__file__).resolve().parent

# Point directly to mlflow_model folder
model_path = BASE_DIR / "mlflow_model"
model = mlflow.pyfunc.load_model(str(model_path))

class CarFeatures(BaseModel):
    model_key: str
    mileage: float
    engine_power: float
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

app = FastAPI(title="Car Price Prediction API")

@app.get("/")
def home():
    return {"status": "ok", "message": "Visit /docs for the interactive API docs."}

@app.post("/predict")
def predict_price(features: CarFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return {"predicted_rental_price": round(float(prediction[0]), 2)}
