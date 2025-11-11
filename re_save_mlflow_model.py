# re_save_mlflow_model.py
from pathlib import Path
import json
import pickle
import pandas as pd
import mlflow.pyfunc

# ----- Fixed PythonModel (uses pickle.load, not pickle.model) -----
class PriceModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.cat_cols = []

    def load_context(self, context):
        self.model = pickle.load(open(context.artifacts["gb_model"], "rb"))
        self.label_encoders = pickle.load(open(context.artifacts["label_encoders"], "rb"))
        self.cat_cols = json.load(open(context.artifacts["cat_cols"], "r"))

    def predict(self, context, model_input: pd.DataFrame):
        if self.model is None:
            return [0.0] * len(model_input)
        df = model_input.copy()
        for col in self.cat_cols:
            if col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except Exception:
                    df[col] = 0
        return self.model.predict(df)

# ----- Paths -----
ROOT = Path(__file__).resolve().parent
artifacts_dir = ROOT / "models"        # expects gb_model.pkl, label_encoders.pkl, cat_cols.json
save_path = ROOT / "mlflow_model"      # will overwrite this folder

# Minimal input example (schema only; values don’t matter)
input_example = pd.DataFrame({
    "model_key": ["Citroën"],
    "mileage": [50000],
    "engine_power": [150],
    "fuel": ["diesel"],
    "paint_color": ["black"],
    "car_type": ["sedan"],
    "private_parking_available": [True],
    "has_gps": [True],
    "has_air_conditioning": [True],
    "automatic_car": [False],
    "has_getaround_connect": [True],
    "has_speed_regulator": [True],
    "winter_tires": [True],
})

# ----- Save fixed MLflow bundle -----
mlflow.pyfunc.save_model(
    path=str(save_path),
    python_model=PriceModel(),
    artifacts={
        "gb_model": str(artifacts_dir / "gb_model.pkl"),
        "label_encoders": str(artifacts_dir / "label_encoders.pkl"),
        "cat_cols": str(artifacts_dir / "cat_cols.json"),
    },
    input_example=input_example.to_dict(orient="list"),
)

print(f"✅ Overwrote: {save_path}")
