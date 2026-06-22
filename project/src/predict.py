import pandas as pd
from catboost import CatBoostClassifier

from src.utils import load_json


class Predictor:
    def __init__(self, model_dir: str = "artifacts"):
        self.model_dir = model_dir
        self.metadata = load_json(f"{model_dir}/metadata.json")
        self.model = CatBoostClassifier()
        self.model.load_model(f"{model_dir}/model.cbm")
        self.features_order = self.metadata["features_order"]
        self.model_version = self.metadata["model_version"]

    def predict_one(self, payload: dict):
        df = pd.DataFrame([payload])
        df = df[self.features_order]

        proba = float(self.model.predict_proba(df)[0, 1])
        pred = int(proba >= 0.5)

        return {
            "prediction": pred,
            "proba": round(proba, 4),
            "model_version": self.model_version
        }
