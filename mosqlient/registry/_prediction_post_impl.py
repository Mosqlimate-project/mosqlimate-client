__all__ = ["upload_prediction"]

from typing import Optional
from datetime import date
import json
import requests
import pandas as pd
from .models import Prediction


def upload_prediction(
    api_key: str,
    model_id: int,
    description: str,
    commit: str,
    predict_date: str | date,
    prediction: list[dict] | pd.DataFrame,
    adm_0: str = "BRA",
    adm_1: Optional[str] = None,
    adm_2: Optional[int] = None,
    adm_3: Optional[int] = None,
) -> requests.Response:

    if type(prediction) == pd.DataFrame:

        required_columns = [
            "date",
            "lower_95",
            "lower_90",
            "lower_80",
            "lower_50",
            "pred",
            "upper_50",
            "upper_80",
            "upper_90",
            "upper_95",
        ]

        assert all(
            col in prediction.columns for col in required_columns
        ), f"Missing required columns: {[col for col in required_columns if col not in prediction.columns]}"

        json_prediction = prediction.to_json(
            orient="records", date_format="iso"
        )

        prediction = [
            {
                "date": str(item["date"]),
                "lower_95": float(item["lower_95"]),
                "lower_90": float(item["lower_90"]),
                "lower_80": float(item["lower_80"]),
                "lower_50": float(item["lower_50"]),
                "pred": float(item["pred"]),
                "upper_95": float(item["upper_95"]),
                "upper_90": float(item["upper_90"]),
                "upper_80": float(item["upper_80"]),
                "upper_50": float(item["upper_50"]),
            }
            for item in json.loads(json_prediction)  # Parse once, then iterate
        ]

    return Prediction.post(
        api_key=api_key,
        model=model_id,
        description=description,
        commit=commit,
        predict_date=predict_date,
        adm_0=adm_0,
        adm_1=adm_1,
        adm_2=adm_2,
        adm_3=adm_3,
        prediction=prediction,
    )
