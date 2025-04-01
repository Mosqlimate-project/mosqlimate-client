__all__ = ["upload_prediction"]

import requests
from datetime import date

from .models import Prediction


def upload_prediction(
    api_key: str,
    model_id: int,
    description: str,
    commit: str,
    predict_date: str | date,
    adm_1: str,
    adm_2: int,
    prediction: list[dict],
) -> requests.Response:
    return Prediction.post(
        api_key=api_key,
        model=model_id,
        description=description,
        commit=commit,
        predict_date=predict_date,
        adm_1=adm_1,
        adm_2=adm_2,
        prediction=prediction,
    )
