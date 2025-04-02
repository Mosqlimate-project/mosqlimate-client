__all__ = ["upload_prediction"]

from typing import Optional
from datetime import date

import requests

from .models import Prediction


def upload_prediction(
    api_key: str,
    model_id: int,
    description: str,
    commit: str,
    predict_date: str | date,
    prediction: list[dict],
    adm_1: Optional[str] = None,
    adm_2: Optional[int] = None,
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
