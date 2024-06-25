__all__ = ["upload_prediction"]

import requests

from mosqlient import Client
from .predictions import Prediction


def upload_prediction(
    model_id: int,
    description: str,
    commit: str,
    predict_date: str,
    prediction: str,
    api_key: str
) -> requests.Response:
    client = Client(x_uid_key=api_key)
    return Prediction(client=client).post(
        model_id=model_id,
        description=description,
        commit=commit,
        predict_date=predict_date,
        prediction=prediction,
    )
