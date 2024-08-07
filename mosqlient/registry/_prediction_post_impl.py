__all__ = ["upload_prediction"]

import requests
from datetime import date

import pandas as pd

from mosqlient import Client
from .models import Prediction, Model


def upload_prediction(
    model_id: int,
    description: str,
    commit: str,
    predict_date: str | date,
    prediction: str | list[dict] | pd.DataFrame,
    api_key: str,
    **kwargs
) -> requests.Response:
    client = Client(x_uid_key=api_key)
    model = Model.get(id=model_id)[0]
    return Prediction(
        client=client,
        model=model,
        description=description,
        commit=commit,
        predict_date=predict_date,
        data=prediction,
    ).post(**kwargs)
