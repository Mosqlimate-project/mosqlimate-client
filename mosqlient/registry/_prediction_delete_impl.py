__all__ = ["delete_prediction"]

import requests

from .models import Prediction


def delete_prediction(
    api_key: str,
    prediction_id: int,
) -> requests.Response:
    return Prediction.delete(api_key=api_key, id=prediction_id)
