__all__ = ["delete_model"]

import requests

from .models import Model


def delete_model(
    api_key: str,
    model_id: int,
) -> requests.Response:
    return Model.delete(api_key=api_key, id=model_id)
