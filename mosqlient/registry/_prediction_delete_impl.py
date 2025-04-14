__all__ = ["delete_prediction"]

import requests

from .models import Prediction


def delete_prediction(
    api_key: str,
    prediction_id: int,
) -> requests.Response:
    """
    Function to delete a prediction registered in the platform.
    Only the author can remove the prediction.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        prediction_id : int
            Prediction id of the model.

    Returns
    --------
    request response
    """
    return Prediction.delete(api_key=api_key, id=prediction_id)
