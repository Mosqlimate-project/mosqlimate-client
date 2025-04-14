__all__ = ["delete_model"]

import requests

from .models import Model


def delete_model(
    api_key: str,
    model_id: int,
) -> requests.Response:
    """
    Function to delete a model registered in the platform.
    Only the author can remove the model.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        model_id : int
            Model id of the model.

    Returns
    --------
    request response
    """

    return Model.delete(api_key=api_key, id=model_id)
