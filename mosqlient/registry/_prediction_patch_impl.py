__all__ = ["update_prediction_status"]

from .models import Prediction


def update_prediction_status(
    api_key: str, prediction_id: int, published: bool
) -> bool:
    """
    Update the publication status of a specific prediction in Mosqlimate.

    Parameters
    ----------
    api_key : str
        API key used to authenticate with the Mosqlimate service.
    prediction_id : int
        The unique identifier (ID) of the prediction to be updated.
    published : bool
        The new status: True to make it public, False to keep it private.

    Returns
    -------
    bool
        True if the update was successful, raises an error otherwise.
    """

    predictions = Prediction.get(api_key=api_key, id=prediction_id)

    if not predictions:
        raise ValueError(f"Prediction with ID {prediction_id} not found.")

    prediction = predictions[0]

    return prediction.update_published(status=published)
