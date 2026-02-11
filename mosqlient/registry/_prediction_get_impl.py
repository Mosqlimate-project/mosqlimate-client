__all__ = [
    "get_predictions",
    "get_prediction_by_id",
    "get_predictions_by_model_id",
    "get_predictions_by_model_name",
    "get_predictions_by_model_owner",
    "get_predictions_by_model_organization",
    "get_predictions_by_adm_level",
    "get_predictions_by_time_resolution",
    "get_predictions_by_disease",
    "get_predictions_between",
]

from datetime import date
from typing import Optional, List, Literal

from .models import Prediction


def get_predictions(
    api_key: str,
    id: Optional[int] = None,
    model_id: Optional[int] = None,
    model_name: Optional[str] = None,
    model_owner: Optional[str] = None,
    model_organization: Optional[str] = None,
    model_adm_level: Optional[int] = None,
    model_time_resolution: Optional[
        Literal["day", "week", "month", "year"]
    ] = None,
    model_disease: Optional[Literal["A90", "A92.0", "A92.5"]] = None,
    model_category: Optional[
        Literal[
            "quantitative",
            "categorical",
            "spatial_quantitative",
            "spatial_categorical",
            "spatio_temporal_quantitative",
            "spatio_temporal_categorical",
        ]
    ] = None,
    model_sprint: Optional[int] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> List[Prediction]:
    """
    Retrieve one or more prediction records from the Mosqlimate API.

    Parameters
    ----------
    api_key : str
        API key used to authenticate with the Mosqlimate service.
    id : int, optional
        Unique identifier of the prediction.
    model_id : int, optional
        Unique identifier of the model.
    model_name : str, optional
        Name of the model (repository name).
    model_owner : str, optional
        Username of the model owner (e.g. GitHub username).
    model_organization : str, optional
        Name of the organization that owns the model.
    model_adm_level : int, optional
        Administrative level (0: national, 1: state, 2: municipality, 3: sub-municipality).
    model_time_resolution : str, optional
        Temporal resolution ('day', 'week', 'month', 'year').
    model_disease : str, optional
        Disease code (e.g., 'A90' for Dengue, 'A92.0' for Chikungunya, 'A92.5' for Zika).
    model_category : str, optional
        Category of the model (e.g., 'quantitative', 'spatio_temporal_quantitative').
    model_sprint : int, optional
        The year of the sprint if the model was part of a competition (e.g., 2024).
    start : date, optional
        Filter predictions with data starting on or after this date.
    end : date, optional
        Filter predictions with data ending on or before this date.

    Returns
    -------
    List[Prediction]
        A list of prediction objects matching the filters.
    """
    params = {
        "id": id,
        "model_id": model_id,
        "model_name": model_name,
        "model_owner": model_owner,
        "model_organization": model_organization,
        "model_adm_level": model_adm_level,
        "model_time_resolution": model_time_resolution,
        "model_disease": model_disease,
        "model_category": model_category,
        "model_sprint": model_sprint,
        "start": str(start) if start else None,
        "end": str(end) if end else None,
    }
    return Prediction.get(api_key=api_key, **params)


def get_prediction_by_id(api_key: str, id: int) -> Prediction | None:
    """Retrieve one prediction record by its ID."""
    res = Prediction.get(api_key=api_key, id=id)
    return res[0] if len(res) == 1 else None


def get_predictions_by_model_id(
    api_key: str, model_id: int
) -> List[Prediction]:
    """Retrieve predictions for a specific model ID."""
    return Prediction.get(api_key=api_key, model_id=model_id)


def get_predictions_by_model_name(
    api_key: str, model_name: str
) -> List[Prediction]:
    """Retrieve predictions for a specific model name (repository name)."""
    return Prediction.get(api_key=api_key, model_name=model_name)


def get_predictions_by_model_owner(
    api_key: str, model_owner: str
) -> List[Prediction]:
    """Retrieve predictions by the model owner's username."""
    return Prediction.get(api_key=api_key, model_owner=model_owner)


def get_predictions_by_model_organization(
    api_key: str, model_organization: str
) -> List[Prediction]:
    """Retrieve predictions by the model's organization name."""
    return Prediction.get(
        api_key=api_key, model_organization=model_organization
    )


def get_predictions_by_adm_level(
    api_key: str, adm_level: int
) -> List[Prediction]:
    """Retrieve predictions by administrative level."""
    return Prediction.get(api_key=api_key, model_adm_level=adm_level)


def get_predictions_by_time_resolution(
    api_key: str, time_resolution: str
) -> List[Prediction]:
    """Retrieve predictions by time resolution."""
    return Prediction.get(
        api_key=api_key, model_time_resolution=time_resolution
    )


def get_predictions_by_disease(api_key: str, disease: str) -> List[Prediction]:
    """Retrieve predictions by disease code (e.g. A90)."""
    return Prediction.get(api_key=api_key, model_disease=disease)


def get_predictions_between(
    api_key: str, start: date, end: date
) -> List[Prediction]:
    """Retrieve predictions where data falls within the date range."""
    return Prediction.get(api_key=api_key, start=str(start), end=str(end))
