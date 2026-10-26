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

from mosqlient import types
from mosqlient.validations import validate_adm_1, validate_geocode
from .models import Prediction


def get_predictions(
    api_key: str,
    id: Optional[int] = None,
    model_id: Optional[int] = None,
    model_name: Optional[str] = None,
    model_owner: Optional[str] = None,
    model_organization: Optional[str] = None,
    adm_level: Optional[int] = None,
    model_time_resolution: Optional[
        Literal["day", "week", "month", "year"]
    ] = None,
    disease: Optional[types.Disease] = None,
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
    imdc_year: Optional[int] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    adm_1: Optional[types.ADM1] = None,
    adm_2: Optional[types.Geocode] = None,
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
    adm_level : int, optional
        Administrative level (0: national, 1: state, 2: municipality, 3: sub-municipality).
    model_time_resolution : str, optional
        Temporal resolution ('day', 'week', 'month', 'year').
    disease : str, optional
        Disease code ('A90', 'A92.0', 'A92.5') or name ('dengue',
        'chikungunya', 'zika').
    model_category : str, optional
        Category of the model (e.g., 'quantitative', 'spatio_temporal_quantitative').
    imdc_year : int, optional
        The year of the IMDC if the model was part of a competition (e.g., 2024).
    start : date, optional
        Filter predictions with data starting on or after this date.
    end : date, optional
        Filter predictions with data ending on or before this date.
    adm_1 : int or str, optional
        Filter results by state code (e.g., 33) or UF abbreviation (e.g.,
        'RJ'). Applied client-side after retrieving predictions.
    adm_2 : int, optional
        Filter results by municipality IBGE geocode (7 digits). Applied
        client-side after retrieving predictions.

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
        "adm_level": adm_level,
        "model_time_resolution": model_time_resolution,
        "disease": disease,
        "model_category": model_category,
        "imdc_year": imdc_year,
        "start": str(start) if start else None,
        "end": str(end) if end else None,
    }
    predictions = Prediction.get(api_key=api_key, **params)

    adm_1_code = validate_adm_1(adm_1) if adm_1 is not None else None
    adm_2_code = validate_geocode(adm_2) if adm_2 is not None else None

    if adm_1_code is not None:
        predictions = [
            prediction
            for prediction in predictions
            if prediction.adm_1 == adm_1_code
        ]
    if adm_2_code is not None:
        predictions = [
            prediction
            for prediction in predictions
            if prediction.adm_2 == adm_2_code
        ]
    return predictions


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
    api_key: str,
    model_name: str,
    adm_1: Optional[types.ADM1] = None,
    adm_2: Optional[types.Geocode] = None,
) -> List[Prediction]:
    """Retrieve predictions for a model name, optionally filtered by adm."""
    return get_predictions(
        api_key,
        model_name=model_name,
        adm_1=adm_1,
        adm_2=adm_2,
    )


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
    return Prediction.get(api_key=api_key, adm_level=adm_level)


def get_predictions_by_time_resolution(
    api_key: str, time_resolution: str
) -> List[Prediction]:
    """Retrieve predictions by time resolution."""
    return Prediction.get(
        api_key=api_key, model_time_resolution=time_resolution
    )


def get_predictions_by_disease(
    api_key: str, disease: types.Disease
) -> List[Prediction]:
    """Retrieve predictions by disease code or name."""
    return Prediction.get(api_key=api_key, disease=disease)


def get_predictions_between(
    api_key: str, start: date, end: date
) -> List[Prediction]:
    """Retrieve predictions where data falls within the date range."""
    return Prediction.get(api_key=api_key, start=str(start), end=str(end))
