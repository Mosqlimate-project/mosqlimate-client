__all__ = ["upload_prediction"]

from typing import Optional, Union, List, Dict, Any
import pandas as pd
from .models import Prediction


def upload_prediction(
    api_key: str,
    repository: str,
    description: str,
    commit: str,
    prediction: Union[List[Dict], pd.DataFrame],
    case_definition: str = "probable",
    published: bool = False,
    adm_0: str = "BRA",
    adm_1: Optional[int] = None,
    adm_2: Optional[int] = None,
    adm_3: Optional[int] = None,
) -> Prediction:
    """
    Upload a prediction to the Mosqlimate API.

    Parameters
    ----------
    api_key : str
        API key used to authenticate with the Mosqlimate service.
    repository : str
        The repository identifier in the format "owner/repo_name".
    description : str
        Textual description of the prediction run.
    commit : str
        Git commit hash associated with the model version.
    prediction : list of dict or pandas.DataFrame
        Forecast data. If a DataFrame is provided, it must contain columns matching
        the prediction schema (date, pred, lower_95, etc.).
    case_definition : str, default="probable"
        The case definition used (e.g., "probable" or "reported").
    published : bool, default=False
        Whether the prediction should be visible to the public.
    adm_0 : str, default="BRA"
        ISO 3166-1 alpha-3 country code.
    adm_1 : int, optional
        State-level administrative division geocode.
    adm_2 : int, optional
        Municipality-level geocode.
    adm_3 : int, optional
        Sub-municipality-level geocode.

    Returns
    -------
    Prediction
        The created Prediction object.
    """

    prediction_data = []

    if isinstance(prediction, pd.DataFrame):
        df = prediction.copy()
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)

        prediction_data = df.to_dict(orient="records")
    else:
        prediction_data = prediction

    float_fields = [
        "lower_95",
        "lower_90",
        "lower_80",
        "lower_50",
        "pred",
        "upper_50",
        "upper_80",
        "upper_90",
        "upper_95",
    ]

    clean_prediction = []
    for item in prediction_data:
        i: Dict[str, Any] = {"date": str(item["date"])}
        for field in float_fields:
            if field in item and item[field] is not None:
                i[field] = float(item[field])
            else:
                i[field] = None
        clean_prediction.append(i)

    return Prediction.post(
        api_key=api_key,
        repository=repository,
        description=description,
        commit=commit,
        case_definition=case_definition,
        published=published,
        adm_0=adm_0,
        adm_1=adm_1,
        adm_2=adm_2,
        adm_3=adm_3,
        prediction=clean_prediction,
    )
