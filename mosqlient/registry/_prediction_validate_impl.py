__all__ = ["validate_prediction"]

from typing import Optional, Union, List, Dict
import pandas as pd
from .models import Prediction


def validate_prediction(
    api_key: str,
    repository: str,
    description: str,
    commit: str,
    prediction: Union[List[Dict], pd.DataFrame],
    case_definition: str = "probable",
    published: bool = True,
    adm_0: str = "BRA",
    adm_1: Optional[int] = None,
    adm_2: Optional[int] = None,
    adm_3: Optional[int] = None,
) -> None:
    """
    Validates a prediction to the Mosqlimate API.

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
    None
    """
    Prediction.validate_prediction(
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
        prediction=prediction,
    )
