__all__ = ["upload_prediction"]

from typing import Optional
from datetime import date
import json
import requests
import pandas as pd
from .models import Prediction


def upload_prediction(
    api_key: str,
    model_id: int,
    description: str,
    commit: str,
    predict_date: str | date,
    prediction: list[dict] | pd.DataFrame,
    adm_0: str = "BRA",
    adm_1: Optional[str] = None,
    adm_2: Optional[int] = None,
    adm_3: Optional[int] = None,
) -> requests.Response:
    """
    Upload a prediction to the Mosqlimate API.

    Converts a DataFrame or list of dictionaries containing prediction results
    to the appropriate format and sends it to the API. It must contain the columns or
    keys: "date", "lower_95", "lower_90", "lower_80", "lower_50",
            "pred", "upper_50", "upper_80", "upper_90", "upper_95".

    Parameters
    ----------
    api_key : str
        API key used to authenticate with the Mosqlimate service.
    model_id : int
        Unique identifier of the model used to generate the prediction.
    description : str
        Textual description of the prediction run.
    commit : str
        Git commit hash associated with the model version.
    predict_date : str or datetime.date
        Date the prediction corresponds to (usually the forecast publication date).
    prediction : list of dict or pandas.DataFrame
        Forecast data. If a DataFrame is provided, it must contain the following columns:
        ['date', 'lower_95', 'lower_90', 'lower_80', 'lower_50', 'pred',
         'upper_50', 'upper_80', 'upper_90', 'upper_95'].
    adm_0 : str, default="BRA"
        ISO 3166-1 alpha-3 country code (e.g., 'BRA' for Brazil).
    adm_1 : str, optional
        State-level administrative division (ADM1), e.g., state abbreviation.
    adm_2 : int, optional
        Municipality-level geocode (ADM2), typically IBGE code.
    adm_3 : int, optional
        Sub-municipality-level geocode (ADM3), if applicable.

    Returns
    -------
    requests.Response
        The response object from the Mosqlimate API.
    """

    if type(prediction) == pd.DataFrame:

        required_columns = [
            "date",
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

        assert all(
            col in prediction.columns for col in required_columns
        ), f"Missing required columns: {[col for col in required_columns if col not in prediction.columns]}"

        json_prediction = prediction.to_json(
            orient="records", date_format="iso"
        )

        prediction = [
            {
                "date": str(item["date"]),
                "lower_95": float(item["lower_95"]),
                "lower_90": float(item["lower_90"]),
                "lower_80": float(item["lower_80"]),
                "lower_50": float(item["lower_50"]),
                "pred": float(item["pred"]),
                "upper_95": float(item["upper_95"]),
                "upper_90": float(item["upper_90"]),
                "upper_80": float(item["upper_80"]),
                "upper_50": float(item["upper_50"]),
            }
            for item in json.loads(json_prediction)  # Parse once, then iterate
        ]

    return Prediction.post(
        api_key=api_key,
        model=model_id,
        description=description,
        commit=commit,
        predict_date=predict_date,
        adm_0=adm_0,
        adm_1=adm_1,
        adm_2=adm_2,
        adm_3=adm_3,
        prediction=prediction,
    )
