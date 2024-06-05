import datetime
import asyncio
import json
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict

import nest_asyncio
import pandas as pd

from mosqlient import types
from mosqlient.client import Client
from mosqlient.requests import get_all
from mosqlient.utils.brasil import UFs
from mosqlient.errors import ClientError
from mosqlient.registry.models import Model  # noqa

nest_asyncio.apply()


def _params(**kwargs) -> dict[str, Any]:
    params = {}
    for k, v in kwargs.items():
        if isinstance(v, (bool, int, float, str)):
            params[k] = str(v)
        elif v is None:
            continue
        else:
            raise TypeError(f"Unknown type f{type(v)}")

    return params


class Prediction:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Client | None

    @classmethod
    def get(cls, **kwargs):
        """
        https://api.mosqlimate.org/docs/registry/GET/predictions/
        """
        env = kwargs["env"] if "env" in kwargs else "prod"
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        cls._validate_fields(**kwargs)
        params = _params(**kwargs)

        async def fetch_models():
            return await get_all(
                "registry",
                "predictions",
                params,
                env=env,
                timeout=timeout
            )

        if asyncio.get_event_loop().is_running():
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(fetch_models())
            return loop.run_until_complete(future)
        return asyncio.run(fetch_models())

    def post(
        self,
        model: int,
        description: str,
        commit: str,
        predict_date: str,
        prediction: str | pd.DataFrame | dict,
        **kwargs,
    ):
        env = kwargs["env"] if "env" in kwargs else "prod"
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more info about "
                "retrieving or inserting data from Mosqlimate, please see the "
                "API Documentation"
            )

        params = {
            "model": model,
            "description": description,
            "commit": commit,
            "predict_date": predict_date,
            "prediction": prediction,
        }

        self._validate_fields(**params)
        params = _params(**params)
        print(params)

    @staticmethod
    def _validate_fields(**kwargs) -> None:
        PredictionFieldValidator(**kwargs)


class PredictionGETParams(BaseModel):
    id: Optional[types.ID] = None
    model_id: Optional[types.ID] = None
    model_name: Optional[types.Name] = None
    model_ADM_level: Optional[types.ADMLevel] = None
    model_time_resolution: Optional[types.TimeResolution] = None
    model_disease: Optional[types.Disease] = None
    author_name: Optional[types.AuthorName] = None
    author_username: Optional[types.AuthorUserName] = None
    author_institution: Optional[types.AuthorInstitution] = None
    repository: Optional[types.Repository] = None
    implementation_language: Optional[types.ImplementationLanguage] = None
    temporal: Optional[types.Temporal] = None
    spatial: Optional[types.Spatial] = None
    categorical: Optional[types.Categorical] = None
    commit: Optional[types.Commit] = None
    predict_date: Optional[types.Date] = None
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None


class PredictionPOSTParams(BaseModel):
    model_id: types.ID
    description: types.Description
    commit: types.Commit
    predict_date: types.Date
    prediction: types.PredictionData


class PredictionPUTParams(BaseModel):
    model_id: types.ID
    description: Optional[types.Description] = None
    commit: Optional[types.Commit] = None
    predict_date: Optional[types.Date] = None
    prediction: Optional[types.PredictionData] = None


class PredictionFieldValidator:
    FIELDS = {
        "id": (int, str),
        "model": int,
        "description": str,
        "commit": str,
        "predict_date": str,
        "prediction": str | pd.DataFrame | dict,
        # --
        "env": str
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if k == "env":
                if v not in ["dev", "prod"]:
                    raise ValueError("`env` must be 'dev' or 'prod'")

            if not isinstance(v, self.FIELDS[k]):
                raise TypeError(
                    f"Field `{k}` must have instance of "
                    f"{' or '.join(self.FIELDS[k])}"
                )

            if k == "id":
                if int(v) <= 0:
                    raise ValueError("Incorrect value for field `id`")

            if k == "model":
                if v < 0:
                    raise ValueError(
                        "Incorrect value for field `model`. Expecting Model ID"
                    )

            if k == "commit":
                if len(v) != 40:
                    raise ValueError(
                        "`commit` must be a valid GitHub Commit hash"
                    )

            if k == "predict_date":
                try:
                    datetime.datetime.fromisoformat(v)
                except ValueError:
                    raise ValueError(
                        "`predict_date` date must be in isoformat: YYYY-MM-DD"
                    )

            if k == "prediction":
                validate_prediction(v)


def validate_prediction(data: str | pd.DataFrame | dict) -> None:
    EXPECTED_FIELDS = [
        "dates",
        "preds",
        "lower",
        "upper",
        "adm_2",
        "adm_1",
        "adm_0",
    ]

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (str, dict)):
        try:
            if isinstance(data, str):
                data = json.loads(data)

            df = pd.DataFrame(data)
        except json.JSONDecodeError as err:
            raise ValueError(
                "`Prediction` data must be compatible with JSON format.\n"
                f"Error: {err}"
            )
        except ValueError as err:
            raise ValueError(
                "`Prediction` data must be compatible with pandas DataFrame."
                f"\nError: {err}"
            )
    else:
        raise ValueError(
            "Incorrect `Prediction` data type. Expecting DataFrame compatible "
            "data"
        )

    if set(df.columns) != set(EXPECTED_FIELDS):
        return (
            "Incorrect columns in `Prediction` data. Expected the exact"
            f" columns: {EXPECTED_FIELDS}"
        )

    try:
        pd.to_datetime(df['dates'], errors="raise")
    except ValueError:
        raise ValueError("Incorrect date found in `dates` column")

    for column in ["preds", "lower", "upper"]:
        if df[column].dtype != float:
            raise ValueError(
                f"`{column}` column should contain only float values"
            )

    for column in ["adm_2", "adm_1", "adm_0"]:
        unique_values = set(df[column].values)
        if len(unique_values) != 1:
            raise ValueError(f"`{column}` must contain one value")

        if df["adm_0"].values[0] != "BR":
            raise ValueError(
                "At this moment, only 'BR' adm_0 is accepted"
            )

        adm_1 = df["adm_1"].values[0]
        if adm_1 not in UFs:
            raise ValueError(f"Unkown UF {adm_1}. Expected format: 'RJ'")

        adm_2 = df["adm_2"].values[0]
        _validate_geocode(adm_2)


def _validate_geocode(geocode: str | int) -> None:
    error = f"Incorrect value for field `adm_2` [{geocode}]. Example: 3304557"

    if not isinstance(geocode, (str, int)):
        raise ValueError(error)

    try:
        geocode = int(geocode)
    except ValueError:
        raise ValueError(error)

    if len(str(geocode)) != 7:
        raise ValueError(error)
