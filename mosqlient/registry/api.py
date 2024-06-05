import datetime
import asyncio
import json
from typing import Any, Literal, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict

import nest_asyncio
import requests
import pandas as pd

from mosqlient import types
from mosqlient.client import Client
from mosqlient.config import API_DEV_URL, API_PROD_URL
from mosqlient.errors import ClientError, ModelPostError
from mosqlient.requests import get_all
from mosqlient.utils.brasil import UFs

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


class Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Client | None

    @classmethod
    def get(cls, **kwargs):
        env = kwargs["env"] if "env" in kwargs else "prod"

        ModelGETParams(**kwargs)
        params = _params(**kwargs)

        async def fetch_models():
            return await get_all("registry", "models", params, env=env)

        if asyncio.get_event_loop().is_running():
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(fetch_models())
            return loop.run_until_complete(future)
        return asyncio.run(fetch_models())

    def post(
        self,
        name: str,
        description: str,
        repository: str,
        implementation_language: str,
        disease: Literal["dengue", "chikungunya", "zika"],
        temporal: bool,
        spatial: bool,
        categorical: bool,
        adm_level: Literal[0, 1, 2, 3],
        time_resolution: Literal["day", "week", "month", "year"],
        **kwargs,
    ):
        env = kwargs["env"] if "env" in kwargs else "prod"

        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more info about "
                "retrieving or inserting data from Mosqlimate, please see the "
                "API Documentation"
            )

        params = {
            "name": name,
            "description": description,
            "repository": repository,
            "implementation_language": implementation_language,
            "disease": disease,
            "temporal": temporal,
            "spatial": spatial,
            "categorical": categorical,
            "ADM_level": adm_level,
            "time_resolution": time_resolution,
        }

        self._validate_fields(**params)
        params = _params(**params)
        base_url = API_DEV_URL if env == "dev" else API_PROD_URL
        url = base_url + "/".join(("registry", "models")) + "/"
        headers = {"X-UID-Key": self.client.X_UID_KEY}
        resp = requests.post(url, json=params, headers=headers, timeout=60)

        if resp.status_code != 201:

            raise ModelPostError(
                "POST request returned status code "
                f"{resp.status_code} \n {resp.reason}"
            )

        return resp

    def update(
        self,
        id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        repository: Optional[str] = None,
        implementation_language: Optional[str] = None,
        disease: Optional[Literal["dengue", "chikungunya", "zika"]] = None,
        temporal: Optional[bool] = None,
        spatial: Optional[bool] = None,
        categorical: Optional[bool] = None,
        adm_level: Optional[Literal[0, 1, 2, 3]] = None,
        # fmt: off
        time_resolution: Optional[Literal["day", "week", "month", "year"]] = None,
        # fmt: on
        **kwargs
    ):
        env = kwargs["env"] if "env" in kwargs else "prod"
        timeout = kwargs["timeout"] if "timeout" in kwargs else 10

        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more infor about "
                "retrieving or inserting data from Mosqlimate, please see the "
                "API Documentation"
            )

        params = {
            "name": name,
            "description": description,
            "repository": repository,
            "implementation_language": implementation_language,
            "disease": disease,
            "temporal": temporal,
            "spatial": spatial,
            "categorical": categorical,
            "ADM_level": adm_level,
            "time_resolution": time_resolution
        }

        ModelPOSTParams(**params)
        params = _params(**params)

        base_url = API_DEV_URL if env == "dev" else API_PROD_URL
        url = base_url + "/".join(("registry", "models")) + f"/{id}"
        headers = {"X-UID-Key": self.client.X_UID_KEY}
        resp = requests.put(url, json=params, headers=headers, timeout=timeout)

        return resp


class Prediction:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Client | None

    @classmethod
    def get(cls, **kwargs):
        env = kwargs["env"] if "env" in kwargs else "prod"

        cls._validate_fields(**kwargs)
        params = _params(**kwargs)

        async def fetch_models():
            return await get_all("registry", "predictions", params, env=env)

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


class ModelGETParams(BaseModel):
    id: Optional[types.ID] = None
    name: Optional[types.Name] = None
    description: Optional[types.Description] = None
    author_name: Optional[types.AuthorName] = None
    author_username: Optional[types.AuthorUserName] = None
    author_institution: Optional[types.AuthorInstitution] = None
    repository: Optional[types.Repository] = None
    implementation_language: Optional[types.ImplementationLanguage] = None
    disease: Optional[types.Disease] = None
    ADM_level: Optional[types.ADMLevel] = None
    temporal: Optional[types.Temporal] = None
    spatial: Optional[types.Spatial] = None
    categorical: Optional[types.Categorical] = None
    time_resolution: Optional[types.TimeResolution] = None


class ModelPOSTParams(BaseModel):
    name: types.Name
    description: Optional[types.Description] = None
    repository: types.Repository
    implementation_language: types.ImplementationLanguage
    disease: types.Disease
    temporal: types.Temporal
    spatial: types.Spatial
    categorical: types.Categorical
    ADM_level: types.ADMLevel
    time_resolution: types.TimeResolution


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
