import asyncio
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict

import requests
import nest_asyncio
import pandas as pd

from mosqlient import types
from mosqlient.client import Client
from mosqlient.requests import get_all
from mosqlient.errors import ClientError, PredictionPostError
from mosqlient.config import API_DEV_URL, API_PROD_URL

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


class Prediction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Client | None

    @classmethod
    def get(cls, **kwargs):
        """
        [DOCUMENTATION](https://api.mosqlimate.org/docs/registry/GET/predictions/)

        SEARCH PARAMETERS
        All parameters are Optional, use them to filter the Predictions in the
        result

        id [int]: Search by Prediction ID
        model_id [int]: Search by Model ID
        model_name [str]: Search by Model Name
        model_ADM_level [int]: Search by ADM Level
        model_time_resolution [str]: Search by Time Resolution
        model_disease [str]: Search by Disease
        author_name [str]: Search by Author Name
        author_username [str]: Search by Author Username
        author_institution [str]: Search by Author Institution
        repository [str]: Search by GitHub repository
        implementation_language [str]: Search by Implementation Language
        temporal [bool]: Search by Temporal Models
        spatial [bool]: Search by Spatial Models
        categorical [bool]: Search by Categorical Models
        commit [str]: Search by Git Commit
        predict_date [str]: Search by prediction date. Format: YYYY-MM-DD
        start [str]: Search by prediction start date. Format: YYYY-MM-DD
        end [str]: Search by prediction end date. Format: YYYY-MM-DD
        """
        env = kwargs["env"] if "env" in kwargs else "prod"
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        PredictionGETParams(**kwargs)
        params = _params(**kwargs)

        async def fetch_prediction():
            return await get_all(
                "registry",
                "predictions",
                params,
                env=env,
                timeout=timeout
            )

        if asyncio.get_event_loop().is_running():
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(fetch_prediction())
            return loop.run_until_complete(future)
        return asyncio.run(fetch_prediction())

    def post(
        self,
        model_id: int,
        description: str,
        commit: str,
        predict_date: str,
        prediction: pd.DataFrame,
        **kwargs,
    ):
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more info about "
                "retrieving or inserting data from Mosqlimate, please see the "
                "API Documentation"
            )

        params = {
            "model_id": model_id,
            "description": description,
            "commit": commit,
            "predict_date": predict_date,
            "prediction": prediction,
        }

        PredictionPOSTParams(
            model_id=model_id,
            description=description,
            commit=commit,
            predict_date=predict_date,
            prediction=prediction
        )

        base_url = API_DEV_URL if self.client.env == "dev" else API_PROD_URL
        url = base_url + "/".join(("registry", "models")) + "/"
        headers = {"X-UID-Key": self.client.X_UID_KEY}

        resp = requests.post(
            url,
            json=params,
            headers=headers,
            timeout=timeout
        )

        if resp.status_code != 201:
            raise PredictionPostError(
                "POST request returned status code "
                f"{resp.status_code} \n {resp.reason}"
            )

        return resp


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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: types.ID
    description: types.Description
    commit: types.Commit
    predict_date: types.Date
    prediction: types.PredictionData


class PredictionPUTParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: types.ID
    description: Optional[types.Description] = None
    commit: Optional[types.Commit] = None
    predict_date: Optional[types.Date] = None
    prediction: Optional[types.PredictionData] = None
