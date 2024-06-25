import json
from typing import Optional
from pydantic import BaseModel, ConfigDict

import requests
import nest_asyncio

from mosqlient import types
from mosqlient.client import Client
from mosqlient.requests import get_all_sync
from mosqlient.errors import ClientError, PredictionPostError
from mosqlient._config import API_DEV_URL, API_PROD_URL
from mosqlient._utils import parse_params

nest_asyncio.apply()


class Prediction(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

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
        params = parse_params(**kwargs)

        return get_all_sync(
            app="registry",
            endpoint="predictions",
            params=params,
            env=env,
            timeout=timeout
        )

    def post(
        self,
        model_id: int,
        description: str,
        commit: str,
        predict_date: str,
        prediction: str,
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
            "model": model_id,
            "description": description,
            "commit": commit,
            "predict_date": predict_date,
            "prediction": json.dumps(json.loads(prediction)),
        }

        PredictionPOSTParams(
            model=model_id,
            description=description,
            commit=commit,
            predict_date=predict_date,
            prediction=prediction
        )

        base_url = API_DEV_URL if self.client.env == "dev" else API_PROD_URL
        url = base_url + "/".join(("registry", "predictions")) + "/"
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
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    id: Optional[types.ID] = None
    model: Optional[types.ID] = None
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
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    model: types.ID
    description: types.Description
    commit: types.Commit
    predict_date: types.Date
    prediction: types.PredictionData


class PredictionPUTParams(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    model: types.ID
    description: Optional[types.Description] = None
    commit: Optional[types.Commit] = None
    predict_date: Optional[types.Date] = None
    prediction: Optional[types.PredictionData] = None
