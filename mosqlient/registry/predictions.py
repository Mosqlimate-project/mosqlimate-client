import json
from datetime import date
from typing import Optional, List, ForwardRef
from urllib.parse import urljoin

import requests
import nest_asyncio
import pandas as pd
from pydantic import BaseModel, ConfigDict

from mosqlient import types
from mosqlient.client import Client
from mosqlient.requests import get_all_sync
from mosqlient.errors import ClientError, PredictionPostError
from mosqlient._config import get_api_url
from mosqlient._utils import parse_params
from mosqlient.registry.schema import PredictionSchema, PredictionDataRowSchema, ModelSchema

nest_asyncio.apply()


class Prediction(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    client: Optional[Client] = None
    _schema: PredictionSchema

    def __init__(
        self,
        id: types.ID,
        # model: ModelSchema,
        model: dict,
        description: types.Description,
        commit: types.Commit,
        predict_date: types.Date,
        # data: List[PredictionDataRowSchema],
        data: list,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._schema = PredictionSchema(
            id=id,
            model=model,
            description=description,
            commit=commit,
            predict_date=predict_date,
            data=data
        )

    def __repr__(self):
        return f"Prediction <{self.id}>"

    @property
    def id(self) -> types.ID:
        return self._schema.id

    @property
    def model(self) -> ForwardRef('Model'):
        return self._schema.model

    @property
    def description(self) -> types.Description:
        return self._schema.description

    @property
    def commit(self) -> types.Commit:
        return self._schema.commit

    @property
    def predict_date(self) -> types.Date:
        return self._schema.predict_date

    @property
    def data(self):  # TODO: -> types.Data?:
        return self._schema.data

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

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
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        PredictionGETParams(**kwargs)
        params = parse_params(**kwargs)

        return [
            Prediction(**p) for p in get_all_sync(
                app="registry",
                endpoint="predictions",
                params=params,
                timeout=timeout
            )
        ]

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

        url = urljoin(
            get_api_url(),
            "/".join(("registry", "predictions")) + "/"
        )
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
