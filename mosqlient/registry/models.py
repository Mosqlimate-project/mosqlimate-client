from urllib.parse import urljoin
from typing import Literal, Optional, Any, Dict, AnyStr, List

import json
import requests
import nest_asyncio
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scoringrules import crps_normal, logs_normal
from pydantic import BaseModel, ConfigDict

from mosqlient import types
from mosqlient.client import Client
from mosqlient.errors import ClientError, ModelPostError, PredictionPostError
from mosqlient.requests import get_all_sync
from mosqlient._utils import parse_params
from mosqlient._config import get_api_url
from mosqlient.registry import schema


nest_asyncio.apply()


class Base(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    def __str__(self):
        return self.json()

    def json(self):
        return self._schema.json()


class User(Base):
    _schema: schema.UserSchema

    def __init__(self, name: types.AuthorName, username: types.AuthorUserName, **kwargs):
        super().__init__(**kwargs)
        self._schema = schema.UserSchema(name=name, username=username)

    @property
    def name(self) -> types.AuthorName:
        return self._schema.name

    @property
    def username(self) -> types.AuthorUserName:
        return self._schema.username


class Author(Base):
    user: User
    _schema: schema.AuthorSchema

    def __init__(self, user: User | dict, institution: types.AuthorInstitution, **kwargs):

        if isinstance(user, dict):
            user = User(**user)

        kwargs["user"] = user

        super().__init__(**kwargs)
        self.user = user

        self._schema = schema.AuthorSchema(user=user._schema, institution=institution)

    def __repr__(self) -> str:
        return self._schema.user.name

    @property
    def institution(self) -> types.AuthorInstitution:
        return self._schema.institution

    @classmethod
    def get(
        cls,
        name: Optional[types.AuthorName] = None,
        institution: Optional[types.AuthorInstitution] = None,
        username: Optional[types.AuthorUserName] = None,
        **kwargs,
    ):
        timeout = kwargs["timeout"] if "timeout" in kwargs else 300

        params = {"name": name, "institution": institution, "username": username}
        params = parse_params(**params)

        return [
            Author(**m)
            for m in get_all_sync(app="registry", endpoint="authors", params=params, pagination=False, timeout=timeout)
        ]


class ImplementationLanguage(Base):
    _schema: schema.ImplementationLanguageSchema

    def __init__(self, language: types.ImplementationLanguage, **kwargs):
        super().__init__(**kwargs)
        if isinstance(language, dict):
            self._schema = schema.ImplementationLanguageSchema(language=language["language"])
        elif isinstance(language, str):
            self._schema = schema.ImplementationLanguageSchema(language=language)
        else:
            raise ValueError("`language` must be a str or a dict with {'language': `language`}")

    def __repr__(self) -> str:
        return self._schema.language

    def __str__(self) -> str:
        return self._schema.language

    @property
    def language(self):
        return self._schema.language


class Model(Base):
    client: Optional[Client] = None
    author: Author
    implementation_language: ImplementationLanguage
    _schema: schema.ModelSchema

    def __init__(
        self,
        name: types.Name,
        description: types.Description,
        author: Author | dict,
        repository: types.Repository,
        implementation_language: types.ImplementationLanguage,
        disease: types.Disease,
        categorical: types.Categorical,
        spatial: types.Spatial,
        temporal: types.Temporal,
        ADM_level: types.ADMLevel,
        time_resolution: types.TimeResolution,
        id: Optional[types.ID] = None,
        **kwargs,
    ):

        if isinstance(author, dict):
            author = Author(user=author["user"], institution=author["institution"])
        kwargs["author"] = author

        language = ImplementationLanguage(language=implementation_language)
        kwargs["implementation_language"] = language

        super().__init__(**kwargs)

        self.author = author
        self.implementation_language = language

        self._schema = schema.ModelSchema(
            id=id,
            name=name,
            description=description,
            author=author._schema,
            repository=repository,
            implementation_language=language._schema,
            disease=disease,
            categorical=categorical,
            spatial=spatial,
            temporal=temporal,
            ADM_level=ADM_level,
            time_resolution=time_resolution,
        )

    def __repr__(self) -> str:
        return self.name

    @property
    def id(self) -> types.ID | None:
        return self._schema.id

    @property
    def name(self) -> types.Name:
        return self._schema.name

    @property
    def description(self) -> types.Description:
        return self._schema.description

    @property
    def repository(self) -> types.Repository:
        return self._schema.repository

    @property
    def disease(self) -> types.Disease:
        return self._schema.disease

    @property
    def categorical(self) -> types.Categorical:
        return self._schema.categorical

    @property
    def spatial(self) -> types.Spatial:
        return self._schema.spatial

    @property
    def temporal(self) -> types.Temporal:
        return self._schema.temporal

    @property
    def ADM_level(self) -> types.ADMLevel:
        return self._schema.ADM_level

    @property
    def time_resolution(self) -> types.TimeResolution:
        return self._schema.time_resolution

    @classmethod
    def get(cls, **kwargs):
        """
        https://api.mosqlimate.org/docs/registry/GET/models/
        """
        timeout = kwargs["timeout"] if "timeout" in kwargs else 300

        ModelGETParams(**kwargs)
        params = parse_params(**kwargs)

        return [
            Model(**m)
            for m in get_all_sync(app="registry", endpoint="models", params=params, pagination=True, timeout=timeout)
        ]

    def post(self, **kwargs):
        """
        https://api.mosqlimate.org/docs/registry/POST/models/
        """
        timeout = kwargs["timeout"] if "timeout" in kwargs else 10

        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more info about "
                "retrieving or inserting data from Mosqlimate, please see the "
                "API Documentation"
            )

        params = {
            "name": self.name,
            "description": self.description,
            "repository": self.repository,
            "implementation_language": str(self.implementation_language),
            "disease": self.disease,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "categorical": self.categorical,
            "ADM_level": self.ADM_level,
            "time_resolution": self.time_resolution,
        }

        url = urljoin(get_api_url(), "/".join(("registry", "models")) + "/")
        headers = {"X-UID-Key": self.client.X_UID_KEY}

        resp = requests.post(url, json=params, headers=headers, timeout=timeout)

        if resp.status_code != 201:
            raise ModelPostError(
                "POST request returned status code " f"{resp.status_code} \n {resp.reason} \n {resp.json()}"
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
        **kwargs,
    ):
        """
        https://github.com/Mosqlimate-project/Data-platform/blob/main/src/registry/api.py#L258
        """
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
            "time_resolution": time_resolution,
        }

        ModelPUTParams(
            id=id,
            name=name,
            description=description,
            repository=repository,
            implementation_language=implementation_language,
            disease=disease,
            temporal=temporal,
            spatial=spatial,
            categorical=categorical,
            ADM_level=adm_level,
            time_resolution=time_resolution,
        )

        url = urljoin(get_api_url(), "/".join(("registry", "models")) + f"/{id}")
        headers = {"X-UID-Key": self.client.X_UID_KEY}
        resp = requests.put(url, json=params, headers=headers, timeout=timeout)

        return resp


class ModelGETParams(Base):
    # https://github.com/Mosqlimate-project/Data-platform/blob/main/src/registry/schema.py#L43
    id: Optional[types.ID] = None
    name: Optional[types.Name] = None
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
    tags: Optional[types.Tags] = None


class ModelPUTParams(Base):
    id: types.ID
    name: Optional[types.Name] = None
    description: Optional[types.Description] = None
    repository: Optional[types.Repository] = None
    implementation_language: Optional[types.ImplementationLanguage] = None
    disease: Optional[types.Disease] = None
    ADM_level: Optional[types.ADMLevel] = None
    temporal: Optional[types.Temporal] = None
    spatial: Optional[types.Spatial] = None
    categorical: Optional[types.Categorical] = None
    time_resolution: Optional[types.TimeResolution] = None


class Prediction(Base):
    client: Optional[Client] = None
    model: Model
    _schema: schema.PredictionSchema

    def __init__(
        self,
        model: Model | dict,
        description: types.Description,
        commit: types.Commit,
        predict_date: types.Date,
        data: types.PredictionData,
        id: Optional[types.ID] = None,
        **kwargs,
    ):
        if isinstance(model, dict):
            model = Model(**model)

        kwargs["model"] = model

        super().__init__(**kwargs)

        if isinstance(data, str):
            try:
                _data = json.loads(data)
            except json.decoder.JSONDecodeError:
                raise ValueError("str `data` must be JSON serializable")
            _data = [schema.PredictionDataRowSchema(**d) for d in _data]
        elif isinstance(data, pd.DataFrame):
            _data = [schema.PredictionDataRowSchema(**d) for d in data.to_dict(orient="records")]
        elif isinstance(data, list):
            _data = [schema.PredictionDataRowSchema(**d) for d in data]
        else:
            raise ValueError("`data` must be rather a DataFrame, a JSON str or a list of" + " dictionaries")

        self.model = model
        self._schema = schema.PredictionSchema(
            id=id, model=model._schema, description=description, commit=commit, predict_date=predict_date, data=_data
        )

    def __repr__(self) -> str:
        return f"Prediction <{self.id}>"

    @property
    def id(self) -> types.ID | None:
        return self._schema.id

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
    def data(self) -> List[Dict[AnyStr, Any]]:
        return [row.dict() for row in self._schema.data]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def calculate_score(
        self,
        data: pd.DataFrame,
        confidence_level: float = 0.9,
    ) -> dict:
        score = {}
        data_df = data[["date", "casos"]]
        data_df.date = pd.to_datetime(data_df.date)

        pred_df = self.to_dataframe()
        pred_df = pred_df.sort_values(by="date")
        pred_df.date = pd.to_datetime(pred_df.date)

        min_date = max(min(data_df.date), min(pred_df.date))
        max_date = min(max(data_df.date), max(pred_df.date))

        def dt_range(df):
            return (df.date >= min_date) & (df.date <= max_date)

        data_df = data_df.loc[dt_range(data_df)]
        data_df.reset_index(drop=True, inplace=True)
        pred_df = pred_df.loc[dt_range(pred_df)]

        z_value = stats.norm.ppf((1 + confidence_level) / 2)

        score["mae"] = mean_absolute_error(
            y_true=data_df.casos,
            y_pred=pred_df.pred,
        )

        score["mse"] = mean_squared_error(y_true=data_df.casos, y_pred=pred_df.pred)

        score["crps"] = np.mean(
            crps_normal(data_df.casos, pred_df.pred, (pred_df.upper - pred_df.lower) / (2 * z_value))
        )

        log_score = logs_normal(
            data_df.casos, pred_df.pred, (pred_df.upper - pred_df.lower) / (2 * z_value), negative=False
        )

        score["log_score"] = np.mean(np.maximum(log_score, np.repeat(-100, len(log_score))))

        alpha = 1 - confidence_level
        upper_bound = pred_df.upper.values
        lower_bound = pred_df.lower.values

        penalty = (2 / alpha * np.maximum(0, lower_bound - data_df.casos.values)) + (
            2 / alpha * np.maximum(0, data_df.casos.values - upper_bound)
        )

        score["interval_score"] = np.mean((upper_bound - lower_bound) + penalty)

        return score

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
        timeout = kwargs["timeout"] if "timeout" in kwargs else 300

        PredictionGETParams(**kwargs)
        params = parse_params(**kwargs)
        data = get_all_sync(
            app="registry",
            endpoint="predictions",
            params=params,
            pagination=True,
            timeout=timeout,
        )
        return [Prediction(**p) for p in data]

    def post(self, **kwargs):
        timeout = kwargs["timeout"] if "timeout" in kwargs else 30

        if self.id is not None:
            raise PredictionPostError("The Prediction already has an ID")

        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more info about "
                "retrieving or inserting data from Mosqlimate, please see the "
                "API Documentation"
            )

        params = {
            "model": self.model.id,
            "description": self.description,
            "commit": self.commit,
            "predict_date": self.predict_date,
            "prediction": json.dumps(self.data),
        }

        url = urljoin(get_api_url(), "/".join(("registry", "predictions")) + "/")
        headers = {"X-UID-Key": self.client.X_UID_KEY}

        resp = requests.post(url, json=params, headers=headers, timeout=timeout)

        if str(resp.status_code).startswith("5"):
            raise ClientError(
                f"{resp.status_code}: {resp.reason}. "
                + "This error is from the API Server, please try again and "
                + "contact the moderation if this error persists"
            )

        if resp.status_code != 201:
            raise PredictionPostError(
                "POST request returned status code " + f"{resp.status_code}: {resp.reason} \n {resp.json()}"
            )

        # TODO: Return a Prediction object retrieving it from the API
        return resp


class PredictionGETParams(Base):
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


class PredictionPUTParams(Base):
    model: types.ID
    description: Optional[types.Description] = None
    commit: Optional[types.Commit] = None
    predict_date: Optional[types.Date] = None
    prediction: Optional[types.PredictionData] = None
