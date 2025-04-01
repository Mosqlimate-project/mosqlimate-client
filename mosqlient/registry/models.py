from urllib.parse import urljoin
from typing import Literal, Optional, Any, Dict, AnyStr, List, Self, Generator

import json
import requests
import nest_asyncio
import pandas as pd
import numpy as np
# import scipy.stats as stats
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scoringrules import crps_normal, logs_normal
from pydantic import BaseModel, ConfigDict

from mosqlient import types
from mosqlient.client import Mosqlient, Client
from mosqlient.errors import ClientError, ModelPostError, PredictionPostError
from mosqlient._utils import parse_params
from mosqlient.registry import schema


nest_asyncio.apply()


class Base(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=()
    )

    def __str__(self):
        return self.json()

    def json(self):
        return self._schema.json()


class User(Base):
    _schema: schema.UserSchema

    def __init__(
        self, name: types.AuthorName, username: types.AuthorUserName, **kwargs
    ):
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

    def __init__(
        self, user: User | dict, institution: types.AuthorInstitution, **kwargs
    ):

        if isinstance(user, dict):
            user = User(**user)

        kwargs["user"] = user

        super().__init__(**kwargs)
        self.user = user

        self._schema = schema.AuthorSchema(
            user=user._schema, institution=institution
        )

    def __str__(self) -> str:
        return self._schema.user.username

    def __repr__(self) -> str:
        return self._schema.user.username

    @staticmethod
    def params(
        method: Literal["GET", "POST", "PUT", "DELETE"]
    ) -> types.Params:
        match method.upper():
            case "GET":
                return schema.AuthorGETParams
            case _:
                raise NotImplementedError()

    @property
    def institution(self) -> types.AuthorInstitution:
        return self._schema.institution

    @classmethod
    def get(
        cls,
        api_key: str,
        name: Optional[types.AuthorName] = None,
        institution: Optional[types.AuthorInstitution] = None,
        username: Optional[types.AuthorUserName] = None,
    ):
        client = Mosqlient(x_uid_key=api_key)
        params = {
            "name": name,
            "institution": institution,
            "username": username,
        }
        params = schema.AuthorGETParams(**parse_params(**params))
        return list(cls(**item) for item in client.get(params=params))


class ImplementationLanguage(Base):
    _schema: schema.ImplementationLanguageSchema

    def __init__(self, language: types.ImplementationLanguage, **kwargs):
        super().__init__(**kwargs)
        if isinstance(language, dict):
            self._schema = schema.ImplementationLanguageSchema(
                language=language["language"]
            )
        elif isinstance(language, str):
            self._schema = schema.ImplementationLanguageSchema(
                language=language
            )
        else:
            raise ValueError(
                "`language` must be a str or a dict with {'language': `language`}"
            )

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
        sprint: bool = False,
        id: Optional[types.ID] = None,
        **kwargs,
    ):
        if isinstance(author, dict):
            author = Author(
                user=author["user"], institution=author["institution"]
            )
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
            sprint=sprint
        )

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def params(
        method: Literal["GET", "POST", "PUT", "DELETE"]
    ) -> types.Params:
        match method.upper():
            case "GET":
                return schema.ModelGETParams
            case "POST":
                return schema.ModelPOSTParams
            case "DELETE":
                return schema.ModelDELETEParams
            case _:
                raise NotImplementedError()

    @classmethod
    def get(cls, api_key: str, **kwargs):
        """
        mosqlient.schema.ModelGETParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.ModelGETParams(**kwargs)
        return list(cls(**item) for item in client.get(params))

    @classmethod
    def post(cls, api_key: str, **kwargs):
        """
        mosqlient.schema.ModelPOSTParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.ModelPOSTParams(**kwargs)
        return client.post(params)

    @classmethod
    def update(cls, api_key: str, **kwargs):
        """
        mosqlient.schema.ModelPUTParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.ModelPUTParams(**kwargs)
        return client.put(params)

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
        adm_0: str = "BRA",
        adm_1: Optional[int] = None,
        adm_2: Optional[int] = None,
        adm_3: Optional[int] = None,
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
            _data = [
                schema.PredictionDataRowSchema(**d)
                for d in data.to_dict(orient="records")
            ]
        elif isinstance(data, list):
            _data = [schema.PredictionDataRowSchema(**d) for d in data]
        else:
            raise ValueError(
                "`data` must be rather a DataFrame, a JSON str or a list of"
                + " dictionaries"
            )

        self.model = model
        self._schema = schema.PredictionSchema(
            id=id,
            model=model._schema,
            description=description,
            commit=commit,
            predict_date=predict_date,
            adm_0=adm_0,
            adm_1=adm_1,
            adm_2=adm_2,
            adm_3=adm_3,
            data=_data,
        )

    def __repr__(self) -> str:
        return f"Prediction <{self.id}>"

    @classmethod
    def get(cls, api_key: str, **kwargs):
        """
        registry.schema.PredictionGETParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.PredictionGETParams(**kwargs)
        return list(cls(**item) for item in client.get(params))

    @classmethod
    def post(self, api_key: str, **kwargs):
        """
        registry.schema.PredictionPOSTParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.PredictionPOSTParams(**kwargs)
        if isinstance(params.prediction, list):
            params.prediction = json.dumps(params.prediction)
        return client.post(params)

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
