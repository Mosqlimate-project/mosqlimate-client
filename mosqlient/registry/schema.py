from datetime import date
from typing import Optional, List, Literal

from mosqlient import types


class UserSchema(types.Schema):
    name: Optional[str] = None
    username: str


class ImplementationLanguageSchema(types.Schema):
    language: str


class AuthorSchema(types.Schema):
    user: UserSchema
    institution: Optional[str] = None


class TagSchema(types.Schema):
    id: Optional[int]
    name: str
    color: str


class ModelSchema(types.Schema):
    id: Optional[types.ID]
    name: types.Name
    description: str
    author: AuthorSchema
    repository: types.Repository
    implementation_language: ImplementationLanguageSchema
    disease: types.Disease
    categorical: types.Categorical
    spatial: types.Spatial
    temporal: types.Temporal
    ADM_level: types.ADMLevel
    time_resolution: types.TimeResolution
    sprint: bool


class PredictionDataRowSchema(types.Schema):
    date: date
    lower_95: Optional[float] = None
    lower_90: float
    lower_80: Optional[float] = None
    lower_50: Optional[float] = None
    pred: float
    upper_50: Optional[float] = None
    upper_80: Optional[float] = None
    upper_90: float
    upper_95: Optional[float] = None

    class Config:
        json_encoders = {date: lambda v: v.strftime("%Y-%m-%d")}

    def dict(self, **kwargs):
        _d = super().dict(**kwargs)
        _d["date"] = _d["date"].strftime("%Y-%m-%d")
        return _d


class PredictionSchema(types.Schema):
    id: Optional[types.ID] = None
    model: ModelSchema
    description: types.Description
    commit: types.Commit
    predict_date: types.Date
    adm_0: str = "BRA"
    adm_1: Optional[str] = None
    adm_2: Optional[int] = None
    adm_3: Optional[int] = None
    data: List[PredictionDataRowSchema]


class AuthorGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "registry"
    endpoint: str = "authors"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
    name: Optional[types.AuthorName] = None
    institution: Optional[types.AuthorInstitution] = None
    username: Optional[types.AuthorUserName] = None

    def params(self) -> dict:
        p = {
            "name": self.name,
            "institution": self.institution,
            "username": self.username,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}


class ModelGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "registry"
    endpoint: str = "models"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
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
    sprint: Optional[bool] = None

    def params(self) -> dict:
        p = {
            "id": self.id,
            "name": self.name,
            "author_name": self.author_name,
            "author_username": self.author_username,
            "repository": self.repository,
            "implementation_language": self.implementation_language,
            "disease": self.disease,
            "ADM_level": self.ADM_level,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "categorical": self.categorical,
            "time_resolution": self.time_resolution,
            "tags": self.tags,
            "sprint": self.sprint,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}


class ModelPOSTParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    app: types.APP = "registry"
    endpoint: str = "models"
    #
    name: types.Name
    description: types.Description
    repository: types.Repository
    implementation_language: types.ImplementationLanguage
    disease: types.Disease
    ADM_level: types.ADMLevel
    temporal: types.Temporal
    spatial: types.Spatial
    categorical: types.Categorical
    time_resolution: types.TimeResolution
    sprint: bool = False

    def params(self):
        return {
            "name": self.name,
            "description": self.description,
            "repository": self.repository,
            "implementation_language": self.implementation_language,
            "disease": self.disease,
            "ADM_level": self.ADM_level,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "categorical": self.categorical,
            "time_resolution": self.time_resolution,
            "sprint": self.sprint,
        }


class ModelPUTParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "PUT"
    app: types.APP = "registry"
    endpoint: str = "models/{model_id}"
    #
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


class ModelDELETEParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "DELETE"
    app: types.APP = "registry"
    endpoint: str = "models/{model_id}"
    #
    id: types.ID

    def __init__(self, id: types.ID, **kwargs):
        super().__init__(id=id, **kwargs)
        self.endpoint = self.endpoint.replace("{model_id}", str(id))

    def params(self):
        return


class PredictionGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "registry"
    endpoint: str = "predictions"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
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
    adm_1_geocode: Optional[int] = None
    adm_2_geocode: Optional[types.Geocode] = None
    sprint: Optional[bool] = None

    def params(self) -> dict:
        p = {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_ADM_level": self.model_ADM_level,
            "model_time_resolution": self.model_time_resolution,
            "model_disease": self.model_disease,
            "author_name": self.author_name,
            "author_username": self.author_username,
            "author_institution": self.author_institution,
            "repository": self.repository,
            "implementation_language": self.implementation_language,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "categorical": self.categorical,
            "commit": self.commit,
            "predict_date": self.predict_date,
            "start": self.start,
            "end": self.end,
            "adm_1_geocode": self.adm_1_geocode,
            "adm_2_geocode": self.adm_2_geocode,
            "sprint": self.sprint,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}


class PredictionPOSTParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "registry"
    endpoint: str = "predictions"
    #
    model: types.ID
    description: types.Description
    commit: types.Commit
    predict_date: types.Date
    adm_0: str = "BRA"
    adm_1: Optional[str] = None
    adm_2: Optional[int] = None
    adm_3: Optional[int] = None
    prediction: types.PredictionData

    def params(self) -> dict:
        return {
            "model": self.model,
            "description": self.description,
            "commit": self.commit,
            "predict_date": self.predict_date,
            "adm_0": self.adm_0,
            "adm_1": self.adm_1,
            "adm_2": self.adm_2,
            "adm_3": self.adm_3,
            "prediction": self.prediction,
        }


class PredictionPUTParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "PUT"
    app: types.APP = "registry"
    endpoint: str = "predictions"
    #
    model: types.ID
    description: Optional[types.Description] = None
    commit: Optional[types.Commit] = None
    predict_date: Optional[types.Date] = None
    prediction: Optional[types.PredictionData] = None


class PredictionDELETEParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "DELETE"
    app: types.APP = "registry"
    endpoint: str = "predictions/{predict_id}"
    #
    id: types.ID

    def __init__(self, id: types.ID, **kwargs):
        super().__init__(id=id, **kwargs)
        self.endpoint = self.endpoint.replace("{predict_id}", str(id))

    def params(self):
        return
