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
    description: types.Description
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
    lower_95: float
    lower_90: float
    lower_80: float
    lower_50: float
    pred: float
    upper_50: float
    upper_80: float
    upper_90: float
    upper_95: float

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
    #
    name: Optional[types.AuthorName] = None
    institution: Optional[types.AuthorInstitution] = None
    username: Optional[types.AuthorUserName] = None

    def params(self) -> dict:
        p = {}
        if self.name:
            p["name"] = self.name
        if self.institution:
            p["institution"] = self.institution
        if self.username:
            p["username"] = self.username
        return p


class ModelGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "registry"
    endpoint: str = "models"
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
    page: Optional[int] = None
    per_page: Optional[int] = None

    def params(self) -> dict:
        return {
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
            "per_page": self.per_page
        }


class ModelPOSTParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    app: types.APP = "registry"
    endpoint: "models"
    #
    name: types.Name
    repository: types.Repository
    implementation_language: types.ImplementationLanguage
    disease: types.Disease
    ADM_level: types.ADMLevel
    temporal: types.Temporal
    spatial: types.Spatial
    categorical: types.Categorical
    time_resolution: types.TimeResolution
    sprint: bool = False

    @property
    def params(self):
        return {
            "name": self.name,
            "repository": self.repository,
            "implementation_language": self.implementation_language,
            "disease": self.disease,
            "ADM_level": self.ADM_level,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "categorical": self.categorical,
            "time_resolution": self.time_resolution,
            "sprint": self.sprint
        }


class ModelPUTParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "PUT"
    app: types.APP = "registry"
    endpoint: "models/{model_id}"
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
