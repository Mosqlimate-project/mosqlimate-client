from datetime import date
from typing import Optional, Literal, List

from pydantic import BaseModel, ConfigDict

from mosqlient import types


class UserSchema(BaseModel):
    name: str
    username: str


class ImplementationLanguageSchema(BaseModel):
    language: str


class AuthorSchema(BaseModel):
    user: UserSchema
    institution: Optional[str] = None


class TagSchema(BaseModel):
    id: Optional[int]
    name: str
    color: str


class ModelSchema(BaseModel):
    id: types.ID
    name: types.Name
    description: types.Description
    author: AuthorSchema
    repository: types.Repository
    implementation_language: ImplementationLanguageSchema
    disease: types.Disease
    categorical: Optional[types.Categorical]
    spatial: Optional[types.Spatial]
    temporal: Optional[types.Temporal]
    ADM_level: types.ADMLevel
    time_resolution: types.TimeResolution


class PredictionDataRowSchema(BaseModel):
    dates: date
    preds: float
    lower: float
    upper: float
    adm_0: str = "BRA"
    adm_1: Optional[str] = None
    adm_2: Optional[int] = None
    adm_3: Optional[int] = None


class PredictionSchema(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    id: Optional[types.ID] = None
    model: ModelSchema
    description: types.Description
    commit: types.Commit
    predict_date: types.Date
    data: types.PredictionData
