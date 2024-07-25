from datetime import date
from typing import Optional, Literal, List

from pydantic import BaseModel


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
    id: Optional[int]
    name: str
    description: str | None = None
    author: AuthorSchema
    repository: str
    implementation_language: ImplementationLanguageSchema
    disease: Literal["dengue", "chikungunya", "zika"] | None = None
    categorical: bool | None = None
    spatial: bool | None = None
    temporal: bool | None = None
    ADM_level: Literal[0, 1, 2, 3] | None = None
    time_resolution: Literal["day", "week", "month", "year"] | None = None


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
    id: Optional[int] = None
    # model: ModelSchema
    model: dict
    description: str
    commit: str
    predict_date: date  # YYYY-mm-dd
    # data: List[PredictionDataRowSchema]
    data: list
