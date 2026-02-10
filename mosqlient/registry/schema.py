from datetime import date
from typing import Optional, List, Literal, Dict

from mosqlient import types


class Model(types.Schema):
    id: int
    repository: str
    description: str
    disease: str
    category: str
    adm_level: int
    time_resolution: str
    sprint: Optional[int] = None
    predictions_count: int
    active: bool
    created_at: date
    last_update: date


class PredictionDataRow(types.Schema):
    date: date
    lower_95: Optional[float] = None
    lower_90: Optional[float] = None
    lower_80: Optional[float] = None
    lower_50: Optional[float] = None
    pred: float
    upper_50: Optional[float] = None
    upper_80: Optional[float] = None
    upper_90: Optional[float] = None
    upper_95: Optional[float] = None

    class Config:
        json_encoders = {date: lambda v: v.strftime("%Y-%m-%d")}

    def dict(self, **kwargs):
        _d = super().dict(**kwargs)
        if _d.get("date"):
            _d["date"] = _d["date"].strftime("%Y-%m-%d")
        return _d


class Prediction(types.Schema):
    id: Optional[types.ID] = None
    model: Model
    predict_date: types.Date
    commit: types.Commit
    description: types.Description
    case_definition: Optional[str] = None
    published: bool
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    scores: Optional[Dict[str, float]] = None
    adm_0: str = "BRA"
    adm_1: Optional[int] = None
    adm_2: Optional[int] = None
    adm_3: Optional[int] = None
    data: Optional[List[PredictionDataRow]] = []


class ModelGETParams(types.Params):
    method: Literal["GET"] = "GET"
    app: str = "registry"
    endpoint: str = "models"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
    id: Optional[int] = None
    repository: Optional[str] = None
    description: Optional[str] = None
    disease: Optional[str] = None
    category: Optional[str] = None
    adm_level: Optional[int] = None
    time_resolution: Optional[str] = None
    sprint: Optional[int] = None
    predictions_count: Optional[int] = None
    active: Optional[bool] = None
    created_at: Optional[date] = None
    last_update: Optional[date] = None

    def params(self) -> dict:
        p = {
            "id": self.id,
            "repository": self.repository,
            "description": self.description,
            "disease": self.disease,
            "category": self.category,
            "adm_level": self.adm_level,
            "time_resolution": self.time_resolution,
            "sprint": self.sprint,
            "predictions_count": self.predictions_count,
            "active": self.active,
            "created_at": self.created_at,
            "last_update": self.last_update,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}


class PredictionGETParams(types.Params):
    method: Literal["get"] = "GET"
    app: str = "registry"
    endpoint: str = "predictions"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
    id: int
    model_id: Optional[types.ID] = None
    model_repository: Optional[types.Repository] = None
    model_adm_level: Optional[types.ADMLevel] = None
    model_category: Optional[types.Category] = None
    model_time_resolution: Optional[types.TimeResolution] = None
    model_disease: Optional[types.Disease] = None
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
            "model_adm_level": self.model_adm_level,
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
    method: Literal["POST"] = "POST"
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


class PredictionDELETEParams(types.Params):
    method: Literal["DELETE"] = "DELETE"
    app: types.APP = "registry"
    endpoint: str = "predictions/{predict_id}"
    #
    id: types.ID

    def __init__(self, id: types.ID, **kwargs):
        super().__init__(id=id, **kwargs)
        self.endpoint = self.endpoint.replace("{predict_id}", str(id))

    def params(self):
        return
