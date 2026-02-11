from datetime import date
from typing import Optional, List, Literal, Dict, Any

from mosqlient import types


class Model(types.Schema):
    id: int
    repository: str
    description: Optional[str] = ""
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
    commit: types.Commit
    description: types.Description
    case_definition: Optional[str] = None
    published: bool
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    scores: Optional[Dict[str, float]] = None
    adm_0: Optional[str] = None
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
    method: Literal["GET"] = "GET"
    app: str = "registry"
    endpoint: str = "predictions"
    page: Optional[int] = None
    per_page: Optional[int] = None

    id: Optional[int] = None
    model_id: Optional[int] = None
    model_owner: Optional[str] = None
    model_organization: Optional[str] = None
    model_name: Optional[str] = None
    model_adm_level: Optional[int] = None
    model_time_resolution: Optional[
        Literal["day", "week", "month", "year"]
    ] = None
    model_disease: Optional[str] = None
    model_category: Optional[str] = None
    model_sprint: Optional[int] = None
    start: Optional[date] = None
    end: Optional[date] = None

    def params(self) -> dict:
        p = {
            "id": self.id,
            "model_id": self.model_id,
            "model_owner": self.model_owner,
            "model_organization": self.model_organization,
            "model_name": self.model_name,
            "model_adm_level": self.model_adm_level,
            "model_time_resolution": self.model_time_resolution,
            "model_disease": self.model_disease,
            "model_category": self.model_category,
            "model_sprint": self.model_sprint,
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}


class PredictionPOSTParams(types.Params):
    method: Literal["POST"] = "POST"
    app: str = "registry"
    endpoint: str = "predictions"

    repository: str
    description: str
    commit: str
    case_definition: str
    published: bool
    prediction: List[Dict[str, Any]]
    adm_0: Optional[str] = "BRA"
    adm_1: Optional[int] = None
    adm_2: Optional[int] = None
    adm_3: Optional[int] = None

    def params(self) -> dict:
        return {
            "repository": self.repository,
            "description": self.description,
            "commit": self.commit,
            "case_definition": self.case_definition,
            "published": self.published,
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


class PredictionDataGETParams(types.Params):
    method: Literal["GET"] = "GET"
    app: str = "registry"
    endpoint: str = "predictions/{predict_id}/data"
    id: int

    def __init__(self, id: int, **kwargs):
        super().__init__(id=id, **kwargs)
        self.endpoint = self.endpoint.replace("{predict_id}", str(id))

    def params(self) -> dict:
        return {}
