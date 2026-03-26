from datetime import timedelta
from datetime import date as dt
from typing import Optional, List, Literal, Dict

import pandas as pd
from mosqlient import types
from pydantic import model_validator, field_serializer
from epiweeks import Week


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
    created_at: dt
    last_update: dt


class PredictionDataRow(types.Schema):
    date: dt
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
        json_encoders = {dt: lambda v: v.strftime("%Y-%m-%d")}

    def dict(self, **kwargs):
        _d = super().dict(**kwargs)
        if _d.get("date") and type(_d["date"]) is dt:
            _d["date"] = _d["date"].strftime("%Y-%m-%d")
        return _d

    @field_serializer("date")
    def serialize_date(self, v: Optional[dt], _info):
        if v is None:
            return None
        return v.strftime("%Y-%m-%d")

    def model_dump(self, **kwargs):
        return super().model_dump(**kwargs)

    @model_validator(mode="after")
    def validate_bounds(cls, values):

        if values.lower_80:
            if not (
                0
                <= values.lower_95
                <= values.lower_90
                <= values.lower_80
                <= values.lower_50
                <= values.pred
                <= values.upper_50
                <= values.upper_80
                <= values.upper_90
                <= values.upper_95
            ):
                raise ValueError(
                    (
                        "Prediction bounds are not in the correct order or "
                        "contain negative values"
                    ),
                )
        else:
            if not (0 <= values.lower_90 <= values.pred <= values.upper_90):
                raise ValueError(
                    (
                        "Prediction bounds are not in the correct order or "
                        "contain negative values"
                    ),
                )
        return values


class Prediction(types.Schema):
    id: Optional[types.ID] = None
    model: Model
    commit: types.Commit
    description: types.Description
    case_definition: Optional[str] = None
    published: bool
    start_date: Optional[dt] = None
    end_date: Optional[dt] = None
    scores: Optional[Dict[str, float]] = None
    adm_0: Optional[str] = None
    adm_1: Optional[int] = None
    adm_2: Optional[int] = None
    adm_3: Optional[int] = None
    data: Optional[List[PredictionDataRow]] = []

    @model_validator(mode="after")
    def validate_dates(self) -> "Prediction":
        if not self.data:
            return self

        self.data.sort(key=lambda x: x.date)

        time_res = self.model.time_resolution
        dates = [p.date for p in self.data]
        is_sprint = bool(self.model.sprint)

        if len(dates) != len(set(dates)):
            raise ValueError("duplicate dates found in predictions.")

        if is_sprint:
            df_dates = pd.to_datetime(dates)
            year = df_dates.year.max()

            expected_range = pd.date_range(
                start=Week(year - 1, 41).startdate(),
                end=Week(year, 40).startdate(),
                freq="W-SUN",
            )

            missing_dates = expected_range.difference(df_dates)

            if not missing_dates.empty:
                missing_str = ", ".join(missing_dates.strftime("%Y-%m-%d"))
                raise ValueError(
                    "the following dates are missing from your"
                    f" predictions: {missing_str}."
                )

        for i in range(len(dates) - 1):
            diff = dates[i + 1] - dates[i]
            if time_res == "week" and diff != timedelta(weeks=1):
                raise ValueError(
                    "gap detected: missing week "
                    f"between {dates[i]} and {dates[i+1]}."
                )
            elif time_res == "day" and diff != timedelta(days=1):
                raise ValueError(
                    f"gap detected: missing day between "
                    f"{dates[i]} and {dates[i+1]}."
                )

        for p in self.data:
            ew = Week.fromdate(p.date)
            if time_res == "week" and ew.startdate() != p.date:
                raise ValueError(
                    f"date {p.date} is not the start of CDC "
                    f"week {ew.week} (Sunday)."
                )

        return self


class ModelGETParams(types.Params):
    method: Literal["GET"] = "GET"
    app: str = "registry"
    endpoint: str = "models"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
    id: Optional[int] = None
    repository_owner: Optional[str] = None
    repository_organization: Optional[str] = None
    repository_name: Optional[str] = None
    description: Optional[str] = None
    disease: Optional[str] = None
    category: Optional[str] = None
    adm_level: Optional[int] = None
    time_resolution: Optional[str] = None
    sprint: Optional[int] = None
    predictions_count: Optional[int] = None
    active: Optional[bool] = None
    created_at: Optional[dt] = None
    last_update: Optional[dt] = None

    def params(self) -> dict:
        p = {
            "id": self.id,
            "repository_owner": self.repository_owner,
            "repository_organization": self.repository_organization,
            "repository_name": self.repository_name,
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
    start: Optional[dt] = None
    end: Optional[dt] = None

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
    prediction: List[PredictionDataRow]
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


class PredictionPublishPATCHParams(types.Params):
    method: Literal["PATCH"] = "PATCH"
    app: str = "registry"
    endpoint: str = "prediction/{prediction_id}/publish"

    id: int
    published: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = self.endpoint.replace("{prediction_id}", str(self.id))

    def params(self) -> dict:
        return {"published": self.published}


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
