from abc import ABC, abstractmethod
from datetime import date
from typing import Union, List, Dict, Optional, Literal

from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator
from pydantic import BaseModel, ConfigDict

from mosqlient import validations as v

APP = Annotated[str, AfterValidator(v.validate_django_app)]
ID = Annotated[int, AfterValidator(v.validate_id)]
Description = Annotated[str, AfterValidator(v.validate_description)]
Disease = Annotated[str, AfterValidator(v.validate_disease)]
Category = Annotated[str, AfterValidator(v.validate_category)]
Repository = Annotated[str, AfterValidator(v.validate_repository)]
ADMLevel = Annotated[int, AfterValidator(v.validate_adm_level)]
TimeResolution = Annotated[str, AfterValidator(v.validate_time_resolution)]
Commit = Annotated[str, AfterValidator(v.validate_commit)]
Date = Annotated[Union[str, date], AfterValidator(v.validate_date)]
PredictionData = Annotated[
    List[Dict], AfterValidator(v.validate_prediction_data)
]
UF = Annotated[str, AfterValidator(v.validate_uf)]
Geocode = Annotated[int, AfterValidator(v.validate_geocode)]
MacroHealthGeocode = Annotated[
    int, AfterValidator(v.validate_macro_health_geocode)
]


class Schema(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=()
    )


class Params(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=()
    )
    method: Literal["GET", "POST", "PUT", "DELETE"]
    app: APP
    endpoint: str
    id: Optional[int] = None

    @abstractmethod
    def params(self) -> dict:
        raise NotImplementedError()


class Model(BaseModel, ABC):
    _schema: Schema
    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=()
    )

    def __str__(self):
        return self.json()

    def json(self):
        return self._schema.json()

    def dict(self):
        return self._schema.dict()
