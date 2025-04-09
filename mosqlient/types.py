from abc import ABC, abstractmethod
from datetime import date
from typing import Union, List, Dict, Optional, Literal

from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator
from pydantic import BaseModel, ConfigDict

from mosqlient import validations as v


APP = Annotated[str, AfterValidator(v.validate_django_app)]

ID = Annotated[int, AfterValidator(v.validate_id)]
Name = Annotated[str, AfterValidator(v.validate_name)]
Description = Annotated[str, AfterValidator(v.validate_description)]
AuthorName = Annotated[Optional[str], AfterValidator(v.validate_author_name)]
AuthorUserName = Annotated[str, AfterValidator(v.validate_author_username)]
AuthorInstitution = Annotated[
    Optional[str], AfterValidator(v.validate_author_institution)
]
Repository = Annotated[str, AfterValidator(v.validate_repository)]
ImplementationLanguage = Annotated[
    str | dict[Literal["language"], str],
    AfterValidator(v.validate_implementation_language),
]
Disease = Annotated[str, AfterValidator(v.validate_disease)]
ADMLevel = Annotated[int, AfterValidator(v.validate_adm_level)]
Temporal = Annotated[bool, AfterValidator(v.validate_temporal)]
Spatial = Annotated[bool, AfterValidator(v.validate_spatial)]
Categorical = Annotated[bool, AfterValidator(v.validate_categorical)]
TimeResolution = Annotated[str, AfterValidator(v.validate_time_resolution)]
Tags = Annotated[list, AfterValidator(v.validate_tags)]  # TODO:

Commit = Annotated[str, AfterValidator(v.validate_commit)]
Date = Annotated[Union[str, date], AfterValidator(v.validate_date)]
PredictionData = Annotated[
    List[Dict], AfterValidator(v.validate_prediction_data)
]

UF = Annotated[str, AfterValidator(v.validate_uf)]
Geocode = Annotated[int, AfterValidator(v.validate_geocode)]


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
