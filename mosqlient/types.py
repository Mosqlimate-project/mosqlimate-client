from abc import ABC, abstractmethod
from datetime import date
from typing import Union, List, Dict, Optional, Literal, Self

import pandas as pd
import requests
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator
from pydantic import BaseModel, ConfigDict

from mosqlient.validations import *  # noqa


APP = Annotated[str, AfterValidator(validate_django_app)]

ID = Annotated[int, AfterValidator(validate_id)]
Name = Annotated[str, AfterValidator(validate_name)]
Description = Annotated[str, AfterValidator(validate_description)]
AuthorName = Annotated[str, AfterValidator(validate_author_name)]
AuthorUserName = Annotated[str, AfterValidator(validate_author_username)]
AuthorInstitution = Annotated[
    Optional[str], AfterValidator(validate_author_institution)
]
Repository = Annotated[str, AfterValidator(validate_repository)]
ImplementationLanguage = Annotated[
    str | dict[Literal["language"], str],
    AfterValidator(validate_implementation_language),
]
Disease = Annotated[str, AfterValidator(validate_disease)]
ADMLevel = Annotated[int, AfterValidator(validate_adm_level)]
Temporal = Annotated[bool, AfterValidator(validate_temporal)]
Spatial = Annotated[bool, AfterValidator(validate_spatial)]
Categorical = Annotated[bool, AfterValidator(validate_categorical)]
TimeResolution = Annotated[str, AfterValidator(validate_time_resolution)]
Tags = Annotated[list, AfterValidator(validate_tags)]  # TODO:

Commit = Annotated[str, AfterValidator(validate_commit)]
Date = Annotated[Union[str, date], AfterValidator(validate_date)]
PredictionData = Annotated[
    Union[List[Dict], str, List],
    AfterValidator(validate_prediction_data),
]

UF = Annotated[str, AfterValidator(validate_uf)]
Geocode = Annotated[int, AfterValidator(validate_geocode)]


class Schema(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=()
    )


class Params(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=()
    )
    method: Literal["GET", "POST", "PUT", "DELETE"] = None
    app: APP = None
    endpoint: str = None

    @abstractmethod
    def params(self) -> dict:
        raise NotImplementedError()
