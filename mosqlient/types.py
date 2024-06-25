from typing import Union
from typing_extensions import Annotated

from pydantic.functional_validators import AfterValidator

from mosqlient.validations import *  # noqa


APP = Annotated[str, AfterValidator(validate_django_app)]

ID = Annotated[int, AfterValidator(validate_id)]
Name = Annotated[str, AfterValidator(validate_name)]
Description = Annotated[str, AfterValidator(validate_description)]
AuthorName = Annotated[str, AfterValidator(validate_author_name)]
AuthorUserName = Annotated[str, AfterValidator(validate_author_username)]
AuthorInstitution = Annotated[str, AfterValidator(validate_author_institution)]
Repository = Annotated[str, AfterValidator(validate_repository)]
ImplementationLanguage = Annotated[
    str, AfterValidator(validate_implementation_language)
]
Disease = Annotated[str, AfterValidator(validate_disease)]
ADMLevel = Annotated[int, AfterValidator(validate_adm_level)]
Temporal = Annotated[bool, AfterValidator(validate_temporal)]
Spatial = Annotated[bool, AfterValidator(validate_spatial)]
Categorical = Annotated[bool, AfterValidator(validate_categorical)]
TimeResolution = Annotated[str, AfterValidator(validate_time_resolution)]
Tags = Annotated[list, AfterValidator(validate_tags)]  # TODO:

Commit = Annotated[str, AfterValidator(validate_commit)]
Date = Annotated[str, AfterValidator(validate_date)]
PredictionData = Annotated[str, AfterValidator(validate_prediction_data)
]

UF = Annotated[str, AfterValidator(validate_uf)]
Geocode = Annotated[int, AfterValidator(validate_geocode)]
