from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator

from mosqlient.client import PlatformClient
from mosqlient.validations import *

APP = Annotated[str, AfterValidator(validate_django_app)]
Client = Annotated[PlatformClient, AfterValidator(validate_client)]
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
ADMLevel = Annotated[str, AfterValidator(validate_adm_level)]
Temporal = Annotated[str, AfterValidator(validate_temporal)]
Spatial = Annotated[str, AfterValidator(validate_spatial)]
Categorical = Annotated[str, AfterValidator(validate_categorical)]
TimeResolution = Annotated[str, AfterValidator(validate_time_resolution)]
