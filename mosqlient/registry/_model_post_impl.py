__all__ = ["upload_model"]

from typing import Literal

import requests

from mosqlient import types
from .models import Model


def upload_model(
    api_key: str,
    name: str,
    description: str,
    repository: str,
    implementation_language: types.ImplementationLanguage,
    disease: Literal["dengue", "chikungunya", "zika"],
    temporal: bool,
    spatial: bool,
    categorical: bool,
    adm_level: Literal[0, 1, 2, 3],
    time_resolution: Literal["day", "week", "month", "year"],
    sprint: bool,
) -> requests.Response:
    return Model.post(
        api_key=api_key,
        name=name,
        description=description,
        categorical=categorical,
        temporal=temporal,
        spatial=spatial,
        disease=disease,
        repository=repository,
        implementation_language=implementation_language,
        ADM_level=adm_level,
        time_resolution=time_resolution,
        sprint=sprint,
    )
