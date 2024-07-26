__all__ = ["upload_model"]

from typing import Literal

import requests

from mosqlient import Client
from .models import Model


def upload_model(
    name: str,
    description: str,
    repository: str,
    implementation_language: str,
    disease: Literal["dengue", "chikungunya", "zika"],
    temporal: bool,
    spatial: bool,
    categorical: bool,
    adm_level: Literal[0, 1, 2, 3],
    time_resolution: Literal["day", "week", "month", "year"],
    api_key: str
) -> requests.Response:
    client = Client(x_uid_key=api_key)
    return Model(
        id=1,
        client=client,
        name=name,
        description=description,
        repository=repository,
        implementation_language=implementation_language,
        disease=disease,
        temporal=temporal,
        spatial=spatial,
        categorical=categorical,
        ADM_level=adm_level,
        time_resolution=time_resolution,
    ).post()
