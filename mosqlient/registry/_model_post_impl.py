__all__ = ["upload_model"]

from typing import Literal

import requests

from mosqlient import Client, types
from mosqlient.errors import ModelPostError
from .models import Model, Author


def upload_model(
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
    api_key: str,
    **kwargs
) -> requests.Response:
    client = Client(x_uid_key=api_key)
    author = Author.get(username=client.username)

    if not author:
        raise ModelPostError("Author not found")

    model = Model(
        client=client,
        author=author[0],
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
    )

    return model.post(**kwargs)
