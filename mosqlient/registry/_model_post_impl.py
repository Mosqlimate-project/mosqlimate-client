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
    """
    Upload a new model to the Mosqlimate platform with the specified metadata.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        name : str
            Name of the model to be registered.
        description : str
            A brief description of the model's purpose and behavior.
        repository : str
            Name of the repository where the model's source code is stored.
        implementation_language : types.ImplementationLanguage
            Programming language used to implement the model.
        disease : {'dengue', 'chikungunya', 'zika'}
            Disease the model is designed to work with.
        temporal : bool
            Indicates whether the model is temporal.
        spatial : bool
            Indicates whether the model is spatial.
        categorical : bool
            Indicates whether the model is categorical.
        adm_level : {0, 1, 2, 3}
            Administrative level the model is designed for:
            - 0: National
            - 1: State
            - 2: Municipality
            - 3: Sub-municipality
        time_resolution : {'day', 'week', 'month', 'year'}
            Temporal resolution at which the model produces outputs.
        sprint : bool
            Indicates whether the model is associated with a sprint challenge or event.

    Returns
    -------
        requests.Response
            HTTP response object returned by the API after attempting to upload the model.
    """
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
