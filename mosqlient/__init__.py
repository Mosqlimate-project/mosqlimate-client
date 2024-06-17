"""
Mosqlimate client package.

client library for the Mosqlimate project data platform.
"""

from importlib import metadata as importlib_metadata
from typing import List

from mosqlient.client import Client
from mosqlient.registry._model_get_impl import *
from mosqlient.registry._model_post_impl import *
from mosqlient.registry._prediction_get_impl import *
from mosqlient.registry._prediction_post_impl import *
from mosqlient.datastore._climate_get_impl import *
from mosqlient.datastore._infodengue_get_impl import *


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "0.1.0"  # changed by semantic-release


version: str = get_version()
__version__: str = version
__all__: List[str] = []  # noqa: WPS410 (the only __variable__ we use)
