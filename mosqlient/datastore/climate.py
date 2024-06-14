import pandas as pd
from typing import Optional
from pydantic import BaseModel

from mosqlient import types
from mosqlient.requests import get_datastore
from mosqlient.datastore.api import DataFieldValidator
from mosqlient._utils import parse_params


class Climate:
    @classmethod
    def get(cls, **kwargs):
        cls._validate_fields(**kwargs)
        params = parse_params(**kwargs)

        gen_data = get_datastore("datastore", "climate", params)

        return pd.DataFrame(list(gen_data))

    @staticmethod
    def _validate_fields(**kwargs) -> None:
        DataFieldValidator(**kwargs)


class ClimateGETParams(BaseModel):
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None
    geocode: Optional[types.Geocode] = None
    uf: Optional[types.UF] = None
