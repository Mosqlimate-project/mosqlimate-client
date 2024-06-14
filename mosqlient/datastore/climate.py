import pandas as pd
from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict

from mosqlient import types
from mosqlient.requests import get_datastore, get_all
from mosqlient._utils import parse_params


class Climate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get(cls, start: str | date, end: str | date, bench: str, **kwargs):
        env = kwargs["env"] if "env" in kwargs else "prod"

        kwargs["start"] = start
        kwargs["end"] = end
        params = parse_params(**kwargs)
        ClimateGETParams(**kwargs)

        if bench == "threads":
            gen_data = get_datastore("datastore", "climate", params, _env=env)
        else:
            gen_data = get_all("datastore", "climate", params, env=env)

        return gen_data


class ClimateGETParams(BaseModel):
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None
    geocode: Optional[types.Geocode] = None
    uf: Optional[types.UF] = None
