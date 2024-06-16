from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict

from mosqlient import types
from mosqlient.requests import get_all_sync
from mosqlient._utils import parse_params


class Climate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get(cls, start: str | date, end: str | date, **kwargs):
        env = kwargs["env"] if "env" in kwargs else "prod"
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        kwargs["start"] = start
        kwargs["end"] = end
        params = parse_params(**kwargs)
        ClimateGETParams(**kwargs)

        return get_all_sync(
            app="datastore",
            endpoint="climate",
            params=params,
            env=env,
            timeout=timeout
        )


class ClimateGETParams(BaseModel):
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None
    geocode: Optional[types.Geocode] = None
    uf: Optional[types.UF] = None
