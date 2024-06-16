from datetime import date
from typing import Optional, Literal

from pydantic import BaseModel

from mosqlient import types
from mosqlient.requests import get_all_sync
from mosqlient._utils import parse_params


class Infodengue(BaseModel):
    @classmethod
    def get(
        cls,
        disease: Literal["dengue", "zika", "chikungunya"],
        start: str | date,
        end: str | date,
        **kwargs
    ):
        env = kwargs["env"] if "env" in kwargs else "prod"
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        kwargs["disease"] = disease
        kwargs["start"] = start
        kwargs["end"] = end
        params = parse_params(**kwargs)
        InfodengueGETParams(**params)

        return get_all_sync(
            app="datastore",
            endpoint="infodengue",
            params=params,
            env=env,
            timeout=timeout
        )


class InfodengueGETParams(BaseModel):
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None
    disease: Optional[types.Disease] = None
    uf: Optional[types.UF] = None
    geocode: Optional[types.Geocode] = None
