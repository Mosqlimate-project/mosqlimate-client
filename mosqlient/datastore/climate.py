import asyncio
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
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        kwargs["start"] = start
        kwargs["end"] = end
        params = parse_params(**kwargs)
        ClimateGETParams(**kwargs)

        if bench == "t":
            return list(get_datastore("datastore", "climate", params, _env=env))
        else:
            async def fetch_climate():
                return await get_all(
                    "datastore",
                    "climate",
                    params,
                    env=env,
                    timeout=timeout
                )

            if asyncio.get_event_loop().is_running():
                loop = asyncio.get_event_loop()
                future = asyncio.ensure_future(fetch_climate())
                return loop.run_until_complete(future)
            return asyncio.run(fetch_climate())


class ClimateGETParams(BaseModel):
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None
    geocode: Optional[types.Geocode] = None
    uf: Optional[types.UF] = None
