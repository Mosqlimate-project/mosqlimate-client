import asyncio
from urllib.parse import urlparse
from typing import Literal, Optional, Any

import requests
import nest_asyncio

from mosqlient import Client
from mosqlient.config import API_PROD_URL, API_DEV_URL
from mosqlient.requests import get_all
from mosqlient.errors import ClientError, ModelPostError

nest_asyncio.apply()


def _params(**kwargs) -> dict[str, Any]:
    params = {}
    for k, v in kwargs.items():
        if isinstance(v, (bool, int, float, str)):
            params[k] = str(v)
        elif v is None:
            continue
        else:
            raise TypeError(f"Unknown type f{type(v)}")

    return params


class Model:
    client: Client | None

    def __init__(self, client: Optional[Client] = None):
        self.client = client

    @classmethod
    def get(cls, **kwargs):
        cls._validate_fields(**kwargs)
        params = _params(**kwargs)

        async def fetch_models():
            return await get_all("registry", "models", params)

        if asyncio.get_event_loop().is_running():
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(fetch_models())
            return loop.run_until_complete(future)
        return asyncio.run(fetch_models())

    def post(
        self,
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
        _env: Literal["dev", "prod"] = "prod"
    ):
        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more infor about "
                "retrieving or inserting data from Mosqlimate, please see the "
                "API Documentation"
            )
        params = {
            "name": name,
            "description": description,
            "repository": repository,
            "implementation_language": implementation_language,
            "disease": disease,
            "temporal": temporal,
            "spatial": spatial,
            "categorical": categorical,
            "ADM_level": adm_level,
            "time_resolution": time_resolution
        }

        self._validate_fields(**params)
        params = _params(**params)
        base_url = API_DEV_URL if _env == "dev" else API_PROD_URL
        url = base_url + "/".join(("registry", "models")) + "/"
        headers = {"X-UID-Key": self.client.X_UID_KEY}
        resp = requests.post(url, json=params, headers=headers, timeout=60)

        if resp.status_code != 201:
            raise ModelPostError(
                f"POST request returned status code {resp.status_code}"
            )

        return resp

    @staticmethod
    def _validate_fields(**kwargs) -> None:
        ModelFieldValidator(**kwargs)


class ModelFieldValidator:
    FIELDS = {
        "id": (int, str),
        "name": str,
        "description": str,
        "author_name": str,
        "author_username": str,
        "author_institution": str,
        "repository": str,
        "implementation_language": str,
        "disease": str,
        "ADM_level": (str, int),
        "temporal": bool,
        "spatial": bool,
        "categorical": bool,
        "time_resolution": str,
    }
    DISEASES = ["dengue", "zika", "chikungunya"]
    ADM_LEVELS = [0, 1, 2, 3]
    TIME_RESOLUTIONS = ["day", "week", "month", "year"]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if not isinstance(v, self.FIELDS[k]):
                raise TypeError(
                    f"Field '{k}' must have instance of "
                    f"{' or '.join(self.FIELDS[k])}"
                )

            if k == "id":
                if v <= 0:
                    raise ValueError("Incorrect value for field 'id'")

            if k == "description":
                if not v:
                    raise ValueError("A Model description must be provided")
                if len(v) > 500:
                    raise ValueError("Model description too long")

            if k == "repository":
                repo_url = urlparse(v)
                if repo_url.netloc != "github.com":
                    raise ValueError(
                        "'repository' must be a valid GitHub repository"
                    )

            if k == "disease":
                if v == "chik":
                    v = "chikungunya"

                if v not in self.DISEASES:
                    raise ValueError(
                        f"Unkown 'disease'. Options: {self.DISEASES}"
                    )

            if k == "ADM_level":
                v = int(v)
                if v not in self.ADM_LEVELS:
                    raise ValueError(
                        f"Unkown 'ADM_level'. Options: {self.ADM_LEVELS}"
                    )

            if k == "time_resolution":
                if v not in self.TIME_RESOLUTIONS:
                    raise ValueError(
                        "Unkown 'time_resolution'. "
                        f"Options: {self.TIME_RESOLUTIONS}"
                    )
