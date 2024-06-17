import asyncio
from typing import Literal, Optional

import requests
import nest_asyncio
from pydantic import BaseModel, ConfigDict

from mosqlient import types
from mosqlient.client import Client
from mosqlient.errors import ClientError, ModelPostError
from mosqlient.requests import get_all_sync
from mosqlient._utils import parse_params
from mosqlient._config import API_DEV_URL, API_PROD_URL


nest_asyncio.apply()


class Model(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    client: Client | None

    @classmethod
    def get(cls, **kwargs):
        """
        https://api.mosqlimate.org/docs/registry/GET/models/
        """
        env = kwargs["env"] if "env" in kwargs else "prod"
        timeout = kwargs["timeout"] if "timeout" in kwargs else 60

        ModelGETParams(**kwargs)
        params = parse_params(**kwargs)

        return get_all_sync(
            app="registry",
            endpoint="models",
            params=params,
            env=env,
            timeout=timeout
        )

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
        **kwargs,
    ):
        """
        https://api.mosqlimate.org/docs/registry/POST/models/
        """
        timeout = kwargs["timeout"] if "timeout" in kwargs else 10

        if self.client is None:
            raise ClientError(
                "A Client instance must be provided, please instantiate Model "
                "passing your Mosqlimate's credentials. For more info about "
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
            "time_resolution": time_resolution,
        }

        ModelPOSTParams(
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
        )

        base_url = API_DEV_URL if self.client.env == "dev" else API_PROD_URL
        url = base_url + "/".join(("registry", "models")) + "/"
        headers = {"X-UID-Key": self.client.X_UID_KEY}

        resp = requests.post(
            url,
            json=params,
            headers=headers,
            timeout=timeout
        )

        if resp.status_code != 201:
            raise ModelPostError(
                "POST request returned status code "
                f"{resp.status_code} \n {resp.reason}"
            )

        return resp

    def update(
        self,
        id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        repository: Optional[str] = None,
        implementation_language: Optional[str] = None,
        disease: Optional[Literal["dengue", "chikungunya", "zika"]] = None,
        temporal: Optional[bool] = None,
        spatial: Optional[bool] = None,
        categorical: Optional[bool] = None,
        adm_level: Optional[Literal[0, 1, 2, 3]] = None,
        # fmt: off
        time_resolution: Optional[Literal["day", "week", "month", "year"]] = None,
        # fmt: on
        **kwargs
    ):
        """
        https://github.com/Mosqlimate-project/Data-platform/blob/main/src/registry/api.py#L258
        """
        timeout = kwargs["timeout"] if "timeout" in kwargs else 10

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

        ModelPUTParams(
            id=id,
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
        )

        base_url = API_DEV_URL if self.client.env == "dev" else API_PROD_URL
        url = base_url + "/".join(("registry", "models")) + f"/{id}"
        headers = {"X-UID-Key": self.client.X_UID_KEY}
        resp = requests.put(url, json=params, headers=headers, timeout=timeout)

        return resp


class ModelGETParams(BaseModel):
    # https://github.com/Mosqlimate-project/Data-platform/blob/main/src/registry/schema.py#L43
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    id: Optional[types.ID] = None
    name: Optional[types.Name] = None
    author_name: Optional[types.AuthorName] = None
    author_username: Optional[types.AuthorUserName] = None
    author_institution: Optional[types.AuthorInstitution] = None
    repository: Optional[types.Repository] = None
    implementation_language: Optional[types.ImplementationLanguage] = None
    disease: Optional[types.Disease] = None
    ADM_level: Optional[types.ADMLevel] = None
    temporal: Optional[types.Temporal] = None
    spatial: Optional[types.Spatial] = None
    categorical: Optional[types.Categorical] = None
    time_resolution: Optional[types.TimeResolution] = None
    tags: Optional[types.Tags] = None


class ModelPOSTParams(BaseModel):
    # https://github.com/Mosqlimate-project/Data-platform/blob/main/src/registry/api.py#L154
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    name: types.Name
    description: Optional[types.Description] = None
    repository: types.Repository
    implementation_language: types.ImplementationLanguage
    disease: types.Disease
    temporal: types.Temporal
    spatial: types.Spatial
    categorical: types.Categorical
    ADM_level: types.ADMLevel
    time_resolution: types.TimeResolution


class ModelPUTParams(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    id: types.ID
    name: Optional[types.Name] = None
    description: Optional[types.Description] = None
    repository: Optional[types.Repository] = None
    implementation_language: Optional[types.ImplementationLanguage] = None
    disease: Optional[types.Disease] = None
    ADM_level: Optional[types.ADMLevel] = None
    temporal: Optional[types.Temporal] = None
    spatial: Optional[types.Spatial] = None
    categorical: Optional[types.Categorical] = None
    time_resolution: Optional[types.TimeResolution] = None
