__all__ = ["Mosqlient", "Client"]

import re
import uuid
import asyncio
import json
from collections import defaultdict
from itertools import chain
from typing import Literal, List

from aiohttp import (
    ClientSession,
    ClientTimeout,
    ClientConnectionError,
    ServerTimeoutError,
)
import requests
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator
from tqdm.asyncio import tqdm_asyncio
from loguru import logger

from mosqlient.types import Params
from mosqlient._utils import parse_params
from mosqlient import errors


class Mosqlient:
    def __init__(
        self,
        x_uid_key: str,
        timeout: int = 300,
        max_items_per_page: int = 300,
        _api_url: str = "https://api.mosqlimate.org/api/",
    ):
        self.username, self.uid_key = x_uid_key.split(":")
        self.timeout = timeout
        self.per_page = max_items_per_page
        self.api_url = _api_url
        self.endpoints: dict = defaultdict(dict)
        self.__validate_uuid4()
        self.__fetch_openapi()
        self._max_concurrent_requests = 10

    def __str__(self):
        return self.username

    @property
    def X_UID_KEY(self):
        return f"{self.username}:{self.uid_key}"

    def get(self, params: Params) -> List[dict]:
        self.__validate_request(params)
        url = self.api_url + params.app + "/" + params.endpoint.strip("/")

        if not hasattr(params, "page"):
            res = requests.get(
                url=url,
                headers={"X-UID-Key": self.X_UID_KEY},
                timeout=self.timeout,
            )
            res.raise_for_status()
            return res.json()

        _params = params.params()

        if "per_page" not in _params:
            _params["per_page"] = self.per_page

        if _params["per_page"] > self.per_page:
            logger.warning(f"Maximum itens per page set to {self.per_page}")
            _params["per_page"] = self.per_page

        if "page" in _params:
            res = requests.get(
                url=url,
                params=parse_params(**_params),
                headers={"X-UID-Key": self.X_UID_KEY},
                timeout=self.timeout,
            )
            res.raise_for_status()
            data = res.json()
            if "message" in data:
                logger.warning(data["message"])
            return data["items"]

        return self.__get_all_sync(url=url, params=parse_params(**_params))

    def post(self, params: Params) -> requests.models.Response:
        self.__validate_request(params)
        headers = {
            "X-UID-Key": self.X_UID_KEY,
            "Content-Type": "application/json",
        }
        res = requests.post(
            url=self.api_url + params.app + "/" + params.endpoint + "/",
            data=json.dumps(params.params()),
            timeout=self.timeout,
            headers=headers,
        )
        if res.status_code == 422:
            raise ValueError(res.text)
        try:
            res.raise_for_status()
        except requests.HTTPError as err:
            logger.error(res.text)
            raise err
        return res

    def put(self, params: Params) -> requests.models.Response:
        self.__validate_request(params)
        raise NotImplementedError()

    def delete(self, params: Params) -> requests.models.Response:
        self.__validate_request(params)
        res = requests.delete(
            url=self.api_url + params.app + "/" + params.endpoint.strip("/"),
            headers={"X-UID-Key": self.X_UID_KEY},
            timeout=self.timeout,
        )
        try:
            res.raise_for_status()
        except requests.HTTPError as err:
            logger.error(res.text)
            raise err
        return res

    async def __aget(
        self,
        session: ClientSession,
        url: str,
        params: dict,
        retries: int = 3,
    ) -> dict:
        headers = {"X-UID-Key": self.X_UID_KEY}
        try:
            if retries < 0:
                raise ClientConnectionError("Too many attempts")
            async with session.get(url, params=params, headers=headers) as res:
                if res.status == 200:
                    return await res.json()
                res.raise_for_status()
                await asyncio.sleep(10 / (retries + 1))
                return await self.__aget(session, url, params, retries - 1)
        except ServerTimeoutError:
            await asyncio.sleep(8 / (retries + 1))
            return await self.__aget(session, url, params, retries - 1)
        raise ClientConnectionError("Invalid request")

    async def __get_all(self, url: str, params: dict) -> List[dict]:
        async with ClientSession(
            timeout=ClientTimeout(total=self.timeout)
        ) as session:
            params["page"] = 1
            first_page = await self.__aget(session, url, params)
            total_pages = first_page["pagination"]["total_pages"]

            results = [first_page]

            tasks = []
            for page in range(2, total_pages + 1):
                params_c = params.copy()
                params_c["page"] = page
                task = asyncio.create_task(self.__aget(session, url, params_c))
                tasks.append(task)

            if tasks:
                results.extend(
                    await tqdm_asyncio.gather(
                        *tasks, total=total_pages - 1, unit="requests"
                    )
                )

            return list(chain.from_iterable(res["items"] for res in results))

    def __get_all_sync(self, url: str, params: dict) -> List[dict]:
        async def fetch_all():
            return await self.__get_all(url=url, params=params)

        if asyncio.get_event_loop().is_running():
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(fetch_all())
            return loop.run_until_complete(future)
        return asyncio.run(fetch_all())

    def __fetch_openapi(self):
        self.openapi = requests.get(
            f"{self.api_url}" + "openapi.json", timeout=self.timeout
        ).json()
        for path, methods in self.openapi["paths"].items():
            _, app, *endpoint = path.strip("/").split("/")
            endpoint = "/".join(endpoint)
            self.endpoints[app][endpoint] = list(methods.keys())

    def __validate_request(self, params: Params) -> None:
        apps = list(self.endpoints.keys())

        if params.app not in apps:
            raise errors.ParameterError(
                f"Unknown app '{params.app}'", options=apps
            )

        self.__validate_endpoint(params.method, params.app, params.endpoint)

        if not params:
            return

        if not isinstance(params, Params):
            raise TypeError("`params` must be of type mosqlient.types.Params")

    def __validate_endpoint(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE"],
        app: str,
        endpoint: str,
    ) -> None:
        endpoints = list(self.endpoints[app].keys())
        options = [f"{k} {v}" for k, v in self.endpoints[app].items()]

        if endpoint in endpoints:
            return

        for e in endpoints:
            pat = "^" + re.sub(r"\{[^}]+\}", r"[^/]+", e) + "$"
            if re.match(pat, endpoint):
                if method.lower() not in self.endpoints[app][e]:
                    raise errors.ParameterError(
                        f"Invalid method '{method}' for endpoint '{endpoint}'",
                        options=options,
                    )
                return

        raise errors.ParameterError(
            f"Unknown endpoint '{endpoint}' for app {app}", options=options
        )

    def __validate_uuid4(self):
        docs_url = "https://api.mosqlimate.org/docs/uid-key/"
        try:
            uuid.UUID(self.uid_key, version=4)
        except ValueError:
            raise ValueError(f"uid_key is not a valid key. See {docs_url}")


# Avoiding circular imports
def validate_client(c: Mosqlient) -> Mosqlient:
    return c


Client = Annotated[Mosqlient, AfterValidator(validate_client)]
