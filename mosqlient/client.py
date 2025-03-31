import re
import uuid
import asyncio
from itertools import chain
from collections import defaultdict
from typing import AnyStr, List, Any, Literal, Optional
from urllib.parse import urljoin

import trio
import aiohttp
import requests
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator
from tqdm.asyncio import tqdm_asyncio

from mosqlient.types import APP, RequestParams
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
        self.endpoints = defaultdict(dict)
        self.__validate_uuid4()
        self.__fetch_openapi()
        self._max_concurrent_requests = 10

    def __str__(self):
        return self.username

    @property
    def X_UID_KEY(self):
        return f"{self.username}:{self.uid_key}"

    def get(
        self,
        app: str,
        endpoint: AnyStr,
        params: Optional[RequestParams] = None,
        page: Optional[int] = None,
    ) -> ...:
        self.__validate_request("GET", app, endpoint, params)
        url = self.api_url + app + "/" + endpoint.strip("/")

        if not params:
            return requests.get(
                url=url,
                headers={"X-UID-Key": self.X_UID_KEY},
                timeout=self.timeout,
            )

        params = params.params

        if not page:
            return self.__get_all_sync(
                url=self.api_url + app + "/" + endpoint.strip("/"),
                params=params,
            )

        params["page"] = page
        params["per_page"] = self.per_page

        return requests.get(
            url=url,
            params=params,
            headers={"X-UID-Key": self.X_UID_KEY},
            timeout=self.timeout,
        )

    def post(
        self,
        app: str,
        endpoint: AnyStr,
        params: RequestParams,
    ) -> requests.models.Response:
        self.__validate_request("POST", app, endpoint, params)
        return requests.post(
            url=self.api_url + app + "/" + endpoint.strip("/"),
            data=params.params,
            headers={"X-UID-Key": self.X_UID_KEY},
            timeout=self.timeout,
        )

    def put(
        self,
        app: str,
        endpoint: AnyStr,
        params: RequestParams,
    ) -> requests.models.Response:
        self.__validate_request("PUT", app, endpoint, params)
        raise NotImplementedError()

    def delete(
        self,
        app: str,
        endpoint: AnyStr,
    ) -> requests.models.Response:
        self.__validate_request("DELETE", app, endpoint)
        return requests.delete(
            url=self.api_url + app + "/" + endpoint.strip("/"),
            headers={"X-UID-Key": self.X_UID_KEY},
            timeout=self.timeout,
        )

    async def __aget(
        self,
        url: str,
        params: dict,
        session: aiohttp.ClientSession,
        retries: int = 3,
    ) -> Any:
        headers = {"X-UID-Key": self.X_UID_KEY}
        try:
            if retries < 0:
                raise aiohttp.ClientConnectionError("Too many attempts")
            async with session.get(url, params=params, headers=headers) as res:
                if res.status == 200:
                    return await res.json()
                if str(res.status).startswith("4"):
                    raise aiohttp.ClientConnectionError(
                        f"Response status: {res.status}. Reason: {res.reason}"
                    )
                if retries == 0:
                    raise aiohttp.ClientConnectionError(
                        f"Response status: {res.status}. Reason: {res.reason}"
                    )
                await asyncio.sleep(10 / (retries + 1))
                return await self.__aget(session, url, params, retries - 1)
        except aiohttp.ServerTimeoutError:
            await asyncio.sleep(8 / (retries + 1))
            return await self.__aget(session, url, params, retries - 1)
        raise aiohttp.ClientConnectionError("Invalid request")

    async def __get_all(self, url: str, params: dict):
        semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:

            async def fetch_page(page):
                params_c = params.copy()
                params_c["page"] = page
                async with semaphore:
                    return await self.__aget(session, url, params_c)

            first_page = await self.__aget(session, url, params)
            total_pages = first_page["pagination"]["total_pages"]

            tasks = [fetch_page(page) for page in range(2, total_pages + 1)]
            results = await tqdm_asyncio.gather(*tasks, total=total_pages - 1)

            if results:
                results.insert(0, first_page)
            res = list(
                chain.from_iterable(
                    result["items"] for result in results if result is not None
                )
            )
            return res

    def __get_all_sync(
        self,
        url: str,
        params: dict,
    ):
        async def fetch_all():
            return await self.__get_all(
                url=url,
                params=params,
            )

        return trio.run(fetch_all)

    def __fetch_openapi(self):
        self.openapi = requests.get(
            f"{self.api_url}" + "openapi.json", timeout=self.timeout
        ).json()
        for path, methods in self.openapi["paths"].items():
            _, app, *endpoint = path.strip("/").split("/")
            endpoint = "/".join(endpoint)
            self.endpoints[app][endpoint] = list(methods.keys())

    def __validate_request(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE"],
        app: str,
        endpoint: str,
        params: Optional[RequestParams] = None,
    ) -> None:
        apps = list(self.endpoints.keys())

        if app not in apps:
            raise errors.ParameterError(f"Unknown app '{app}'", options=apps)

        self.__validate_endpoint(method, app, endpoint)

        if not params:
            return

        if not isinstance(params, RequestParams):
            raise TypeError(
                "`params` must be of type mosqlient.types.RequestParams"
            )

        if params._method.lower() != method.lower():
            raise TypeError(f"{type(params)} doesn't allow {method} requests")

        if params._app != app:
            raise TypeError(
                f"{type(params)} doesn't allow requests to '{app}'"
            )

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
                if not method.lower() in self.endpoints[app][e]:
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
