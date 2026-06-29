__all__ = ["Mosqlient", "Client"]

import re
import uuid
import asyncio
import json
import os
import time
from collections import defaultdict
from itertools import chain
from typing import Literal, List, Optional

from aiohttp import (
    ClientSession,
    ClientTimeout,
    ClientConnectionError,
    ServerTimeoutError,
    ClientResponseError,
)
import requests
from dotenv import load_dotenv
from typing_extensions import Annotated
from pydantic.functional_validators import AfterValidator
from pydantic_core import to_jsonable_python
from tqdm.asyncio import tqdm_asyncio
from loguru import logger

from mosqlient.types import Params
from mosqlient._utils import parse_params
from mosqlient import errors

load_dotenv()


class Mosqlient:
    def __init__(
        self,
        x_uid_key: Optional[str] = None,
        timeout: int = 300,
        max_items_per_page: int = 300,
        _api_url: str = os.getenv(
            "MOSQLIENT_API_URL",
            "https://api.mosqlimate.org/api/",
        ),
    ):
        self.timeout = timeout
        self.per_page = max_items_per_page
        self.api_url = _api_url
        self.endpoints: dict = defaultdict(dict)
        self._max_concurrent_requests = 10

        self._token_capacity = 0
        self._tokens = 0
        self._token_refill_rate = 0
        self._last_request_time = time.monotonic()
        self._delay_between_requests = 0

        self.is_demo_user = False

        if x_uid_key:
            self.username, self.uid_key = x_uid_key.split(":")
        else:
            self.is_demo_user = True
            self.__fetch_temp_credentials()

        self.__validate_uuid4()
        self.__fetch_openapi()
        self.__fetch_rate_limit()

    def __str__(self):
        return self.username

    @property
    def X_UID_KEY(self):
        return f"{self.username}:{self.uid_key}"

    def __fetch_temp_credentials(self):
        url = self.api_url + "user/create-temp-user/"
        res = requests.post(url, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()
        self.username, self.uid_key = data["api_key"].split(":")

        logger.warning(
            "You are using a demo version of the API. "
            "If you want a full version, please register at www.mosqlimate.org "
            "and use your own api_key."
        )

    def __fetch_rate_limit(self):
        url = self.api_url + "user/rate-limit/"
        res = requests.get(
            url, headers={"X-UID-Key": self.X_UID_KEY}, timeout=self.timeout
        )
        if res.status_code == 200:
            rate_limit_str = res.json().get("rate_limit", "unlimited")
            if rate_limit_str != "unlimited":
                try:
                    count_str, period = rate_limit_str.split("/")
                    count = int(count_str)
                    period_seconds = {
                        "s": 1,
                        "m": 60,
                        "h": 3600,
                        "d": 86400,
                    }.get(period, 60)

                    self._token_capacity = count
                    self._tokens = count
                    self._token_refill_rate = count / period_seconds
                    self._last_request_time = time.monotonic()
                except (ValueError, AttributeError):
                    pass

    def get(self, params: Params) -> List[dict]:
        self.__validate_request(params)
        url = self.api_url + params.app + "/" + params.endpoint.strip("/")

        if not hasattr(params, "page"):
            res = requests.get(
                url=url,
                params=parse_params(**params.params()),
                headers={"X-UID-Key": self.X_UID_KEY},
                timeout=self.timeout,
            )
            if res.status_code == 422:
                raise ValueError(res.text)
            try:
                res.raise_for_status()
            except requests.HTTPError as err:
                logger.error(res.text)
                raise err
            data = res.json()
            if isinstance(data, dict):
                return data.get("items", data)
            return data

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
            if res.status_code == 422:
                raise ValueError(res.text)
            try:
                res.raise_for_status()
            except requests.HTTPError as err:
                logger.error(res.text)
                raise err
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
            data=json.dumps(to_jsonable_python(params.params())),
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

    def patch(self, params: Params) -> requests.models.Response:
        self.__validate_request(params)
        headers = {
            "X-UID-Key": self.X_UID_KEY,
            "Content-Type": "application/json",
        }
        res = requests.patch(
            url=(
                self.api_url
                + params.app
                + "/"
                + params.endpoint.strip("/")
                + "/"
            ),
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

    async def __throttle(self):
        if self._token_capacity <= 0:
            return

        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            self._last_request_time = now

            self._tokens = min(
                self._token_capacity,
                self._tokens + (elapsed * self._token_refill_rate),
            )

            if self._tokens >= 1:
                self._tokens -= 1
                return

            wait_time = (1 - self._tokens) / self._token_refill_rate
            self._tokens -= 1
            await asyncio.sleep(wait_time)

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

            await self.__throttle()

            async with session.get(url, params=params, headers=headers) as res:
                if res.status == 200:
                    return await res.json()

                error_text = await res.text()
                error_message = error_text

                try:
                    error_json = json.loads(error_text)
                    if isinstance(error_json, dict):
                        error_message = str(
                            error_json.get(
                                "message", error_json.get("detail", error_text)
                            )
                        )
                except json.JSONDecodeError:
                    pass

                raise ClientResponseError(
                    request_info=res.request_info,
                    history=res.history,
                    status=res.status,
                    message=str(error_message),
                    headers=res.headers,
                )

        except ServerTimeoutError:
            await asyncio.sleep(8 / (retries + 1))
            return await self.__aget(session, url, params, retries - 1)

    async def __get_all(self, url: str, params: dict) -> List[dict]:
        self._request_lock = asyncio.Lock()
        self._last_request_time = time.monotonic()

        async with ClientSession(
            timeout=ClientTimeout(total=self.timeout)
        ) as session:
            params["page"] = 1
            first_page = await self.__aget(session, url, params)
            total_pages = first_page["pagination"]["total_pages"]

            results = [first_page]

            if self.is_demo_user and total_pages > 1:
                logger.warning(
                    "Demo users are limited to the first page of results. "
                    "To access all pages, please register at www.mosqlimate.org "
                    "and use your own api_key."
                )
                return list(
                    chain.from_iterable(res["items"] for res in results)
                )

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
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"],
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


def validate_client(c: Mosqlient) -> Mosqlient:
    return c


Client = Annotated[Mosqlient, AfterValidator(validate_client)]
