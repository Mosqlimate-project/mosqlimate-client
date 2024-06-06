import asyncio
import concurrent.futures
import time
from itertools import chain
from typing import AnyStr, Generator, Literal
from urllib.parse import urljoin

import aiohttp
import requests

from mosqlient.config import API_DEV_URL, API_PROD_URL
from mosqlient.types import APP
from mosqlient.utils import validate


def get(
    app: APP,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    pagination: bool = True,
    timeout: int = 10,
    env: Literal["dev", "prod"] = "prod"
) -> requests.models.Response:
    if pagination:
        validate.url_pagination(params)

    if not endpoint:
        raise ValueError("endpoint is required")

    base_url = API_DEV_URL if env == "dev" else API_PROD_URL

    url = urljoin(base_url, "/".join((str(app), str(endpoint)))) + "/?"

    return requests.get(url, params, timeout=timeout)


async def aget(
    session: aiohttp.ClientSession, url: str, params: dict[str, str | int | float], timeout: int = 10, retries: int = 3
) -> dict:
    try:
        if retries < 0:
            raise aiohttp.ClientConnectionError("Too many attempts")
        async with session.get(url, params=params, timeout=timeout) as res:
            if res.status == 200:
                return await res.json()
            if retries == 0:
                raise aiohttp.ClientConnectionError(
                    f"Response status: {res.status}. Reason: {res.reason}")
            await asyncio.sleep(10 / (retries + 1))
            await aget(session, url, params, timeout, retries - 1)
    except aiohttp.ServerTimeoutError:
        await asyncio.sleep(8 / (retries + 1))
        await aget(session, url, params, timeout, retries - 1)
    raise aiohttp.ClientConnectionError("Invalid request")


async def get_all(
    app: APP,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    timeout: int = 60,
    _max_per_page: int = 50,
    env: Literal["dev", "prod"] = "prod"
) -> list[dict]:
    params["page"] = 1
    params["per_page"] = _max_per_page

    base_url = API_DEV_URL if env == "dev" else API_PROD_URL

    url = urljoin(base_url, "/".join((str(app), str(endpoint)))) + "/?"

    async with aiohttp.ClientSession() as session:
        first_page = await aget(session, url, params)

    total_pages = first_page["pagination"]["total_pages"]

    if total_pages == 1:
        return first_page["items"]

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for page in range(2, total_pages + 1):
            params["page"] = page
            task = asyncio.create_task(aget(session, url, params))
            tasks.append(task)
        results = await asyncio.gather(*tasks)

    res = list(chain(result["items"] for result in results))

    if len(res) == 1:
        return res[0]

    return res


def compose_url(base_url: str, parameters: dict, page: int = 1) -> str:
    """Helper method to compose the API url with parameters"""
    url = base_url + "?" if not base_url.endswith("?") else base_url
    params = "&".join(
        [f"{p}={v}" for p, v in parameters.items()]) + f"&page={page}"
    return url + params


def fetch_data(session: requests.Session, url: str):
    """Uses ClientSession to create the async call to the API"""
    response = session.get(url)
    return response.json()


def attempt_delay(session: requests.Session, url: str):
    """The request may fail. This method adds a delay to the failing requests"""
    try:
        return fetch_data(session, url)
    except Exception as e:
        time.sleep(0.2)
        return attempt_delay(session, url)


def get_datastore(
    app: APP, endpoint: AnyStr, params: dict[str, str | int | float], _env: Literal["dev", "prod"] = "prod"
) -> Generator[dict, None, None]:

    base_url = API_DEV_URL if _env == "dev" else API_PROD_URL

    base_url = urljoin(base_url, "/".join((str(app), str(endpoint)))) + "/?"

    result = []
    with requests.Session() as session:
        url = compose_url(base_url, params)
        data = attempt_delay(session, url)
        total_pages = data["pagination"]["total_pages"]
        result.extend(data["items"])
        if total_pages == 1:
            for i in data["items"]:
                yield i

        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Start the load operations and mark each future with its URL
            for page in range(2, total_pages + 1):
                url = compose_url(base_url, params, page)
                futures[executor.submit(attempt_delay, session, url)] = url
            for future in concurrent.futures.as_completed(futures):
                _ = futures[future]
                resp = future.result()
                if result:  # incorporating the first page
                    resp["items"].extend(result)
                    result = []
                for i in resp["items"]:
                    yield i
