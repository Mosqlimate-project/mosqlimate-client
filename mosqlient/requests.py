import asyncio
import requests
from urllib.parse import urljoin
from typing import AnyStr, Literal
from itertools import chain

import aiohttp

from mosqlient.config import API_DEV_URL, API_PROD_URL, APPS, APPS_TYPE
from mosqlient.utils import validate


def get(
    app: APPS_TYPE,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    pagination: bool = True,
    timeout: int = 10,
    _env: Literal["dev", "prod"] = "prod"
) -> requests.models.Response:
    if app not in APPS:
        raise ValueError(f"unkown Mosqlimate app. Options: {APPS}")

    if pagination:
        validate.url_pagination(params)

    if not endpoint:
        raise ValueError("endpoint is required")

    base_url = API_DEV_URL if _env == "dev" else API_PROD_URL

    url = urljoin(
        base_url,
        "/".join((str(app), str(endpoint)))
    ) + "/?"

    return requests.get(url, params, timeout=timeout)


async def aget(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, str | int | float],
    timeout: int = 10,
    retries: int = 3
) -> dict:
    try:
        if retries < 0:
            raise aiohttp.ClientConnectionError("Too many attempts")
        async with session.get(url, params=params, timeout=timeout) as res:
            if res.status == 200:
                return await res.json()
            if retries == 0:
                raise aiohttp.ClientConnectionError(
                    f"Response status: {res.status}. Reason: {res.reason}"
                )
            await asyncio.sleep(10/(retries + 1))
            await aget(session, url, params, timeout, retries - 1)
    except aiohttp.ServerTimeoutError:
        await asyncio.sleep(8/(retries + 1))
        await aget(session, url, params, timeout, retries - 1)
    raise aiohttp.ClientConnectionError("Invalid request")


async def get_all(
    app: APPS_TYPE,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    timeout: int = 60,
    _max_per_page: int = 50,
    _env: Literal["dev", "prod"] = "prod"
) -> list[dict]:
    params["page"] = 1
    params["per_page"] = _max_per_page

    base_url = API_DEV_URL if _env == "dev" else API_PROD_URL

    url = urljoin(
        base_url,
        "/".join((str(app), str(endpoint)))
    ) + "/?"

    async with aiohttp.ClientSession() as session:
        first_page = await aget(session, url, params)

    total_pages = first_page['pagination']['total_pages']

    if total_pages == 1:
        return first_page['items']

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for page in range(2, total_pages + 1):
            params["page"] = page
            task = asyncio.create_task(aget(session, url, params))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
    return list(chain(result['items'] for result in results))
