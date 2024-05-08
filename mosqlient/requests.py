import asyncio
import requests
from urllib.parse import urljoin
from typing import Literal, AnyStr

import aiohttp

from mosqlient.config import API_BASE_URL, APPS
from mosqlient.utils import validate


def get(
    app: Literal[APPS],
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    pagination: bool = True,
    timeout: int = 10,
) -> requests.models.Response:
    if app not in APPS:
        raise ValueError(f"unkown Mosqlimate app. Options: {APPS}")

    if pagination:
        validate.url_pagination(params)

    if not endpoint:
        raise ValueError("endpoint is required")

    base_url = urljoin(API_BASE_URL, "/".join((app, endpoint))) + "/?"

    return requests.get(base_url, params, timeout=timeout)


async def _aget(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, str | int | float],
    timeout: int = 10,
    retries: int = 3
):
    try:
        if retries < 0:
            raise aiohttp.ClientConnectionError("Too many attempts")
        async with session.get(url, params=params, timeout=timeout) as res:
            if res.status == 200:
                return await res.json()
            asyncio.sleep(10/(retries + 1))
            await _aget(session, url, params, timeout, retries - 1)
    except aiohttp.TimeoutError:
        asyncio.sleep(10/(retries + 1))
        await _aget(session, url, params, timeout, retries - 1)


def get_all(
    app: Literal[APPS],
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    timeout: int = 60,
    _max_per_page: int = 50
) -> dict:
    params["page"] = 1
    params["per_page"] = _max_per_page
    res = get(app, endpoint, params)

    if res.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"{res.url} returned {res.status_code}"
        )

    total_pages = res.json()['pagination']['total_pages']
