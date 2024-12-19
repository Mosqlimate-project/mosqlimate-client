import asyncio
from itertools import chain
from typing import AnyStr, List, Any
from urllib.parse import urljoin

import aiohttp
import requests

from mosqlient.types import APP
from mosqlient._config import get_api_url


def get(
    app: APP,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    pagination: bool = True,
    timeout: int = 300,
) -> requests.models.Response:
    if pagination and ("page" not in params or "per_page" not in params):
        raise ValueError("'page' and 'per_page' parameters are required to requests", " with pagination")

    if not endpoint:
        raise ValueError("endpoint is required")

    url = urljoin(get_api_url(), "/".join((str(app), str(endpoint)))) + "/?"

    return requests.get(url, params, timeout=timeout)


async def aget(
    session: aiohttp.ClientSession, url: str, params: dict[str, str | int | float], timeout: int = 300, retries: int = 3
) -> Any:
    try:
        if retries < 0:
            raise aiohttp.ClientConnectionError("Too many attempts")
        async with session.get(url, params=params, timeout=timeout) as res:
            if res.status == 200:
                return await res.json()
            if str(res.status).startswith("4"):
                raise aiohttp.ClientConnectionError(f"Response status: {res.status}. Reason: {res.reason}")
            if retries == 0:
                raise aiohttp.ClientConnectionError(f"Response status: {res.status}. Reason: {res.reason}")
            await asyncio.sleep(10 / (retries + 1))
            return await aget(session, url, params, timeout, retries - 1)
    except aiohttp.ServerTimeoutError:
        await asyncio.sleep(8 / (retries + 1))
        return await aget(session, url, params, timeout, retries - 1)
    raise aiohttp.ClientConnectionError("Invalid request")


async def get_all(
    app: APP,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    timeout: int = 300,
    pagination: bool = False,
    _max_per_page: int = 300,
) -> List[dict]:
    if pagination:
        params["page"] = 1
        params["per_page"] = _max_per_page

    url = urljoin(get_api_url(), "/".join((str(app), str(endpoint)))) + "/?"

    async with aiohttp.ClientSession() as session:
        first_page = await aget(session, url, params)

    if not pagination:
        return first_page

    if not first_page:
        return first_page

    total_pages = first_page["pagination"]["total_pages"]

    if total_pages == 1:
        return first_page["items"]

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        tasks = []
        for page in range(2, total_pages + 1):
            params_c = params.copy()
            params_c["page"] = page
            task = asyncio.create_task(aget(session, url, params_c))
            tasks.append(task)
        results = await asyncio.gather(*tasks)

    if results:
        results.insert(0, first_page)

    res = list(chain.from_iterable(result["items"] for result in results if result is not None))

    return res


def get_all_sync(
    app: APP,
    endpoint: AnyStr,
    params: dict[str, Any],
    timeout: int = 300,
    pagination: bool = False,
    _max_per_page: int = 300,
):
    async def fetch_all():
        return await get_all(app=app, endpoint=endpoint, params=params, timeout=timeout, pagination=pagination)

    if asyncio.get_event_loop().is_running():
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(fetch_all())
        return loop.run_until_complete(future)
    return asyncio.run(fetch_all())
