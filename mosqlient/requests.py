import asyncio
from itertools import chain
from typing import AnyStr, Literal
from urllib.parse import urljoin

import aiohttp
import requests

from mosqlient.types import APP
from mosqlient._config import API_DEV_URL, API_PROD_URL


def get(
    app: APP,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    pagination: bool = True,
    timeout: int = 10,
    env: Literal["dev", "prod"] = "prod"
) -> requests.models.Response:
    if pagination and ("page" not in params or "per_page" not in params):
        raise ValueError(
            "'page' and 'per_page' parameters are required to requests",
            " with pagination"
        )

    if not endpoint:
        raise ValueError("endpoint is required")

    base_url = API_DEV_URL if env == "dev" else API_PROD_URL

    url = urljoin(base_url, "/".join((str(app), str(endpoint)))) + "/?"

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
                    f"Response status: {res.status}. Reason: {res.reason}")
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

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        tasks = []
        for page in range(2, total_pages + 1):
            params_c = params.copy()
            params_c["page"] = page
            task = asyncio.create_task(aget(session, url, params_c))
            tasks.append(task)
        results = await asyncio.gather(*tasks)

    if results:
        results.insert(0, first_page)

    res = list(chain.from_iterable(
        result["items"] for result in results if result is not None
    ))

    if len(res) == 1:
        return res[0]

    return res


def get_all_sync(
    app: APP,
    endpoint: AnyStr,
    params: dict[str, str | int | float],
    timeout: int = 60,
    _max_per_page: int = 50,
    env: Literal["dev", "prod"] = "prod"
):
    async def fetch_all():
        return await get_all(
            app=app,
            endpoint=endpoint,
            params=params,
            env=env,
            timeout=timeout
        )

    if asyncio.get_event_loop().is_running():
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(fetch_all())
        return loop.run_until_complete(future)
    return asyncio.run(fetch_all())
