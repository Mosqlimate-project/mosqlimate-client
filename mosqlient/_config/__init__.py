from .globals import *  # noqa


def get_api_url() -> str:
    return API_URL


def set_api_url(url: str) -> None:
    global API_URL
    API_URL = url
