"""Tests for the datastore models module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import date


class AsyncContextManagerMock:
    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


def make_aiohttp_patch(item_list):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(
        return_value={
            "items": item_list,
            "pagination": {"total_pages": 1, "total_items": len(item_list)},
        }
    )
    mock_resp.raise_for_status = MagicMock()
    session_mock = AsyncMock()
    session_mock.get = MagicMock(
        return_value=AsyncContextManagerMock(mock_resp)
    )
    session_mock.__aenter__ = AsyncMock(return_value=session_mock)
    session_mock.__aexit__ = AsyncMock(return_value=None)
    return session_mock


from mosqlient.datastore.models import (
    Infodengue,
    Climate,
    ClimateWeekly,
    Episcanner,
    Mosquito,
)


class TestInfodengue:
    def test_get(
        self, valid_api_key, mock_openapi_response, sample_infodengue_data
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {
                "items": sample_infodengue_data
            }
            mock_response_2.status_code = 200
            mock_get.side_effect = [mock_response, mock_response_2]

            session_mock = make_aiohttp_patch(sample_infodengue_data)
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                result = Infodengue.get(
                    api_key=valid_api_key,
                    disease="dengue",
                    start=date(2023, 1, 1),
                    end=date(2023, 12, 31),
                )
                assert isinstance(result, list)


class TestClimate:
    def test_get(
        self, valid_api_key, mock_openapi_response, sample_climate_data
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {"items": sample_climate_data}
            mock_response_2.status_code = 200
            mock_get.side_effect = [mock_response, mock_response_2]

            session_mock = make_aiohttp_patch(sample_climate_data)
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                result = Climate.get(
                    api_key=valid_api_key,
                    start=date(2023, 1, 1),
                    end=date(2023, 12, 31),
                )
                assert isinstance(result, list)


class TestClimateWeekly:
    def test_get(
        self, valid_api_key, mock_openapi_response, sample_climate_data
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {"items": sample_climate_data}
            mock_response_2.status_code = 200
            mock_get.side_effect = [mock_response, mock_response_2]

            session_mock = make_aiohttp_patch(sample_climate_data)
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                result = ClimateWeekly.get(
                    api_key=valid_api_key,
                    start="202401",
                    end="202452",
                )
                assert isinstance(result, list)


class TestEpiscanner:
    def test_get(self, valid_api_key, mock_openapi_response):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {"items": []}
            mock_response_2.status_code = 200
            mock_get.side_effect = [mock_response, mock_response_2]

            session_mock = make_aiohttp_patch([])
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                result = Episcanner.get(
                    api_key=valid_api_key,
                    disease="dengue",
                    uf="SP",
                )
                assert isinstance(result, list)


class TestMosquito:
    def test_get(self, valid_api_key, mock_openapi_response):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {"items": []}
            mock_response_2.status_code = 200
            mock_get.side_effect = [mock_response, mock_response_2]

            session_mock = make_aiohttp_patch([])
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                result = Mosquito.get(
                    api_key=valid_api_key,
                    date_start=date(2024, 1, 1),
                    date_end=date(2024, 12, 31),
                )
                assert isinstance(result, list)
