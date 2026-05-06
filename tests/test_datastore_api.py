"""Tests for the datastore convenience API functions."""

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


from mosqlient.datastore._climate_get_impl import (
    get_climate,
    get_climate_weekly,
)
from mosqlient.datastore._infodengue_get_impl import get_infodengue
from mosqlient.datastore._episcanner_get_impl import get_episcanner
from mosqlient.datastore._mosquito_get_impl import get_mosquito


class TestGetClimate:
    def test_get_climate_geocode(
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
                result = get_climate(
                    api_key=valid_api_key,
                    start_date=date(2023, 1, 1),
                    end_date=date(2023, 12, 31),
                    geocode=3304557,
                )
                assert len(result) == 1

    def test_get_climate_uf(
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
                result = get_climate(
                    api_key=valid_api_key,
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    uf="RJ",
                )
                assert len(result) == 1


class TestGetClimateWeekly:
    def test_get_climate_weekly(
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
                result = get_climate_weekly(
                    api_key=valid_api_key,
                    start="202401",
                    end="202452",
                    geocode=3550308,
                )
                assert len(result) == 1

    def test_get_climate_weekly_macro_health(
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
                result = get_climate_weekly(
                    api_key=valid_api_key,
                    start="202401",
                    end="202452",
                    macro_health_code=1101,
                )
                assert len(result) == 1


class TestGetInfodengue:
    def test_get_infodengue_geocode(
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
                result = get_infodengue(
                    api_key=valid_api_key,
                    disease="dengue",
                    start_date=date(2023, 1, 1),
                    end_date=date(2023, 12, 31),
                    geocode=3304557,
                )
                assert len(result) == 1

    @pytest.mark.parametrize("disease", ["dengue", "zika", "chikungunya"])
    def test_get_infodengue_diseases(
        self,
        valid_api_key,
        mock_openapi_response,
        sample_infodengue_data,
        disease,
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
                result = get_infodengue(
                    api_key=valid_api_key,
                    disease=disease,
                    start_date="2023-01-01",
                    end_date="2023-12-31",
                    uf="RJ",
                )
                assert len(result) == 1


class TestGetEpiscanner:
    def test_get_episcanner(self, valid_api_key, mock_openapi_response):
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
                result = get_episcanner(
                    api_key=valid_api_key,
                    uf="SP",
                    disease="dengue",
                )
                assert len(result) == 0

    def test_get_episcanner_default_year(
        self, valid_api_key, mock_openapi_response
    ):
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
                from datetime import date

                result = get_episcanner(
                    api_key=valid_api_key,
                    uf="SP",
                    year=date.today().year,
                )
                assert len(result) == 0


class TestGetMosquito:
    def test_get_mosquito(self, valid_api_key, mock_openapi_response):
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
                result = get_mosquito(
                    api_key=valid_api_key,
                    date_start="2024-01-01",
                    date_end="2024-12-31",
                    state="MG",
                    municipality="Ponta Porã",
                )
                assert len(result) == 0

    def test_get_mosquito_minimal(self, valid_api_key, mock_openapi_response):
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
                result = get_mosquito(api_key=valid_api_key)
                assert len(result) == 0
