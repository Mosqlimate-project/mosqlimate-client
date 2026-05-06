"""Tests for the client module."""

import pytest
import uuid
from unittest.mock import MagicMock, patch
from mosqlient.client import Mosqlient, validate_client, Client


class TestMosqlientInit:
    def test_init_valid(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response
            mock_get.return_value.status_code = 200
            client = Mosqlient(x_uid_key=valid_api_key)
            assert client.username == "testuser"
            assert client.timeout == 300
            assert client.per_page == 300
            assert "registry" in client.endpoints

    def test_init_invalid_uuid(self):
        with pytest.raises(ValueError, match="uid_key is not a valid key"):
            Mosqlient(x_uid_key="testuser:invalid-uuid")

    def test_init_custom_timeout(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response
            client = Mosqlient(x_uid_key=valid_api_key, timeout=600)
            assert client.timeout == 600

    def test_init_custom_api_url(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response
            client = Mosqlient(
                x_uid_key=valid_api_key,
                _api_url="https://custom.api.com/",
            )
            assert client.api_url == "https://custom.api.com/"

    def test_str(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response
            client = Mosqlient(x_uid_key=valid_api_key)
            assert str(client) == "testuser"

    def test_x_uid_key_property(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response
            client = Mosqlient(x_uid_key=valid_api_key)
            assert client.X_UID_KEY == valid_api_key


class TestMosqlientGet:
    def test_get_simple(
        self, mock_openapi_response, valid_api_key, mock_response_200
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response_200.json.return_value = {"items": [{"id": 1}]}
            mock_get.side_effect = [
                MagicMock(json=lambda: mock_openapi_response),
                mock_response_200,
            ]

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams(page=1)
            result = client.get(params)
            assert result == [{"id": 1}]

    def test_get_422_error(
        self, mock_openapi_response, valid_api_key, mock_response_422
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(json=lambda: mock_openapi_response),
                mock_response_422,
            ]

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams(page=1)
            with pytest.raises(ValueError, match="Validation error"):
                client.get(params)

    def test_get_http_error(
        self, mock_openapi_response, valid_api_key, mock_response_500
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(json=lambda: mock_openapi_response),
                mock_response_500,
            ]

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams()
            with pytest.raises(Exception):
                client.get(params)

    def test_get_paginated(
        self, mock_openapi_response, valid_api_key, mock_response_200
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response_200.json.return_value = {
                "items": [{"id": 1}],
                "pagination": {"total_pages": 1},
            }
            mock_get.side_effect = [
                MagicMock(json=lambda: mock_openapi_response),
                mock_response_200,
            ]

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams(page=1)
            result = client.get(params)
            assert result == [{"id": 1}]

    def test_get_pagination_warning(
        self, mock_openapi_response, valid_api_key, mock_response_200
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response_200.json.return_value = {
                "items": [{"id": 1}],
                "message": "Too many pages",
            }
            mock_get.side_effect = [
                MagicMock(json=lambda: mock_openapi_response),
                mock_response_200,
            ]

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams(page=1)
            result = client.get(params)
            assert result == [{"id": 1}]

    def test_get_per_page_limit(
        self, mock_openapi_response, valid_api_key, mock_response_200
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response_200.json.return_value = {
                "items": [{"id": 1}],
            }
            mock_get.side_effect = [
                MagicMock(json=lambda: mock_openapi_response),
                mock_response_200,
            ]

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key, max_items_per_page=100)
            params = ModelGETParams(page=1, per_page=200)
            result = client.get(params)
            assert result == [{"id": 1}]


class TestMosqlientPost:
    def test_post_success(
        self, mock_openapi_response, valid_api_key, mock_response_200
    ):
        with patch("mosqlient.client.requests.post") as mock_post:
            mock_post.return_value = mock_response_200

            from mosqlient.registry.schema import PredictionPOSTParams
            from datetime import date

            client = Mosqlient(x_uid_key=valid_api_key)
            params = PredictionPOSTParams(
                repository="owner/repo",
                description="Test",
                disease="A90",
                commit="a" * 40,
                case_definition="probable",
                published=True,
                adm_level=1,
                prediction=[],
            )
            result = client.post(params)
            assert result.status_code == 200

    def test_post_422_error(
        self, mock_openapi_response, valid_api_key, mock_response_422
    ):
        with patch("mosqlient.client.requests.post") as mock_post:
            mock_post.return_value = mock_response_422

            from mosqlient.registry.schema import PredictionPOSTParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = PredictionPOSTParams(
                repository="owner/repo",
                description="Test",
                disease="A90",
                commit="a" * 40,
                case_definition="probable",
                published=True,
                adm_level=1,
                prediction=[],
            )
            with pytest.raises(ValueError):
                client.post(params)


class TestMosqlientPatch:
    def test_patch_success(
        self, mock_openapi_response, valid_api_key, mock_response_200
    ):
        with patch("mosqlient.client.requests.patch") as mock_patch:
            mock_patch.return_value = mock_response_200

            from mosqlient.registry.schema import PredictionPublishPATCHParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = PredictionPublishPATCHParams(id=1, published=True)
            result = client.patch(params)
            assert result.status_code == 200


class TestMosqlientPut:
    def test_put_not_implemented(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams()
            with pytest.raises(NotImplementedError):
                client.put(params)


class TestMosqlientDelete:
    def test_delete_success(
        self, mock_openapi_response, valid_api_key, mock_response_200
    ):
        with patch("mosqlient.client.requests.delete") as mock_delete:
            mock_delete.return_value = mock_response_200

            from mosqlient.registry.schema import PredictionDELETEParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = PredictionDELETEParams(id=1)
            result = client.delete(params)
            assert result.status_code == 200


class TestMosqlientValidateRequest:
    def test_unknown_app(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams()
            params.app = "invalid_app"
            with pytest.raises(Exception, match="Unknown app"):
                client.get(params)

    def test_unknown_endpoint(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response

            from mosqlient.registry.schema import ModelGETParams

            client = Mosqlient(x_uid_key=valid_api_key)
            params = ModelGETParams()
            params.endpoint = "invalid/endpoint"
            with pytest.raises(Exception, match="Unknown endpoint"):
                client.get(params)


class TestValidateClient:
    def test_validate_client(self, mock_openapi_response, valid_api_key):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_openapi_response
            client = Mosqlient(x_uid_key=valid_api_key)
            result = validate_client(client)
            assert result is client


class TestClientType:
    def test_client_type_exists(self):
        assert Client is not None
