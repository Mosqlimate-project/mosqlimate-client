"""Tests for the registry convenience API functions."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mosqlient.registry._model_get_impl import (
    get_all_models,
    get_models,
    get_model_by_repository,
    get_models_by_disease,
    get_models_by_category,
    get_models_by_adm_level,
    get_models_by_time_resolution,
    get_models_by_imdc_year,
)
from mosqlient.registry._prediction_get_impl import (
    get_predictions,
    get_prediction_by_id,
    get_predictions_by_model_id,
    get_predictions_by_model_name,
    get_predictions_by_model_owner,
    get_predictions_by_model_organization,
    get_predictions_by_adm_level,
    get_predictions_by_time_resolution,
    get_predictions_by_disease,
    get_predictions_between,
)
from mosqlient.registry._prediction_post_impl import upload_prediction
from mosqlient.registry._prediction_delete_impl import delete_prediction
from mosqlient.registry._prediction_patch_impl import update_prediction_status
from mosqlient.registry._prediction_validate_impl import validate_prediction


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


SAMPLE_PREDICTION_DATA = {
    "id": 1,
    "model": {
        "id": 1,
        "repository": "owner/repo",
        "description": "Test model",
        "category": "quantitative",
        "time_resolution": "week",
        "imdc_year": None,
        "predictions_count": 5,
        "active": True,
        "created_at": "2023-01-01",
        "last_update": "2023-06-01",
    },
    "disease": "A90",
    "commit": "a" * 40,
    "description": "Test prediction",
    "case_definition": "probable",
    "published": True,
    "created_at": "2023-01-01",
    "adm_level": 1,
    "adm_0": "BRA",
    "adm_1": 33,
}


@pytest.fixture
def mock_model_list(valid_api_key, sample_model_data, mock_openapi_response):
    with patch("mosqlient.client.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "paths": mock_openapi_response["paths"]
        }
        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = {"items": [sample_model_data]}
        mock_response_2.status_code = 200
        mock_get.side_effect = [mock_response, mock_response_2]

        session_mock = make_aiohttp_patch([sample_model_data])
        with patch(
            "mosqlient.client.ClientSession", return_value=session_mock
        ):
            yield


@pytest.fixture
def mock_prediction_list(valid_api_key, mock_openapi_response):
    with patch("mosqlient.client.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "paths": mock_openapi_response["paths"]
        }
        mock_get.return_value = mock_response

        session_mock = make_aiohttp_patch([SAMPLE_PREDICTION_DATA])
        with patch(
            "mosqlient.client.ClientSession", return_value=session_mock
        ):
            yield


@pytest.fixture
def mock_empty_list(valid_api_key, mock_openapi_response):
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
            yield


class TestModelGetFunctions:
    def test_get_all_models(self, valid_api_key, mock_model_list):
        models = get_all_models(valid_api_key)
        assert isinstance(models, list)

    def test_get_models(self, valid_api_key, mock_model_list):
        models = get_models(valid_api_key, id=1, disease="A90")
        assert isinstance(models, list)

    def test_get_model_by_repository_owner(
        self, valid_api_key, mock_model_list
    ):
        model = get_model_by_repository(
            valid_api_key, name="repo", owner="owner"
        )
        assert model is not None

    def test_get_model_by_repository_org(self, valid_api_key, mock_model_list):
        model = get_model_by_repository(
            valid_api_key, name="repo", organization="org"
        )
        assert model is not None

    def test_get_model_by_repository_not_found(
        self, valid_api_key, mock_empty_list
    ):
        model = get_model_by_repository(
            valid_api_key, name="repo", owner="owner"
        )
        assert model is None

    def test_get_models_by_disease(self, valid_api_key, mock_model_list):
        models = get_models_by_disease(valid_api_key, "A90")
        assert isinstance(models, list)

    def test_get_models_by_category(self, valid_api_key, mock_model_list):
        models = get_models_by_category(valid_api_key, "quantitative")
        assert isinstance(models, list)

    def test_get_models_by_adm_level(self, valid_api_key, mock_model_list):
        models = get_models_by_adm_level(valid_api_key, 1)
        assert isinstance(models, list)

    def test_get_models_by_time_resolution(
        self, valid_api_key, mock_model_list
    ):
        models = get_models_by_time_resolution(valid_api_key, "week")
        assert isinstance(models, list)

    def test_get_models_by_imdc_year(self, valid_api_key, mock_model_list):
        models = get_models_by_imdc_year(valid_api_key, 2024)
        assert isinstance(models, list)


class TestPredictionGetFunctions:
    def test_get_predictions(self, valid_api_key, mock_prediction_list):
        predictions = get_predictions(valid_api_key, id=1)
        assert isinstance(predictions, list)

    def test_get_prediction_by_id_found(
        self, valid_api_key, mock_prediction_list
    ):
        pred = get_prediction_by_id(valid_api_key, id=1)
        assert pred is not None

    def test_get_prediction_by_id_not_found(
        self, valid_api_key, mock_empty_list
    ):
        pred = get_prediction_by_id(valid_api_key, id=999)
        assert pred is None

    def test_get_predictions_by_model_id(
        self, valid_api_key, mock_prediction_list
    ):
        preds = get_predictions_by_model_id(valid_api_key, model_id=1)
        assert isinstance(preds, list)

    def test_get_predictions_by_model_name(
        self, valid_api_key, mock_prediction_list
    ):
        preds = get_predictions_by_model_name(valid_api_key, model_name="repo")
        assert isinstance(preds, list)

    def test_get_predictions_by_model_owner(
        self, valid_api_key, mock_prediction_list
    ):
        preds = get_predictions_by_model_owner(
            valid_api_key, model_owner="owner"
        )
        assert isinstance(preds, list)

    def test_get_predictions_by_model_organization(
        self, valid_api_key, mock_prediction_list
    ):
        preds = get_predictions_by_model_organization(
            valid_api_key, model_organization="org"
        )
        assert isinstance(preds, list)

    def test_get_predictions_by_adm_level(
        self, valid_api_key, mock_prediction_list
    ):
        preds = get_predictions_by_adm_level(valid_api_key, adm_level=1)
        assert isinstance(preds, list)

    def test_get_predictions_by_time_resolution(
        self, valid_api_key, mock_prediction_list
    ):
        preds = get_predictions_by_time_resolution(
            valid_api_key, time_resolution="week"
        )
        assert isinstance(preds, list)

    def test_get_predictions_by_disease(
        self, valid_api_key, mock_prediction_list
    ):
        preds = get_predictions_by_disease(valid_api_key, disease="A90")
        assert isinstance(preds, list)

    def test_get_predictions_between(
        self, valid_api_key, mock_prediction_list
    ):
        from datetime import date

        preds = get_predictions_between(
            valid_api_key, start=date(2023, 1, 1), end=date(2023, 12, 31)
        )
        assert isinstance(preds, list)


class TestPredictionPostFunctions:
    def test_upload_prediction(
        self, valid_api_key, sample_model_data, mock_openapi_response
    ):
        import pandas as pd

        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_get.side_effect = [mock_response]

            with patch("mosqlient.client.requests.post") as mock_post:
                mock_post.return_value = MagicMock(
                    status_code=201, text='{"id": 1}'
                )

                pred_df = pd.DataFrame(
                    {
                        "date": ["2023-01-01"],
                        "lower_95": [10.0],
                        "lower_90": [15.0],
                        "lower_80": [20.0],
                        "lower_50": [30.0],
                        "pred": [50.0],
                        "upper_50": [70.0],
                        "upper_80": [80.0],
                        "upper_90": [85.0],
                        "upper_95": [90.0],
                    }
                )

                mock_get_2_resp = MagicMock()
                mock_get_2_resp.json.return_value = {
                    "paths": mock_openapi_response["paths"]
                }
                mock_get_2_resp_2 = MagicMock()
                mock_get_2_resp_2.json.return_value = {
                    "items": [
                        {
                            "id": 1,
                            "model": sample_model_data,
                            "disease": "A90",
                            "commit": "a" * 40,
                            "description": "Test",
                            "case_definition": "probable",
                            "published": True,
                            "created_at": "2023-01-01",
                            "adm_level": 1,
                            "adm_0": "BRA",
                            "adm_1": 33,
                        }
                    ]
                }
                mock_get.side_effect = [
                    mock_response,
                    mock_get_2_resp,
                    mock_get_2_resp_2,
                ]

                session_mock = make_aiohttp_patch(
                    [
                        {
                            "id": 1,
                            "model": sample_model_data,
                            "disease": "A90",
                            "commit": "a" * 40,
                            "description": "Test",
                            "case_definition": "probable",
                            "published": True,
                            "created_at": "2023-01-01",
                            "adm_level": 1,
                            "adm_0": "BRA",
                            "adm_1": 33,
                        }
                    ]
                )
                with patch(
                    "mosqlient.client.ClientSession", return_value=session_mock
                ):
                    result = upload_prediction(
                        api_key=valid_api_key,
                        repository="owner/repo",
                        disease="A90",
                        description="Test",
                        commit="a" * 40,
                        prediction=pred_df,
                        adm_level=1,
                    )
                    assert result is not None


class TestPredictionDeleteFunctions:
    def test_delete_prediction(
        self, valid_api_key, mock_openapi_response, mock_response_200
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.return_value = MagicMock(
                json=lambda: {"paths": mock_openapi_response["paths"]}
            )

            with patch("mosqlient.client.requests.delete") as mock_delete:
                mock_delete.return_value = mock_response_200

                result = delete_prediction(valid_api_key, prediction_id=1)
                assert result.status_code == 200


class TestPredictionPatchFunctions:
    def test_update_prediction_status_found(
        self, valid_api_key, sample_model_data, mock_openapi_response
    ):
        session_mock = make_aiohttp_patch(
            [
                {
                    "id": 1,
                    "model": sample_model_data,
                    "disease": "A90",
                    "commit": "a" * 40,
                    "description": "Test",
                    "case_definition": "probable",
                    "published": True,
                    "created_at": "2023-01-01",
                    "adm_level": 1,
                    "adm_0": "BRA",
                    "adm_1": 33,
                }
            ]
        )
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_get.return_value = mock_response

            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                with patch("mosqlient.client.requests.patch") as mock_patch:
                    mock_patch.return_value = MagicMock(status_code=201)

                    result = update_prediction_status(
                        valid_api_key, prediction_id=1, published=False
                    )
                    assert result is not None

    def test_update_prediction_status_not_found(
        self, valid_api_key, mock_openapi_response
    ):
        session_mock = make_aiohttp_patch([])
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_get.return_value = mock_response

            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                with pytest.raises(
                    ValueError, match="Prediction with ID 999 not found"
                ):
                    update_prediction_status(
                        valid_api_key, prediction_id=999, published=False
                    )


class TestPredictionValidateFunctions:
    def test_validate_prediction(
        self, valid_api_key, sample_model_data, mock_openapi_response
    ):
        import pandas as pd

        session_mock = make_aiohttp_patch([sample_model_data])
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_get.return_value = mock_response

            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                prediction_df = pd.DataFrame(
                    {
                        "date": ["2023-01-01"],
                        "lower_95": [10.0],
                        "lower_90": [15.0],
                        "lower_80": [20.0],
                        "lower_50": [30.0],
                        "pred": [50.0],
                        "upper_50": [70.0],
                        "upper_80": [80.0],
                        "upper_90": [85.0],
                        "upper_95": [90.0],
                    }
                )

                validate_prediction(
                    api_key=valid_api_key,
                    repository="owner/repo",
                    disease="A90",
                    description="Test",
                    commit="a" * 40,
                    prediction=prediction_df,
                    adm_level=1,
                    adm_1=33,
                )
