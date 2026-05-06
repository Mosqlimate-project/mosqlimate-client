"""Tests for the registry models module."""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock


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


from mosqlient.registry.models import Model, Prediction


class TestModel:
    def test_init(self, sample_model_data):
        m = Model(**sample_model_data)
        assert m.id == 1
        assert m.repository == "owner/repo"
        assert m.description == "Test model"
        assert m.category == "quantitative"
        assert m.time_resolution == "week"
        assert m.predictions_count == 5
        assert m.active is True

    def test_repr(self, sample_model_data):
        m = Model(**sample_model_data)
        assert repr(m) == "owner/repo"

    def test_properties(self, sample_model_data):
        m = Model(**sample_model_data)
        assert m.id == 1
        assert m.repository == "owner/repo"
        assert m.description == "Test model"
        assert m.category == "quantitative"
        assert m.time_resolution == "week"
        assert m.imdc_year is None
        assert m.predictions_count == 5
        assert m.active is True
        assert m.created_at == date(2023, 1, 1)
        assert m.last_update == date(2023, 6, 1)

    def test_get_method(
        self, valid_api_key, sample_model_data, mock_openapi_response
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {
                "items": [sample_model_data],
            }
            mock_response_2.status_code = 200
            mock_get.side_effect = [mock_response, mock_response_2]

            session_mock = make_aiohttp_patch([sample_model_data])
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                models = Model.get(api_key=valid_api_key)
                assert len(models) == 1
                assert isinstance(models[0], Model)

    def test_predictions_method(
        self, sample_model_data, valid_api_key, mock_openapi_response
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {
                "items": [],
            }
            mock_get.side_effect = [mock_response, mock_response_2]

            session_mock = make_aiohttp_patch([])
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                m = Model(**sample_model_data)
                predictions = m.predictions(api_key=valid_api_key)
                assert predictions == []


class TestPrediction:
    def test_init_with_dict_model(self, sample_prediction_dict):
        p = Prediction(**sample_prediction_dict)
        assert p.id == 1
        assert isinstance(p.model, Model)
        assert p.disease == "A90"
        assert p.published is True

    def test_repr(self, sample_prediction_dict):
        p = Prediction(**sample_prediction_dict)
        assert repr(p) == "Prediction <1>"

    def test_properties(self, sample_prediction_dict):
        p = Prediction(**sample_prediction_dict)
        assert p.id == 1
        assert p.disease == "A90"
        assert p.description == "Test prediction"
        assert p.commit == "a" * 40
        assert p.case_definition == "probable"
        assert p.published is True
        assert p.adm_0 == "BRA"
        assert p.adm_1 == 33
        assert p.adm_2 is None
        assert p.scores == {}

    def test_update_published_no_id(self):
        model = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )

        with pytest.raises(
            ValueError, match="Cannot update a prediction that has no ID"
        ):
            p = Prediction(
                id=None,
                model=model,
                disease="A90",
                commit="a" * 40,
                description="Test prediction",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
                adm_1=33,
            )
            p.update_published(False)

    def test_delete_no_id(self):
        model = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )

        with pytest.raises(
            ValueError, match="Cannot delete a prediction that has no ID"
        ):
            p = Prediction(
                id=None,
                model=model,
                disease="A90",
                commit="a" * 40,
                description="Test prediction",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
                adm_1=33,
            )
            p.delete(api_key="test:123")

    def test_delete_by_id(
        self, valid_api_key, mock_openapi_response, mock_response_200
    ):
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(json=lambda: mock_openapi_response),
            ]

            with patch("mosqlient.client.requests.delete") as mock_delete:
                mock_delete.return_value = mock_response_200

                result = Prediction.delete_by_id(api_key=valid_api_key, id=1)
                assert result.status_code == 200

    def test_to_dataframe(self, sample_prediction_dict):
        from mosqlient.registry.schema import PredictionDataRow

        p = Prediction(
            **sample_prediction_dict,
            data=[
                {
                    "date": date(2023, 1, 1),
                    "lower_90": 10.0,
                    "pred": 50.0,
                    "upper_90": 90.0,
                }
            ],
        )
        df = p.to_dataframe()
        assert len(df) == 1

    def test_data_property_with_cached_data(self, sample_prediction_dict):
        p = Prediction(
            **sample_prediction_dict,
            data=[
                {
                    "date": date(2023, 1, 1),
                    "lower_90": 10.0,
                    "pred": 50.0,
                    "upper_90": 90.0,
                }
            ],
        )
        data = p.data
        assert len(data) == 1

    def test_data_property_empty(self):
        model = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )

        p = Prediction(
            id=1,
            model=model,
            disease="A90",
            commit="a" * 40,
            description="Test prediction",
            case_definition="probable",
            published=True,
            created_at=date(2023, 1, 1),
            adm_level=1,
            adm_0="BRA",
            adm_1=33,
        )
        data = p.data
        assert data == []

    def test_start_end_properties(self):
        model = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )

        p = Prediction(
            id=1,
            model=model,
            disease="A90",
            commit="a" * 40,
            description="Test prediction",
            case_definition="probable",
            published=True,
            created_at=date(2023, 1, 1),
            adm_level=1,
            adm_0="BRA",
            adm_1=33,
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
        )
        assert p.start == date(2023, 1, 1)
        assert p.end == date(2023, 12, 31)

    def test_validate_prediction_method(
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

                Prediction.validate_prediction(
                    api_key=valid_api_key,
                    repository="owner/repo",
                    disease="A90",
                    description="Test",
                    commit="a" * 40,
                    prediction=prediction_df,
                    adm_level=1,
                    adm_1=33,
                )

    def test_validate_prediction_model_not_found(
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
                    ValueError, match="Model 'owner/repo' not found"
                ):
                    Prediction.validate_prediction(
                        api_key=valid_api_key,
                        repository="owner/repo",
                        disease="A90",
                        description="Test",
                        commit="a" * 40,
                        prediction=[{"date": "2023-01-01", "pred": 50.0}],
                        adm_level=1,
                    )

    def test_validate_prediction_dict_data(
        self, valid_api_key, sample_model_data, mock_openapi_response
    ):
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
                prediction_data = [
                    {
                        "date": "2023-01-01",
                        "lower_95": 10.0,
                        "lower_90": 15.0,
                        "lower_80": 20.0,
                        "lower_50": 30.0,
                        "pred": 50.0,
                        "upper_50": 70.0,
                        "upper_80": 80.0,
                        "upper_90": 85.0,
                        "upper_95": 90.0,
                    }
                ]

                Prediction.validate_prediction(
                    api_key=valid_api_key,
                    repository="owner/repo",
                    disease="A90",
                    description="Test",
                    commit="a" * 40,
                    prediction=prediction_data,
                    adm_level=1,
                    adm_1=33,
                )

    def test_init_with_string_data(self):
        import json

        model = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )

        data = [
            {
                "date": "2023-01-01",
                "lower_90": 10.0,
                "pred": 50.0,
                "upper_90": 90.0,
            }
        ]
        p = Prediction(
            id=1,
            model=model,
            disease="A90",
            commit="a" * 40,
            description="Test prediction",
            case_definition="probable",
            published=True,
            created_at=date(2023, 1, 1),
            adm_level=1,
            adm_0="BRA",
            adm_1=33,
            data=json.dumps(data),
        )
        assert len(p.data) == 1

    def test_init_with_invalid_string_data(self):
        model = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )

        with pytest.raises(
            ValueError, match="str `data` must be JSON serializable"
        ):
            Prediction(
                id=1,
                model=model,
                disease="A90",
                commit="a" * 40,
                description="Test prediction",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
                adm_1=33,
                data="not valid json{{{",
            )

    def test_init_with_dataframe_data(self):
        import pandas as pd

        model = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )

        df = pd.DataFrame(
            [
                {
                    "date": "2023-01-01",
                    "lower_90": 10.0,
                    "pred": 50.0,
                    "upper_90": 90.0,
                }
            ]
        )
        p = Prediction(
            id=1,
            model=model,
            disease="A90",
            commit="a" * 40,
            description="Test prediction",
            case_definition="probable",
            published=True,
            created_at=date(2023, 1, 1),
            adm_level=1,
            adm_0="BRA",
            adm_1=33,
            data=df,
        )
        assert len(p.data) == 1
