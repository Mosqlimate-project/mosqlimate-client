"""Configuration for the pytest test suite."""

import pytest
import datetime
import uuid
from unittest.mock import MagicMock, patch
from datetime import date

VALID_UUID = str(uuid.uuid4())
API_KEY = f"testuser:{VALID_UUID}"


@pytest.fixture
def valid_api_key():
    return API_KEY


@pytest.fixture
def valid_uid_key():
    return VALID_UUID


@pytest.fixture
def mock_response_200():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {}
    response.text = "{}"
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_response_422():
    response = MagicMock()
    response.status_code = 422
    response.text = '{"detail": "Validation error"}'
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_response_500():
    response = MagicMock()
    response.status_code = 500
    response.text = "Internal Server Error"
    response.raise_for_status.side_effect = Exception("500 Server Error")
    return response


@pytest.fixture
def sample_model_data():
    return {
        "id": 1,
        "repository": "owner/repo",
        "description": "Test model",
        "category": "quantitative",
        "time_resolution": "week",
        "imdc_year": None,
        "predictions_count": 5,
        "active": True,
        "created_at": date(2023, 1, 1),
        "last_update": date(2023, 6, 1),
    }


@pytest.fixture
def sample_model_schema():
    from mosqlient.registry.schema import Model

    return Model(
        id=1,
        repository="owner/repo",
        description="Test model",
        category="quantitative",
        time_resolution="week",
        imdc_year=None,
        predictions_count=5,
        active=True,
        created_at=date(2023, 1, 1),
        last_update=date(2023, 6, 1),
    )


@pytest.fixture
def sample_prediction_data_row():
    return {
        "date": date(2023, 1, 1),
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


@pytest.fixture
def sample_prediction_row_dict():
    return {
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


@pytest.fixture
def sample_prediction_data(sample_model_schema, sample_prediction_data_row):
    from mosqlient.registry.schema import PredictionDataRow

    return [PredictionDataRow(**sample_prediction_data_row)]


@pytest.fixture
def sample_prediction_schema(sample_model_schema, sample_prediction_data):
    from mosqlient.registry.schema import Prediction

    return Prediction(
        id=1,
        model=sample_model_schema,
        disease="A90",
        commit="a" * 40,
        description="Test prediction",
        case_definition="probable",
        published=True,
        created_at=date(2023, 1, 1),
        adm_level=1,
        adm_0="BRA",
        adm_1=33,
        data=sample_prediction_data,
    )


@pytest.fixture
def sample_prediction_dict(sample_model_data):
    return {
        "id": 1,
        "model": sample_model_data,
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
def mock_openapi_response():
    return {
        "paths": {
            "/v1/registry/models": {"get": {}, "post": {}},
            "/v1/registry/predictions": {"get": {}, "post": {}},
            "/v1/registry/predictions/{predict_id}": {"delete": {}},
            "/v1/registry/prediction/{prediction_id}/publish": {"patch": {}},
            "/v1/registry/predictions/{predict_id}/data": {"get": {}},
            "/v1/datastore/infodengue": {"get": {}},
            "/v1/datastore/climate": {"get": {}},
            "/v1/datastore/climate/weekly": {"get": {}},
            "/v1/datastore/episcanner": {"get": {}},
            "/v1/datastore/mosquito": {"get": {}},
        }
    }


@pytest.fixture
def sample_climate_data():
    return [
        {
            "date": "2023-01-01",
            "geocode": 3304557,
            "temperature_min": 20.0,
            "temperature_max": 30.0,
        }
    ]


@pytest.fixture
def sample_infodengue_data():
    return [
        {
            "date": "2023-01-01",
            "geocode": 3304557,
            "casos": 100,
            "p_inc": 0.8,
        }
    ]


@pytest.fixture
def sample_df_true():
    import pandas as pd

    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="W-SUN"),
            "casos": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        }
    )


@pytest.fixture
def sample_df_pred():
    import pandas as pd

    dates = pd.date_range("2023-01-01", periods=10, freq="W-SUN")
    return pd.DataFrame(
        {
            "date": dates,
            "lower_90": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
            "pred": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            "upper_90": [12, 18, 24, 30, 36, 42, 48, 54, 60, 66],
        }
    )


@pytest.fixture
def sample_ensemble_df():
    import pandas as pd

    dates = pd.date_range("2023-01-01", periods=5, freq="W-SUN")
    return pd.DataFrame(
        {
            "date": list(dates) * 3,
            "pred": [10, 15, 20, 25, 30] * 3,
            "lower_90": [8, 12, 16, 20, 24] * 3,
            "upper_90": [12, 18, 24, 30, 36] * 3,
            "model_id": ["model_a"] * 5 + ["model_b"] * 5 + ["model_c"] * 5,
        }
    )


@pytest.fixture
def sample_obs_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="W-SUN"),
            "casos": [10, 15, 20, 25, 30],
        }
    )


@pytest.fixture
def sample_arima_df():
    import pandas as pd

    return pd.DataFrame(
        {"y": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]},
        index=pd.date_range("2023-01-01", periods=15, freq="W-SUN"),
    )
