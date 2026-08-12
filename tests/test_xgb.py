"""Lightweight contract tests for the XGBoost forecasters."""

import numpy as np
import pandas as pd
import pytest

from mosqlient.forecast.xgb import ForecastXGB, ForecastXGBResidual


@pytest.fixture
def sample_forecast_df():
    dates = pd.date_range("2020-01-05", periods=20, freq="W-SUN")
    return pd.DataFrame(
        {
            "date": dates,
            "casos": np.arange(20, dtype=float),
            "temp": np.linspace(20, 25, 20),
        }
    )


def make_model(df, model_class=ForecastXGB):
    return model_class(
        df,
        columns=["casos", "temp"],
        date_col="date",
        target_col="casos",
        look_back=4,
        predict_n=4,
    )


def test_xgb_contract(sample_forecast_df):
    model = make_model(sample_forecast_df)

    assert model.target_col == "casos"
    assert model.look_back == 4
    assert model.predict_n == 4


def test_features_use_past_values_and_future_targets(sample_forecast_df):
    model = make_model(sample_forecast_df)
    features, targets = model._create_features(
        sample_forecast_df.set_index("date")
    )

    first_date = features.index[0]
    previous_date = first_date - pd.Timedelta(weeks=1)
    next_date = first_date + pd.Timedelta(weeks=1)

    assert features.loc[first_date, "casos_lag_1"] == np.log1p(
        sample_forecast_df.set_index("date").loc[previous_date, "casos"]
    )
    assert targets.loc[first_date, "target_h1"] == np.log1p(
        sample_forecast_df.set_index("date").loc[next_date, "casos"]
    )


def test_invalid_target_is_rejected(sample_forecast_df):
    with pytest.raises(
        ValueError, match="must be one of the available columns"
    ):
        ForecastXGB(
            sample_forecast_df,
            columns=["temp"],
            target_col="casos",
        )


def test_residual_uses_same_contract(sample_forecast_df):
    model = make_model(sample_forecast_df, ForecastXGBResidual)

    assert model.residual is True
    assert model.columns == ["casos", "temp"]
