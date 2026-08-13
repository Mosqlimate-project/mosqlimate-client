"""Lightweight contract tests for the XGBoost forecasters."""

import numpy as np
import pandas as pd
import pytest

from mosqlient.forecast.xgb import ForecastXGB


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


@pytest.fixture
def train_forecast_df():
    dates = pd.date_range("2020-01-05", periods=40, freq="W-SUN")
    return pd.DataFrame(
        {
            "date": dates,
            "casos": np.arange(1, 41, dtype=float),
            "temp": np.linspace(20, 25, 40),
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


def test_features_can_use_raw_target_values(sample_forecast_df):
    model = make_model(sample_forecast_df)
    features, targets = model._create_features(
        sample_forecast_df.set_index("date"), use_log=False
    )

    first_date = features.index[0]
    assert features.loc[first_date, "casos_lag_1"] == 3
    assert targets.loc[first_date, "target_h1"] == 5


def test_train_and_predictions_return_intervals(
    train_forecast_df, monkeypatch
):
    class FakeXGBRegressor:
        best_iteration = 1
        best_score = 0.1

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, **kwargs):
            return self

        def evals_result(self):
            return {
                "validation_0": {"quantile": [0.2]},
                "validation_1": {"quantile": [0.1]},
            }

        def predict(self, X):
            return np.tile(np.arange(9, dtype=float), (len(X), 1))

    import sys
    from types import ModuleType

    fake_xgboost = ModuleType("xgboost")
    fake_xgboost.XGBRegressor = FakeXGBRegressor
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)

    model = ForecastXGB(
        train_forecast_df,
        columns=["casos", "temp"],
        date_col="date",
        target_col="casos",
        look_back=4,
        predict_n=2,
        n_estimators=2,
        max_depth=2,
        learning_rate=0.1,
    )

    fitted, history = model.train(
        ini_train_date="2020-01-05",
        end_train_date="2020-07-26",
        end_date="2020-10-04",
        early_stopping_rounds=1,
        val_ratio=0.2,
    )

    assert fitted is model.model
    assert len(model.models) == 2
    assert history["train_loss"]
    assert len(model.X_train) < len(model.X_test) + len(model.X_train)
    assert model.X_train.index.max() < model.X_test.index.min()

    in_sample = model.predict_in_sample()
    out_of_sample = model.predict_out_of_sample()
    forecast = model.forecast("2020-10-04")

    for predictions in (in_sample, out_of_sample, forecast):
        assert {"date", "pred", "lower_95", "upper_95"}.issubset(
            predictions.columns
        )
        assert (predictions["lower_95"] <= predictions["pred"]).all()
        assert (predictions["pred"] <= predictions["upper_95"]).all()


def test_prediction_methods_require_training(sample_forecast_df):
    model = make_model(sample_forecast_df)

    for prediction in (
        model.predict_in_sample,
        model.predict_out_of_sample,
        lambda: model.forecast("2020-02-02"),
    ):
        with pytest.raises(RuntimeError, match="trained"):
            prediction()


def test_residual_baseline_requires_rolling_feature(sample_forecast_df):
    model = make_model(sample_forecast_df)

    with pytest.raises(
        ValueError, match="Residual baseline feature is missing"
    ):
        model._residual_baseline(pd.DataFrame(index=[0]))


def test_invalid_target_is_rejected(sample_forecast_df):
    with pytest.raises(
        ValueError, match="must be one of the available columns"
    ):
        ForecastXGB(
            sample_forecast_df,
            columns=["temp"],
            target_col="casos",
        )


def test_residual_is_default_and_can_be_disabled(sample_forecast_df):
    model = make_model(sample_forecast_df)

    assert model.residual is True
    assert model.columns == ["casos", "temp"]

    current = ForecastXGB(
        sample_forecast_df,
        columns=["casos", "temp"],
        residual=False,
    )
    assert current.residual is False
