"""Tests for the ForecastXGB module."""

import numpy as np
import pandas as pd
import pytest

from mosqlient.forecast.baseline import get_next_n_weeks
from mosqlient.forecast.xgb import ForecastXGB, ForecastXGBResidual


@pytest.fixture
def sample_forecast_df():
    """Generates a synthetic time-series dataset for testing ForecastXGB."""
    dates = pd.date_range("2020-01-05", periods=100, freq="W-SUN")
    np.random.seed(42)
    casos = (
        np.random.poisson(lam=50, size=100)
        + np.sin(np.linspace(0, 10, 100)) * 20
    )
    casos = np.clip(casos, a_min=0, a_max=None)
    temp = 25.0 + np.sin(np.linspace(0, 10, 100)) * 5

    df = pd.DataFrame(
        {
            "date": dates,
            "casos": casos,
            "temp": temp,
        }
    )
    return df


class TestGetNextNWeeks:
    def test_basic(self):
        result = get_next_n_weeks("2023-01-01", 4)
        assert len(result) == 4

    def test_dates_increasing(self):
        result = get_next_n_weeks("2023-01-01", 3)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]


class TestForecastXGB:
    def test_init_valid(self, sample_forecast_df):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
        )
        assert model.target_col == "casos"
        assert model.look_back == 4

    def test_features_use_only_previous_target_values(
        self, sample_forecast_df
    ):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
        )
        features, targets = model._create_features(
            sample_forecast_df.set_index("date")
        )

        date = features.index[0]
        previous = sample_forecast_df.set_index("date").loc[
            date - pd.Timedelta(weeks=1), "casos"
        ]
        assert features.loc[date, "casos_lag_1"] == np.log1p(previous)
        assert targets.loc[date, "target_h1"] == np.log1p(
            sample_forecast_df.set_index("date").loc[date, "casos"]
        )

        changed = sample_forecast_df.copy()
        changed.loc[changed["date"] > date, "casos"] = 999999
        changed.loc[changed["date"] > date, "temp"] = -999
        changed_features, _ = model._create_features(
            changed.set_index("date"), require_complete_targets=False
        )
        pd.testing.assert_frame_equal(
            features.loc[:date], changed_features.loc[:date]
        )

    def test_training_targets_do_not_cross_cutoff(self, sample_forecast_df):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
            n_estimators=10,
        )
        cutoff = pd.Timestamp("2021-06-01")
        model.train(
            ini_train_date="2020-01-05",
            end_train_date=cutoff.strftime("%Y-%m-%d"),
            end_date="2021-12-01",
        )

        assert model.Y_train is not None
        assert model.Y_train.index.max() + pd.Timedelta(weeks=3) <= cutoff

        assert model.X_test is not None
        assert model.X_test.index.max() == sample_forecast_df["date"].max()

    def test_init_invalid_target(self, sample_forecast_df):
        with pytest.raises(
            ValueError, match="must be one of the available columns"
        ):
            ForecastXGB(
                df_data=sample_forecast_df,
                columns=["temp"],
                target_col="casos",
            )

    def test_train(self, sample_forecast_df):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
            n_estimators=10,
        )

        ini_train = "2020-01-05"
        end_train = "2021-06-01"
        end_date = "2021-12-01"

        fitted_model, _history = model.train(
            ini_train_date=ini_train,
            end_train_date=end_train,
            end_date=end_date,
            use_log=True,
        )

        assert fitted_model is not None
        assert model.X_train is not None
        assert not model.X_train.empty
        assert model.X_test is not None

    def test_predict_in_sample(self, sample_forecast_df):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
            n_estimators=10,
        )
        model.train(
            ini_train_date="2020-01-05",
            end_train_date="2021-06-01",
            end_date="2021-12-01",
        )
        res = model.predict_in_sample()
        assert isinstance(res, pd.DataFrame)
        assert "pred" in res.columns
        assert "casos" in res.columns
        assert "date" in res.columns
        assert len(res) > 0

    def test_predict_out_of_sample(self, sample_forecast_df):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
            n_estimators=10,
        )
        model.train(
            ini_train_date="2020-01-05",
            end_train_date="2021-06-01",
            end_date="2021-12-01",
        )
        res = model.predict_out_of_sample()
        assert isinstance(res, pd.DataFrame)
        assert "pred" in res.columns
        assert "date" in res.columns

    def test_forecast_future(self, sample_forecast_df):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
            n_estimators=10,
        )
        model.train(
            ini_train_date="2020-01-05",
            end_train_date="2021-06-01",
            end_date="2021-12-01",
        )

        future_date = "2022-02-01"
        res = model.forecast(end_date=future_date)
        assert isinstance(res, pd.DataFrame)
        assert "pred" in res.columns
        assert "date" in res.columns
        assert len(res) > 0

    def test_predict_before_train_raises(self, sample_forecast_df):
        model = ForecastXGB(
            df_data=sample_forecast_df,
            columns=["casos"],
            target_col="casos",
        )
        with pytest.raises(RuntimeError, match="must be trained"):
            model.predict_in_sample()

    def test_residual_variant_uses_same_contract(self, sample_forecast_df):
        model = ForecastXGBResidual(
            df_data=sample_forecast_df,
            columns=["casos", "temp"],
            date_col="date",
            target_col="casos",
            look_back=4,
            predict_n=4,
            n_estimators=10,
        )
        model.train(
            ini_train_date="2020-01-05",
            end_train_date="2021-06-01",
            end_date="2021-12-01",
        )
        result = model.predict_out_of_sample()

        assert model.residual is True
        assert {"date", "pred", "casos"}.issubset(result.columns)
        assert result["pred"].ge(0).all()

        future = model.forecast(end_date="2022-02-01")
        assert {"date", "pred"}.issubset(future.columns)
        assert future["pred"].ge(0).all()
