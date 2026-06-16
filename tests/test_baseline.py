"""Tests for the baseline module."""

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from mosqlient.forecast.baseline import (
    get_next_n_weeks,
    get_prediction_dataframe,
    plot_predictions,
    plot_forecast,
    InvalidDataFrameError,
    Arima,
)


class TestGetNextNWeeks:
    def test_basic(self):
        result = get_next_n_weeks("2023-01-01", 4)
        assert len(result) == 4

    def test_dates_are_date_objects(self):
        from datetime import date

        result = get_next_n_weeks("2023-01-01", 2)
        assert all(isinstance(d, date) for d in result)

    def test_dates_increasing(self):
        result = get_next_n_weeks("2023-01-01", 3)
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]


class TestGetPredictionDataFrame:
    def test_with_horizon(self):
        import pmdarima as pm
        from pmdarima import preprocessing as ppc

        dates = pd.date_range("2023-01-01", periods=20, freq="W-SUN")
        df = pd.DataFrame({"y": np.random.poisson(50, 20)}, index=dates)

        boxcox = ppc.BoxCoxEndogTransformer().fit(df.y)
        df_transformed = df.copy()
        df_transformed = df_transformed.astype({
                                            "y": float,
                                        })
        df_transformed.loc[:, "y"] = boxcox.transform(df.y)[0]

        model = pm.auto_arima(
            df_transformed.y,
            seasonal=False,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        model.fit(df_transformed.y)

        pred_dates = get_next_n_weeks("2023-01-01", 4)
        result = get_prediction_dataframe(
            model, pred_dates, boxcox, horizon=4, alphas=[0.05]
        )
        assert "pred" in result.columns
        assert "date" in result.columns


class TestPlotPredictions:
    def test_returns_none(self):
        df_preds = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="W-SUN"),
                "data": [10, 15, 20, 25, 30],
                "pred": [12, 16, 18, 24, 28],
                "lower_95": [8, 12, 14, 20, 24],
                "upper_95": [16, 20, 22, 28, 32],
                "lower_90": [9, 13, 15, 21, 25],
                "upper_90": [15, 19, 21, 27, 31],
                "lower_80": [10, 14, 16, 22, 26],
                "upper_80": [14, 18, 20, 26, 30],
                "lower_50": [11, 15, 17, 23, 27],
                "upper_50": [13, 17, 19, 25, 29],
            }
        )
        result = plot_predictions(df_preds, title="Test")
        assert result is None


class TestPlotForecast:
    def test_returns_none(self):
        df_for = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="W-SUN"),
                "pred": [12, 16, 18, 24, 28],
                "lower_95": [8, 12, 14, 20, 24],
                "upper_95": [16, 20, 22, 28, 32],
                "lower_90": [9, 13, 15, 21, 25],
                "upper_90": [15, 19, 21, 27, 31],
                "lower_80": [10, 14, 16, 22, 26],
                "upper_80": [14, 18, 20, 26, 30],
                "lower_50": [11, 15, 17, 23, 27],
                "upper_50": [13, 17, 19, 25, 29],
            }
        )
        df_train = pd.DataFrame(
            {"data": [10, 15, 20, 25, 30]},
            index=pd.date_range("2022-12-01", periods=5, freq="W-SUN"),
        )
        result = plot_forecast(df_for, df_train, last_obs=3)
        assert result is None


class TestInvalidDataFrameError:
    def test_exception(self):
        with pytest.raises(InvalidDataFrameError):
            raise InvalidDataFrameError("test error")


class TestArima:
    def test_init_valid(self, sample_arima_df):
        model = Arima(
            sample_arima_df,
            seasonal=False,
            trace=False,
            suppress_warnings=True,
        )
        assert model.df.shape[1] == 1

    def test_init_invalid_index(self):
        df = pd.DataFrame({"y": [10, 20, 30]})
        with pytest.raises(
            InvalidDataFrameError, match="not of datetime type"
        ):
            Arima(df)

    def test_init_invalid_columns(self):
        df = pd.DataFrame(
            {"y1": [10, 20, 30], "y2": [15, 25, 35]},
            index=pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
        )
        with pytest.raises(InvalidDataFrameError, match="one single column"):
            Arima(df)

    def test_init_invalid_column_name(self):
        df = pd.DataFrame(
            {"cases": [10, 20, 30]},
            index=pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
        )
        with pytest.raises(InvalidDataFrameError, match="must be named `y`"):
            Arima(df)

    def test_train(self, sample_arima_df):
        model = Arima(
            sample_arima_df,
            seasonal=False,
            trace=False,
            suppress_warnings=True,
            stepwise=True,
        )
        result = model.train("2023-01-01", "2023-02-26")
        assert result is not None
        assert hasattr(model, "model")

    def test_predict_in_sample(self, sample_arima_df):
        model = Arima(
            sample_arima_df,
            seasonal=False,
            trace=False,
            suppress_warnings=True,
            stepwise=True,
        )
        model.train("2023-01-01", "2023-02-26")
        result = model.predict_in_sample(plot=False)
        assert isinstance(result, pd.DataFrame)
        assert "pred" in result.columns

    def test_predict_out_of_sample(self, sample_arima_df):
        model = Arima(
            sample_arima_df,
            seasonal=False,
            trace=False,
            suppress_warnings=True,
            stepwise=True,
        )
        model.train("2023-01-01", "2023-02-26")
        result = model.predict_out_of_sample(
            horizon=2, end_date="2023-03-19", plot=False
        )
        assert isinstance(result, pd.DataFrame)

    def test_forecast(self, sample_arima_df):
        model = Arima(
            sample_arima_df,
            seasonal=False,
            trace=False,
            suppress_warnings=True,
            stepwise=True,
        )
        model.train("2023-01-01", "2023-02-26")
        result = model.forecast(horizon=4, plot=False, last_obs=3)
        assert isinstance(result, pd.DataFrame)
