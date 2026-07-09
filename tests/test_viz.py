"""Tests for the visualization module."""

import pytest
import pandas as pd
import matplotlib.pyplot as plt

from mosqlient.forecast.viz import (
    plot_forecasts,
    plot_model_comparison,
    plot_single_forecast,
)


@pytest.fixture(autouse=True)
def close_plots():
    """Garante que nenhum gráfico fique aberto na memória após cada teste."""
    yield
    plt.close("all")


class TestPlotForecasts:
    def test_with_data(self):
        data = pd.DataFrame(
            {
                "casprov": [10, 15, 20],
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
            },
        )
        df_preds = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "lower_90": [8, 12, 16],
                "pred": [10, 15, 20],
                "upper_90": [12, 18, 24],
                "model_id": ["model_a"] * 3,
            }
        )
        fig, ax = plot_forecasts(
            df_preds, data, target_col="casprov", conf_levels=[0.9]
        )
        assert fig is not None
        assert ax is not None

    def test_without_data(self):
        df_preds = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "lower_90": [8, 12, 16],
                "pred": [10, 15, 20],
                "upper_90": [12, 18, 24],
                "model_id": ["model_a"] * 3,
            }
        )
        fig, ax = plot_forecasts(df_preds, None, conf_levels=[0.9])
        assert fig is not None
        assert ax is not None

    def test_multiple_models(self):
        df_preds = pd.DataFrame(
            {
                "date": list(
                    pd.date_range("2023-01-01", periods=3, freq="W-SUN")
                )
                * 2,
                "lower_90": [8, 12, 16, 9, 13, 17],
                "pred": [10, 15, 20, 11, 16, 21],
                "upper_90": [12, 18, 24, 13, 19, 25],
                "model_id": ["model_a"] * 3 + ["model_b"] * 3,
            }
        )
        fig, ax = plot_forecasts(
            df_preds, None, model_col="model_id", conf_levels=[0.9]
        )
        assert fig is not None
        assert ax is not None

    def test_custom_target_col(self):
        data = pd.DataFrame(
            {
                "cases": [10, 15, 20],
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
            },
        )
        df_preds = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "lower_90": [8, 12, 16],
                "pred": [10, 15, 20],
                "upper_90": [12, 18, 24],
                "model_id": ["model_a"] * 3,
            }
        )
        fig, ax = plot_forecasts(
            df_preds, data, conf_levels=[0.9], target_col="cases"
        )
        assert fig is not None
        assert ax is not None

    def test_invalid_df_preds(self):
        with pytest.raises(ValueError, match="df_preds cannot be None"):
            plot_forecasts(None, None)


class TestPlotSingleForecast:
    def test_arima_fan_style(self):
        # O wrapper usa as datas do treino localizadas no INDEX
        df_train = pd.DataFrame(
            {"data": [10, 15, 20]},
            index=pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
        )
        df_for = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-22", periods=3, freq="W-SUN"),
                "pred": [22, 25, 28],
                "lower_50": [20, 23, 26],
                "upper_50": [24, 27, 30],
                "lower_90": [18, 21, 24],
                "upper_90": [26, 29, 32],
                "lower_95": [17, 20, 23],
                "upper_95": [27, 30, 33],
            }
        )
        fig, ax = plot_single_forecast(df_for, df_train, last_obs=2)
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Forecast ARIMA"


class TestPlotModelComparison:
    def test_comparison_style(self):
        df_preds = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "lower_50": [8, 12, 16],
                "pred": [10, 15, 20],
                "upper_50": [12, 18, 24],
                "lower_90": [7, 11, 15],
                "upper_90": [13, 19, 25],
                "lower_95": [6, 10, 14],
                "upper_95": [14, 20, 26],
            }
        )
        fig, ax = plot_model_comparison(
            df_preds,
            title="In sample predictions",
            xlabel="Date",
            ylabel="New cases",
        )
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "In sample predictions"
