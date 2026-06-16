"""Tests for the visualization module."""

import pytest
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from mosqlient.forecast.viz import plot_preds


class TestPlotPreds:
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
        ax = plot_preds(data, df_preds, data_col="casprov", conf_level=0.9)
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
        ax = plot_preds(None, df_preds, conf_level=0.9)
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
        ax = plot_preds(None, df_preds, conf_level=0.9)
        assert ax is not None

    def test_custom_data_col(self):
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
        ax = plot_preds(data, df_preds, conf_level=0.9, data_col="cases")
        assert ax is not None

    def test_invalid_df_preds(self):
        df_preds = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "pred": [10, 15, 20],
            }
        )
        with pytest.raises(ValueError, match="Missing columns in df_preds"):
            plot_preds(None, df_preds, conf_level=0.9)
