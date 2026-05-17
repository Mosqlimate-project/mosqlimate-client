"""Tests for the prediction_optimize module."""

import pytest
import numpy as np
import pandas as pd
from mosqlient.prediction_optimize.pred_opt import (
    get_lognormal_pars,
    get_normal_pars,
    get_df_pars,
)


class TestGetLognormalPars:
    def test_median_loss(self):
        mu, sigma = get_lognormal_pars(
            med=50.0, lwr=30.0, upr=70.0, conf_level=0.9, fn_loss="median"
        )
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_lower_loss(self):
        mu, sigma = get_lognormal_pars(
            med=50.0, lwr=30.0, upr=70.0, conf_level=0.9, fn_loss="lower"
        )
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_med_zero(self):
        mu, sigma = get_lognormal_pars(
            med=0.0, lwr=0.0, upr=70.0, conf_level=0.9, fn_loss="median"
        )
        assert isinstance(mu, float)
        assert sigma > 0

    def test_lwr_zero(self):
        mu, sigma = get_lognormal_pars(
            med=50.0, lwr=0.0, upr=70.0, conf_level=0.9, fn_loss="lower"
        )
        assert isinstance(mu, float)
        assert sigma > 0

    def test_invalid_fn_loss(self):
        with pytest.raises(ValueError, match="Invalid value for fn_loss"):
            get_lognormal_pars(med=50.0, lwr=30.0, upr=70.0, fn_loss="invalid")

    def test_negative_values(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            get_lognormal_pars(med=-1.0, lwr=30.0, upr=70.0)


class TestGetNormalPars:
    def test_median_loss(self):
        mu, sigma = get_normal_pars(
            med=50.0, lwr=30.0, upr=70.0, conf_level=0.9, fn_loss="median"
        )
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_lower_loss(self):
        mu, sigma = get_normal_pars(
            med=50.0, lwr=30.0, upr=70.0, conf_level=0.9, fn_loss="lower"
        )
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_lwr_zero(self):
        mu, sigma = get_normal_pars(
            med=50.0, lwr=0.0, upr=70.0, conf_level=0.9, fn_loss="lower"
        )
        assert isinstance(mu, float)
        assert sigma > 0


class TestGetDfPars:
    def test_lognormal_dist(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "pred": [50.0],
                "lower_90": [30.0],
                "upper_90": [70.0],
                "model_id": ["test"],
            }
        )
        result = get_df_pars(df, conf_level=0.9, dist="log_normal")
        assert "mu" in result.columns
        assert "sigma" in result.columns
        assert len(result) == 1

    def test_normal_dist(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "pred": [50.0],
                "lower_90": [30.0],
                "upper_90": [70.0],
                "model_id": ["test"],
            }
        )
        result = get_df_pars(df, conf_level=0.9, dist="normal")
        assert "mu" in result.columns
        assert "sigma" in result.columns
        assert len(result) == 1

    def test_return_estimations_lognormal(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "pred": [50.0],
                "lower_90": [30.0],
                "upper_90": [70.0],
                "model_id": ["test"],
            }
        )
        result = get_df_pars(
            df, conf_level=0.9, dist="log_normal", return_estimations=True
        )
        assert "fit_med" in result.columns
        assert "fit_lwr" in result.columns
        assert "fit_upr" in result.columns

    def test_return_estimations_normal(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "pred": [50.0],
                "lower_90": [30.0],
                "upper_90": [70.0],
                "model_id": ["test"],
            }
        )
        result = get_df_pars(
            df, conf_level=0.9, dist="normal", return_estimations=True
        )
        assert "fit_med" in result.columns
        assert "fit_lwr" in result.columns
        assert "fit_upr" in result.columns

    def test_multiple_rows(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-08"],
                "pred": [50.0, 60.0],
                "lower_90": [30.0, 40.0],
                "upper_90": [70.0, 80.0],
                "model_id": ["test", "test"],
            }
        )
        result = get_df_pars(df, conf_level=0.9, dist="log_normal")
        assert len(result) == 2
