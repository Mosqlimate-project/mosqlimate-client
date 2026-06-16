"""Tests for the ensemble module."""

import pytest
import numpy as np
import pandas as pd
from mosqlient.forecast.ensemble import (
    invlogit,
    alpha_01,
    pool_par_gauss,
    get_score,
    get_epiweek,
    get_ci_columns,
    EnsembleDistPool,
    dlnorm_mix,
    compute_ppf,
    crps_lognormal_mix,
    find_opt_weights_linear,
    get_quantiles_log,
    get_quantiles_linear,
)


class TestInvlogit:
    def test_zero(self):
        result = invlogit(0.0)
        assert np.isclose(result, 0.5)

    def test_positive(self):
        result = invlogit(1.0)
        assert 0.5 < result < 1.0

    def test_negative(self):
        result = invlogit(-1.0)
        assert 0.0 < result < 0.5


class TestAlpha01:
    def test_simplex_sum_to_one(self):
        alpha_inv = np.array([0.0, 0.0])
        result = alpha_01(alpha_inv)
        assert np.isclose(np.sum(result), 1.0)

    def test_positive_values(self):
        alpha_inv = np.array([1.0, 1.0])
        result = alpha_01(alpha_inv)
        assert all(x > 0 for x in result)


class TestPoolParGauss:
    def test_same_length_arrays(self):
        alpha = np.array([0.5, 0.5])
        m = np.array([1.0, 2.0])
        v = np.array([1.0, 1.0])
        mstar, sd = pool_par_gauss(alpha, m, v)
        assert isinstance(mstar, float)
        assert isinstance(sd, float)

    def test_different_lengths_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            pool_par_gauss(
                np.array([0.5]),
                np.array([1.0, 2.0]),
                np.array([1.0, 1.0]),
            )


class TestGetScore:
    def test_crps_lognormal(self):
        score = get_score(10.0, 2.0, 0.5, dist="log_normal", metric="crps")
        assert isinstance(score, float)

    def test_log_score_lognormal(self):
        score = get_score(
            10.0, 2.0, 0.5, dist="log_normal", metric="log_score"
        )
        assert isinstance(score, float)

    def test_invalid_dist_metric(self):
        with pytest.raises(ValueError, match="Invalid distribution"):
            get_score(10.0, 2.0, 0.5, dist="invalid", metric="invalid")


class TestGetEpiweek:
    def test_returns_tuple(self):
        from datetime import date

        result = get_epiweek(date(2023, 1, 1))
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGetCiColumns:
    def test_pred_column(self):
        cols = get_ci_columns(np.array([0.5]))
        assert "pred" in cols

    def test_lower_column(self):
        cols = get_ci_columns(np.array([0.025]))
        assert any("lower" in c for c in cols)

    def test_upper_column(self):
        cols = get_ci_columns(np.array([0.975]))
        assert any("upper" in c for c in cols)

    def test_multiple_columns(self):
        cols = get_ci_columns(np.array([0.025, 0.5, 0.975]))
        assert len(cols) == 3


class TestEnsemble:
    def test_init(self, sample_ensemble_df):
        order = ["model_a", "model_b", "model_c"]
        ens = EnsembleDistPool(
            df=sample_ensemble_df,
            order_models=order,
            mixture="log",
            dist="log_normal",
        )
        assert ens.order_models == order
        assert ens.dist == "log_normal"

    def test_init_invalid_columns(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "pred": [50.0]})
        with pytest.raises(ValueError, match="Missing required keys in the pred"):
            EnsembleDistPool(df=df, order_models=["model_a"])

    def test_compute_weights(self, sample_ensemble_df, sample_obs_df):
        order = ["model_a", "model_b", "model_c"]
        ens = EnsembleDistPool(
            df=sample_ensemble_df,
            order_models=order,
            mixture="log",
            dist="log_normal",
        )
        weights = ens.compute_weights(df_obs=sample_obs_df, metric="crps")
        assert "weights" in weights
        assert "loss" in weights

    def test_apply_ensemble(self, sample_ensemble_df, sample_obs_df):
        order = ["model_a", "model_b", "model_c"]
        ens = EnsembleDistPool(
            df=sample_ensemble_df,
            order_models=order,
            mixture="log",
            dist="log_normal",
        )
        ens.compute_weights(df_obs=sample_obs_df, metric="crps")
        result = ens.apply_ensemble()
        assert "date" in result.columns
        assert "pred" in result.columns

    def test_apply_ensemble_no_weights(self, sample_ensemble_df):
        order = ["model_a", "model_b", "model_c"]
        ens = EnsembleDistPool(
            df=sample_ensemble_df,
            order_models=order,
            mixture="log",
            dist="log_normal",
        )
        with pytest.raises(ValueError, match="Weights must be computed first"):
            ens.apply_ensemble()

    def test_linear_mixture(self, sample_ensemble_df, sample_obs_df):
        order = ["model_a", "model_b", "model_c"]
        ens = EnsembleDistPool(
            df=sample_ensemble_df,
            order_models=order,
            mixture="linear",
            dist="log_normal",
        )
        weights = ens.compute_weights(df_obs=sample_obs_df, metric="crps")
        result = ens.apply_ensemble()
        assert "date" in result.columns


class TestDlnormMix:
    def test_basic(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.5])
        weights = np.array([0.5, 0.5])
        result = dlnorm_mix(10.0, mu, sigma, weights)
        assert isinstance(result, (float, np.floating))

    def test_array_obs(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.5])
        weights = np.array([0.5, 0.5])
        result = dlnorm_mix(np.array([10.0, 20.0]), mu, sigma, weights)
        assert len(result) == 2

    def test_log_true(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.5])
        weights = np.array([0.5, 0.5])
        result = dlnorm_mix(10.0, mu, sigma, weights, log=True)
        assert isinstance(result, (float, np.floating))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same length"):
            dlnorm_mix(
                10.0,
                np.array([1.0]),
                np.array([0.5, 0.5]),
                np.array([0.5, 0.5]),
            )


class TestComputePpf:
    def test_basic(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.5])
        weights = np.array([0.5, 0.5])
        result = compute_ppf(mu, sigma, weights)
        assert len(result) == 3


class TestCrpsLognormalMix:
    def test_basic(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.5])
        weights = np.array([0.5, 0.5])
        result = crps_lognormal_mix(10.0, mu, sigma, weights)
        assert isinstance(result, float)

    def test_array_obs(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5, 0.5])
        weights = np.array([0.5, 0.5])
        result = crps_lognormal_mix(np.array([10.0, 20.0]), mu, sigma, weights)
        assert len(result) == 2

    def test_mismatch_raises_error(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([0.5])
        weights = np.array([0.5, 0.5])
        with pytest.raises(
            ValueError, match="mu and sigma should be the same length"
        ):
            crps_lognormal_mix(10.0, mu, sigma, weights)


class TestFindOptWeightsLinear:
    def test_lognormal(self, sample_obs_df, sample_ensemble_df):
        preds = sample_ensemble_df.copy()
        preds = preds[["date", "pred", "lower_90", "upper_90", "model_id"]]
        preds[["mu", "sigma"]] = preds.apply(
            lambda row: [np.log(row["pred"]), 0.5],
            axis=1,
            result_type="expand",
        )

        result = find_opt_weights_linear(
            obs=sample_obs_df,
            preds=preds[["date", "mu", "sigma", "model_id"]],
            order_models=["model_a", "model_b", "model_c"],
            dist="log_normal",
            metric="crps",
        )
        assert "weights" in result
        assert "loss" in result


class TestGetQuantilesLog:
    def test_lognormal(self):
        weights = np.array([0.5, 0.5])
        ms = np.array([1.0, 2.0])
        vs = np.array([0.25, 0.25])
        p = np.array([0.5])
        result = get_quantiles_log("log_normal", weights, ms, vs, p)
        assert len(result) == 1


class TestGetQuantilesLinear:
    def test_lognormal(self):
        preds = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "pred": [50.0],
                "mu": [3.0],
                "sigma": [0.5],
                "model_id": ["test"],
            }
        )
        weights = np.array([1.0])
        p = np.array([0.5])
        result = get_quantiles_linear("log_normal", weights, preds, p)
        assert len(result) == 1
