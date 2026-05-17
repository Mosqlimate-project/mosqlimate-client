"""Tests for the scoring module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock


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


from mosqlient.scoring.score import (
    evaluate_point_metrics,
    compute_interval_score,
    compute_wis,
    plot_bar_score,
    plot_score,
    Scorer,
)


class TestEvaluatePointMetrics:
    def test_mae(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.2, 3.3, 4.4, 5.5]
        result = evaluate_point_metrics(y_true, y_pred, "MAE")
        assert isinstance(result, float)

    def test_mse(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.2, 3.3, 4.4, 5.5]
        result = evaluate_point_metrics(y_true, y_pred, "MSE")
        assert isinstance(result, float)


class TestComputeIntervalScore:
    def test_basic(self):
        score = compute_interval_score(
            lower_bound=40.0, upper_bound=60.0, observed_value=50.0, alpha=0.05
        )
        assert isinstance(score, (float, np.ndarray))

    def test_observation_outside_interval(self):
        score = compute_interval_score(
            lower_bound=40.0,
            upper_bound=60.0,
            observed_value=100.0,
            alpha=0.05,
        )
        assert score > 20.0

    def test_array_inputs(self):
        score = compute_interval_score(
            lower_bound=np.array([40.0, 30.0]),
            upper_bound=np.array([60.0, 50.0]),
            observed_value=np.array([50.0, 40.0]),
            alpha=0.05,
        )
        assert len(score) == 2


class TestComputeWis:
    def test_basic(self):
        df = pd.DataFrame(
            {
                "pred": [50.0],
                "lower_50": [45.0],
                "upper_50": [55.0],
                "lower_80": [40.0],
                "upper_80": [60.0],
                "lower_90": [35.0],
                "upper_90": [65.0],
                "lower_95": [30.0],
                "upper_95": [70.0],
            }
        )
        observed = np.array([50.0])
        result = compute_wis(df, observed)
        assert len(result) == 1

    def test_custom_weights(self):
        df = pd.DataFrame(
            {
                "pred": [50.0],
                "lower_50": [45.0],
                "upper_50": [55.0],
            }
        )
        observed = np.array([50.0])
        result = compute_wis(df, observed, w_k=np.array([0.25]))
        assert len(result) == 1

    def test_weights_mismatch(self):
        df = pd.DataFrame(
            {
                "pred": [50.0],
                "lower_50": [45.0],
                "upper_50": [55.0],
                "lower_80": [40.0],
                "upper_80": [60.0],
            }
        )
        observed = np.array([50.0])
        with pytest.raises(ValueError, match="Weights length"):
            compute_wis(df, observed, w_k=np.array([0.1]))

    def test_scalar_observed(self):
        df = pd.DataFrame(
            {
                "pred": [50.0],
                "lower_90": [40.0],
                "upper_90": [60.0],
            }
        )
        result = compute_wis(df, 50.0)
        assert len(result) == 1


class TestPlotBarScore:
    def test_returns_chart(self):
        data = pd.DataFrame({"id": [1, 2], "mae": [0.5, 0.3]})
        chart = plot_bar_score(data, "mae")
        assert chart is not None


class TestPlotScore:
    def test_returns_chart(self):
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "casos": [10, 20, 30],
            }
        )
        df_melted = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "variable": ["model_a"] * 3,
                "CRPS_score": [0.5, 0.3, 0.4],
            }
        )
        chart = plot_score(data, df_melted, score="CRPS")
        assert chart is not None

    def test_different_scores(self):
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "casos": [10, 20, 30],
            }
        )
        df_melted = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="W-SUN"),
                "variable": ["model_a"] * 3,
                "interval_score": [0.5, 0.3, 0.4],
            }
        )
        for score in ["interval", "wis", "log"]:
            df_melted_copy = df_melted.copy()
            df_melted_copy.columns = ["date", "variable", f"{score}_score"]
            chart = plot_score(data, df_melted_copy, score=score)
            assert chart is not None


class TestScorer:
    def test_init_missing_columns(self, sample_df_true):
        df_bad = sample_df_true.drop(columns=["casos"])
        with pytest.raises(ValueError, match="Missing required keys"):
            Scorer(api_key="test:123", df_true=df_bad, pred=sample_df_true)

    def test_init_pred_missing_columns(self, sample_df_true, sample_df_pred):
        df_bad = sample_df_pred.drop(columns=["pred"])
        with pytest.raises(ValueError, match="Missing required keys"):
            Scorer(api_key="test:123", df_true=sample_df_true, pred=df_bad)

    def test_init_no_ids_no_pred(self, sample_df_true):
        with pytest.raises(
            ValueError, match="It must be provide and id or DataFrame"
        ):
            Scorer(
                api_key="test:123", df_true=sample_df_true, ids=[], pred=None
            )

    def test_init_with_pred(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        assert "pred" in scorer.dict_df_ids

    def test_init_with_ids(
        self,
        sample_df_true,
        valid_api_key,
        sample_model_data,
        mock_openapi_response,
    ):
        pred_data = [
            {
                "id": 1,
                "model": sample_model_data,
                "disease": "A90",
                "commit": "a" * 40,
                "description": "Test",
                "case_definition": "probable",
                "published": True,
                "created_at": "2023-01-01",
                "adm_level": 1,
                "adm_0": "BRA",
                "adm_1": 33,
            }
        ]
        prediction_rows = [
            {
                "date": "2023-01-01",
                "lower_90": 10.0,
                "pred": 50.0,
                "upper_90": 90.0,
            }
        ]
        session_mock = make_aiohttp_patch(pred_data)

        def mock_get_side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            if "openapi" in url:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "paths": mock_openapi_response["paths"]
                }
                return mock_resp
            elif "predictions" in url and "data" in url:
                mock_resp = MagicMock()
                mock_resp.json.return_value = prediction_rows
                return mock_resp
            else:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"items": pred_data}
                return mock_resp

        with patch(
            "mosqlient.client.requests.get", side_effect=mock_get_side_effect
        ):
            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                scorer = Scorer(
                    api_key=valid_api_key,
                    df_true=sample_df_true,
                    ids=[1],
                )
                assert scorer.ids == ["1"]

    def test_init_id_not_found(self, valid_api_key, mock_openapi_response):
        session_mock = make_aiohttp_patch([])
        with patch("mosqlient.client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "paths": mock_openapi_response["paths"]
            }
            mock_response_2 = MagicMock()
            mock_response_2.json.return_value = {"items": []}
            mock_get.side_effect = [mock_response, mock_response_2]

            with patch(
                "mosqlient.client.ClientSession", return_value=session_mock
            ):
                with pytest.raises(ValueError, match="No Prediction found"):
                    Scorer(
                        api_key=valid_api_key,
                        df_true=pd.DataFrame(
                            {
                                "date": pd.date_range(
                                    "2023-01-01", periods=1, freq="W-SUN"
                                ),
                                "casos": [10],
                            }
                        ),
                        ids=[999],
                    )

    def test_set_date_range(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        scorer.set_date_range("2023-01-15", "2023-02-15")
        assert scorer.filtered_df_true is not None

    def test_set_date_range_invalid(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        with pytest.raises(ValueError, match="must be between"):
            scorer.set_date_range("2000-01-01", "2000-02-01")

    def test_mae(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        mae = scorer.mae
        assert isinstance(mae, dict)
        assert "pred" in mae

    def test_mse(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        mse = scorer.mse
        assert isinstance(mse, dict)

    def test_crps(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        curve, mean = scorer.crps
        assert isinstance(curve, dict)
        assert isinstance(mean, dict)

    def test_log_score(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        curve, mean = scorer.log_score
        assert isinstance(curve, dict)
        assert isinstance(mean, dict)

    def test_interval_score(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        curve, mean = scorer.interval_score
        assert isinstance(curve, dict)
        assert isinstance(mean, dict)

    def test_wis(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        curve, mean = scorer.wis
        assert isinstance(curve, dict)
        assert isinstance(mean, dict)

    def test_summary(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        summary = scorer.summary
        assert isinstance(summary, pd.DataFrame)
        assert "mae" in summary.columns

    def test_plot_mae(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        chart = scorer.plot_mae()
        assert chart is not None

    def test_plot_mse(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        chart = scorer.plot_mse()
        assert chart is not None

    def test_plot_crps(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        scorer.crps
        chart = scorer.plot_crps()
        assert chart is not None

    def test_plot_log_score(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        scorer.log_score
        chart = scorer.plot_log_score()
        assert chart is not None

    def test_plot_interval_score(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        scorer.interval_score
        chart = scorer.plot_interval_score()
        assert chart is not None

    def test_plot_wis(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        scorer.wis
        chart = scorer.plot_wis()
        assert chart is not None

    def test_plot_predictions_with_ci(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        chart = scorer.plot_predictions(show_ci=True)
        assert chart is not None

    def test_plot_predictions_without_ci(self, sample_df_true, sample_df_pred):
        scorer = Scorer(
            api_key="test:123",
            df_true=sample_df_true,
            pred=sample_df_pred,
        )
        chart = scorer.plot_predictions(show_ci=False)
        assert chart is not None
