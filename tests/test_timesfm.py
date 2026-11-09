"""Lightweight contract tests for the zero-shot TimesFM 3 forecaster."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mosqlient.forecast.timesfm import (
    ForecastTimesFM3,
    enforce_quantile_order,
    map_quantile_curve,
)

QUANTILE_COLUMNS = [
    "lower_95",
    "lower_90",
    "lower_80",
    "lower_50",
    "pred",
    "upper_50",
    "upper_80",
    "upper_90",
    "upper_95",
]


@pytest.fixture
def sample_forecast_df():
    dates = pd.date_range("2020-01-05", periods=20, freq="W-SUN")
    return pd.DataFrame(
        {
            "date": dates,
            "casos": np.arange(20, dtype=float),
            "temp": np.linspace(20, 25, 20),
            "humidity": np.linspace(40, 60, 20),
        }
    )


@pytest.fixture
def future_forecast_df(sample_forecast_df):
    dates = pd.date_range(
        sample_forecast_df["date"].iloc[-1] + pd.Timedelta(weeks=1),
        periods=4,
        freq="W-SUN",
    )
    return pd.DataFrame(
        {
            "date": dates,
            "temp": np.array([26.0, 27.0, 28.0, 29.0]),
            "humidity": np.array([61.0, 62.0, 63.0, 64.0]),
        }
    )


class FakeTimesFM3Evaluator:
    """Deterministic evaluator with the same predict_batch call contract."""

    def __init__(self, quantiles=None, forecast=None, empty=False):
        self.calls = []
        self.quantiles = quantiles
        self.forecast_values = forecast
        self.empty = empty

    def predict_batch(self, **kwargs):
        self.calls.append(kwargs)
        if self.empty:
            return []

        horizon = kwargs["horizon"]
        forecast = (
            np.arange(1, horizon + 1, dtype=float)
            if self.forecast_values is None
            else np.asarray(self.forecast_values, dtype=float)
        )
        quantiles = (
            np.tile(np.arange(1, 10, dtype=float), (horizon, 1))
            if self.quantiles is None
            else np.asarray(self.quantiles, dtype=float)
        )
        return [SimpleNamespace(forecast=forecast, quantiles=quantiles)]


def make_model(evaluator=None, **kwargs):
    if evaluator is None:
        evaluator = FakeTimesFM3Evaluator()
    return ForecastTimesFM3(
        forecaster=evaluator,
        date_col="date",
        target_col="casos",
        frequency="W-SUN",
        **kwargs,
    )


def test_timesfm_contract():
    evaluator = FakeTimesFM3Evaluator()
    model = make_model(evaluator)

    assert model.date_col == "date"
    assert model.target_col == "casos"
    assert model.frequency == "W-SUN"
    assert model.forecaster is evaluator
    assert list(model.OUTPUT_QUANTILES.values()) == QUANTILE_COLUMNS


def test_prepare_history_sorts_without_mutating_input(sample_forecast_df):
    shuffled = sample_forecast_df.sample(frac=1.0, random_state=42)
    original = shuffled.copy(deep=True)
    model = make_model()

    history = model._prepare_history(shuffled)

    assert isinstance(history.index, pd.DatetimeIndex)
    assert history.index.name == "date"
    assert history.index.is_monotonic_increasing
    assert list(history.columns) == ["casos", "temp", "humidity"]
    np.testing.assert_array_equal(history["casos"], np.arange(20, dtype=float))
    pd.testing.assert_frame_equal(shuffled, original)


@pytest.mark.parametrize("invalid", [pd.DataFrame(), None, "not a dataframe"])
def test_prepare_history_rejects_empty_or_invalid_data(invalid):
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        make_model()._prepare_history(invalid)


def test_prepare_history_rejects_duplicate_dates(sample_forecast_df):
    invalid = pd.concat([sample_forecast_df, sample_forecast_df.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicated timestamps"):
        make_model()._prepare_history(invalid)


def test_prepare_history_rejects_missing_target(sample_forecast_df):
    with pytest.raises(ValueError, match="Target column"):
        make_model()._prepare_history(sample_forecast_df.drop(columns="casos"))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_prepare_history_rejects_invalid_target(sample_forecast_df, bad_value):
    invalid = sample_forecast_df.copy()
    invalid.loc[0, "casos"] = bad_value
    with pytest.raises(ValueError, match="missing values|non-finite values"):
        make_model()._prepare_history(invalid)


def test_numeric_matrix_transposes_and_validates(sample_forecast_df):
    matrix = ForecastTimesFM3._numeric_matrix(
        sample_forecast_df, ["temp", "humidity"], 20, "historical"
    )

    assert matrix.shape == (2, 20)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[0], sample_forecast_df["temp"])
    assert (
        ForecastTimesFM3._numeric_matrix(sample_forecast_df, [], 20, "test")
        is None
    )


@pytest.mark.parametrize(
    ("data", "columns", "expected_rows", "message"),
    [
        ({"temp": [1.0, 2.0]}, ["missing"], 2, "Missing test covariate"),
        ({"temp": [1.0, 2.0]}, ["temp"], 3, "must contain 3 timestamps"),
        ({"temp": [1.0, np.nan]}, ["temp"], 2, "non-finite values"),
    ],
)
def test_numeric_matrix_rejects_invalid_covariates(
    data, columns, expected_rows, message
):
    with pytest.raises(ValueError, match=message):
        ForecastTimesFM3._numeric_matrix(
            pd.DataFrame(data), columns, expected_rows, "test"
        )


def test_forecast_without_covariates_or_quantiles(sample_forecast_df):
    evaluator = FakeTimesFM3Evaluator(forecast=[-3.0, 4.0, 5.0])
    model = make_model(evaluator)

    result = model.forecast(
        sample_forecast_df, predict_n=3, return_quantiles=False
    )
    call = evaluator.calls[0]

    assert result.columns.tolist() == ["date", "horizon", "pred"]
    assert result["horizon"].tolist() == [1, 2, 3]
    assert result["pred"].tolist() == [0.0, 4.0, 5.0]
    assert (
        result["date"].tolist()
        == pd.date_range("2020-05-24", periods=3, freq="W-SUN").tolist()
    )
    assert call["horizon"] == 3
    assert call["return_quantiles"] is False
    assert call["past_only_covariates"] is None
    assert call["past_future_covariates"] is None
    assert len(call["contexts"]) == 1
    assert call["contexts"][0].shape == (1, 20)


def test_forecast_respects_look_back_and_past_only_covariates(
    sample_forecast_df,
):
    evaluator = FakeTimesFM3Evaluator()
    model = make_model(evaluator)

    model.forecast(
        sample_forecast_df,
        predict_n=2,
        look_back=5,
        past_only_covariates=["temp", "humidity"],
        return_quantiles=False,
        use_symmetric_averaging=True,
    )
    call = evaluator.calls[0]

    np.testing.assert_array_equal(
        call["contexts"][0][0], sample_forecast_df["casos"].tail(5)
    )
    matrix = call["past_only_covariates"][0]
    assert matrix.shape == (2, 5)
    np.testing.assert_allclose(matrix[0], sample_forecast_df["temp"].tail(5))
    assert call["use_symmetric_averaging"] is True


def test_forecast_combines_historical_and_future_covariates(
    sample_forecast_df, future_forecast_df
):
    evaluator = FakeTimesFM3Evaluator()
    model = make_model(evaluator)

    model.forecast(
        sample_forecast_df,
        predict_n=4,
        look_back=6,
        past_future_covariates=["temp", "humidity"],
        future_covariates=future_forecast_df,
        return_quantiles=False,
    )

    matrix = evaluator.calls[0]["past_future_covariates"][0]
    assert matrix.shape == (2, 10)
    np.testing.assert_allclose(
        matrix[0, :6], sample_forecast_df["temp"].tail(6)
    )
    np.testing.assert_allclose(matrix[0, 6:], future_forecast_df["temp"])
    np.testing.assert_allclose(matrix[1, 6:], future_forecast_df["humidity"])


def test_forecast_returns_nonnegative_ordered_quantiles(sample_forecast_df):
    # Original q10 is negative, so the forecast must repair the lower tail.
    original_quantiles = np.tile(
        np.array([-2.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), (3, 1)
    )
    evaluator = FakeTimesFM3Evaluator(quantiles=original_quantiles)
    result = make_model(evaluator).forecast(sample_forecast_df, predict_n=3)

    assert {"date", "horizon", *QUANTILE_COLUMNS} == set(result.columns)
    assert len(result) == 3
    values = result[QUANTILE_COLUMNS].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert (np.diff(values, axis=1) >= 0).all()
    assert (result["lower_80"] == 0).all()
    assert (result["pred"] == 4).all()


def test_forecast_repairs_crossing_quantiles(sample_forecast_df):
    # Crossing from q40=10 to q50=8 in the original evaluator output.
    original_quantiles = np.tile(
        np.array([1.0, 2.0, 3.0, 10.0, 8.0, 12.0, 13.0, 14.0, 15.0]), (2, 1)
    )
    result = make_model(
        FakeTimesFM3Evaluator(quantiles=original_quantiles)
    ).forecast(sample_forecast_df, predict_n=2)

    values = result[QUANTILE_COLUMNS].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()
    assert (values >= 0).all()


@pytest.mark.parametrize("predict_n", [0, -1, 1.5, True])
def test_forecast_rejects_invalid_horizon(sample_forecast_df, predict_n):
    with pytest.raises(
        ValueError, match="predict_n must be a positive integer"
    ):
        make_model().forecast(sample_forecast_df, predict_n=predict_n)


@pytest.mark.parametrize("look_back", [0, -1, 1.5, True, 21])
def test_forecast_rejects_invalid_look_back(sample_forecast_df, look_back):
    with pytest.raises(ValueError, match="look_back"):
        make_model().forecast(
            sample_forecast_df, predict_n=2, look_back=look_back
        )


def test_forecast_rejects_conflicting_covariate_groups(sample_forecast_df):
    with pytest.raises(ValueError, match="both past-only and past-and-future"):
        make_model().forecast(
            sample_forecast_df,
            predict_n=2,
            past_only_covariates=["temp"],
            past_future_covariates=["temp"],
        )


def test_forecast_rejects_target_as_covariate(sample_forecast_df):
    with pytest.raises(ValueError, match="target column cannot"):
        make_model().forecast(
            sample_forecast_df,
            predict_n=2,
            past_only_covariates=["casos"],
        )


def test_forecast_requires_future_covariates(sample_forecast_df):
    with pytest.raises(ValueError, match="future_covariates is required"):
        make_model().forecast(
            sample_forecast_df,
            predict_n=2,
            past_future_covariates=["temp"],
        )


def test_forecast_rejects_incorrect_future_dates(
    sample_forecast_df, future_forecast_df
):
    invalid = future_forecast_df.copy()
    invalid["date"] += pd.Timedelta(weeks=1)
    with pytest.raises(
        ValueError, match="index must match the forecast dates"
    ):
        make_model().forecast(
            sample_forecast_df,
            predict_n=4,
            past_future_covariates=["temp"],
            future_covariates=invalid,
        )


def test_forecast_rejects_duplicate_future_dates(
    sample_forecast_df, future_forecast_df
):
    invalid = future_forecast_df.copy()
    invalid.loc[1, "date"] = invalid.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicated dates"):
        make_model().forecast(
            sample_forecast_df,
            predict_n=4,
            past_future_covariates=["temp"],
            future_covariates=invalid,
        )


@pytest.mark.parametrize(
    ("evaluator", "message"),
    [
        (FakeTimesFM3Evaluator(empty=True), "no forecast"),
        (FakeTimesFM3Evaluator(forecast=[1.0]), "Unexpected forecast size"),
        (
            FakeTimesFM3Evaluator(quantiles=np.ones((1, 9))),
            "Unexpected quantile size",
        ),
    ],
)
def test_forecast_rejects_invalid_evaluator_results(
    sample_forecast_df, evaluator, message
):
    with pytest.raises(RuntimeError, match=message):
        make_model(evaluator).forecast(sample_forecast_df, predict_n=2)


def test_map_quantile_curve_interpolates_and_extrapolates():
    levels = np.arange(0.1, 1.0, 0.1)
    values = np.arange(1.0, 10.0)

    assert map_quantile_curve(levels, values, 0.1) == pytest.approx(1.0)
    assert map_quantile_curve(levels, values, 0.5) == pytest.approx(5.0)
    assert map_quantile_curve(levels, values, 0.25) == pytest.approx(2.5)
    assert 0.0 <= map_quantile_curve(levels, values, 0.025) <= 1.0
    assert map_quantile_curve(levels, values, 0.975) >= 9.0


def test_map_quantile_curve_allows_exact_zeros():
    levels = np.arange(0.1, 1.0, 0.1)
    values = np.array([-5.0, -3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert map_quantile_curve(levels, values, 0.025) == 0.0
    assert map_quantile_curve(levels, values, 0.1) == 0.0


@pytest.mark.parametrize(
    ("levels", "values", "target", "message"),
    [
        ([0.1], [1.0], 0.5, "At least two finite quantiles"),
        ([0.1, 0.2], [1.0], 0.5, "matching 1D arrays"),
        ([0.1, 0.1], [1.0, 2.0], 0.5, "must be unique"),
        ([0.1, 0.9], [1.0, np.nan], 0.5, "At least two finite quantiles"),
        ([0.1, 0.9], [1.0, 2.0], 1.0, "strictly between zero and one"),
    ],
)
def test_map_quantile_curve_rejects_invalid_inputs(
    levels, values, target, message
):
    with pytest.raises(ValueError, match=message):
        map_quantile_curve(np.array(levels), np.array(values), target)


def test_enforce_quantile_order_preserves_valid_rows():
    row = np.arange(9, dtype=float)
    df = pd.DataFrame([row], columns=QUANTILE_COLUMNS)
    df.insert(0, "date", pd.Timestamp("2020-01-05"))
    df.insert(1, "horizon", 1)

    actual = enforce_quantile_order(df)

    pd.testing.assert_frame_equal(actual, df)
    assert actual is not df


def test_enforce_quantile_order_repairs_only_invalid_rows():
    original = pd.DataFrame(
        [
            np.arange(9, dtype=float),
            [-2.0, 1.0, 3.0, 4.0, 2.0, 5.0, 6.0, 7.0, 8.0],
        ],
        columns=QUANTILE_COLUMNS,
    )
    corrected = enforce_quantile_order(original)

    np.testing.assert_array_equal(
        corrected.loc[0, QUANTILE_COLUMNS].to_numpy(dtype=float),
        original.loc[0, QUANTILE_COLUMNS].to_numpy(dtype=float),
    )
    adjusted = corrected.loc[1, QUANTILE_COLUMNS].to_numpy(dtype=float)
    assert (adjusted >= 0).all()
    assert (np.diff(adjusted) >= 0).all()
    pd.testing.assert_frame_equal(
        original,
        pd.DataFrame(
            [
                np.arange(9, dtype=float),
                [-2.0, 1.0, 3.0, 4.0, 2.0, 5.0, 6.0, 7.0, 8.0],
            ],
            columns=QUANTILE_COLUMNS,
        ),
    )


def test_enforce_quantile_order_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        enforce_quantile_order(pd.DataFrame({"date": ["2020-01-05"]}))


def test_enforce_quantile_order_rejects_nonfinite_values():
    df = pd.DataFrame([np.arange(9, dtype=float)], columns=QUANTILE_COLUMNS)
    df.loc[0, "pred"] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        enforce_quantile_order(df)
