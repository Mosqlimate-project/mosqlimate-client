"""
Zero-shot weekly forecasting with TimesFM 3.0.

The module wraps the pretrained TimesFM 3.0 PyTorch evaluator. It does not
train or fine-tune a model.

Covariate types supported by TimesFM 3.0:
- past-only: observed in the historical context; no future values required.
- past-and-future: observed historically and known/provided for every forecast
  timestamp (e.g. calendar variables or externally supplied climate forecasts).

Input data must contain one row per weekly timestamp.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from timesfm3 import ModelConfig, TimesFM3Evaluator
from sklearn.isotonic import IsotonicRegression

_INV_CDF = NormalDist().inv_cdf


def map_quantile_curve(
    avail_levels: np.ndarray,
    avail_values: np.ndarray,
    target_level: float,
) -> float:
    """Interpolate/extrapolate a non-negative predictive quantile.

    The lower tail is interpolated between zero and the first available
    quantile using a Gaussian CDF truncated to that interval. This does
    not estimate a point mass at zero. The upper Gaussian extrapolation
    is anchored at the largest available quantile for continuity.

    Notes
    -----
    Input quantiles are expected to be non-decreasing; use
    ``enforce_quantile_order`` on the final output to repair any crossings.
    """
    levels = np.asarray(avail_levels, dtype=float)
    values = np.asarray(avail_values, dtype=float)

    if levels.ndim != 1 or values.ndim != 1 or levels.size != values.size:
        raise ValueError(
            "Quantile levels and values must be matching 1D arrays."
        )
    if (
        levels.size < 2
        or not np.isfinite(levels).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("At least two finite quantiles are required.")
    if not (0.0 < target_level < 1.0):
        raise ValueError(
            "target_level must lie strictly between zero and one."
        )

    order = np.argsort(levels)
    levels = levels[order]
    values = values[order]
    if levels[0] <= 0 or levels[-1] >= 1 or np.any(np.diff(levels) <= 0):
        raise ValueError(
            "Quantile levels must be unique and between zero and one."
        )

    # Non-negative support. Source quantile crossings, if any, are handled
    # by the final isotonic correction over the entire returned quantile curve.
    values = np.maximum(values, 0.0)
    p_low, p_high = float(levels[0]), float(levels[-1])
    q_low, q_high = float(values[0]), float(values[-1])

    if p_low <= target_level <= p_high:
        return float(np.interp(target_level, levels, values))

    sigma = (q_high - q_low) / (_INV_CDF(p_high) - _INV_CDF(p_low))

    if target_level < p_low:
        if q_low <= 0.0:
            return 0.0
        if sigma <= 0.0:
            return float(q_low * target_level / p_low)

        mu = q_low - _INV_CDF(p_low) * sigma
        normal = NormalDist(mu=mu, sigma=sigma)
        cdf_zero = normal.cdf(0.0)
        cdf_low = normal.cdf(q_low)
        fraction = target_level / p_low

        # Handle numerical collapse of the narrow conditional interval.
        if cdf_low <= cdf_zero:
            return float(q_low * fraction)

        probability = cdf_zero + fraction * (cdf_low - cdf_zero)
        probability = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
        return float(np.clip(normal.inv_cdf(probability), 0.0, q_low))

    if sigma <= 0.0:
        return max(q_high, 0.0)

    value = q_high + sigma * (_INV_CDF(target_level) - _INV_CDF(p_high))
    return float(max(value, q_high, 0.0))


def enforce_quantile_order(df: pd.DataFrame) -> pd.DataFrame:
    """Fix negative/crossing quantiles only for rows violating constraints.

    Rows already in non-decreasing order with non-negative values are
    returned unchanged. Isotonic regression minimizes squared adjustments
    on the remaining rows.
    """
    quantile_cols = [
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
    levels = np.array(
        [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975], dtype=float
    )

    missing = [col for col in quantile_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot enforce quantile order: missing columns {missing}. "
            "Call this function after all quantile columns are populated."
        )

    result = df.copy()
    values = result.loc[:, quantile_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast quantiles must be finite.")

    crossing_mask = np.any(np.diff(values, axis=1) < 0, axis=1)
    negative_mask = np.any(values < 0, axis=1)
    correction_mask = crossing_mask | negative_mask
    if not correction_mask.any():
        return result

    model = IsotonicRegression(increasing=True, y_min=0.0)
    corrected = np.stack(
        [model.fit_transform(levels, row) for row in values[correction_mask]]
    )
    result.loc[correction_mask, quantile_cols] = corrected
    return result


class ForecastTimesFM3:
    """Run zero-shot forecasts with a pretrained TimesFM 3.0 model."""

    QUANTILE_LEVELS = tuple(np.arange(0.1, 1.0, 0.1).round(1))
    OUTPUT_QUANTILES = {
        0.025: "lower_95",
        0.05: "lower_90",
        0.10: "lower_80",
        0.25: "lower_50",
        0.50: "pred",
        0.75: "upper_50",
        0.90: "upper_80",
        0.95: "upper_90",
        0.975: "upper_95",
    }

    def __init__(
        self,
        checkpoint_path: str = "google/timesfm-3.0-pytorch",
        device: str = "cuda",
        per_core_batch_size: int = 16,
        date_col: str = "date",
        target_col: str = "casos",
        frequency: str = "W",
        forecaster: Optional[TimesFM3Evaluator] = None,
    ) -> None:
        """Initialize the pretrained TimesFM 3.0 forecasting model.

        The model is loaded from a pretrained checkpoint and used for
        zero-shot time series forecasting. Alternatively, an existing
        TimesFM3Evaluator instance can be provided.

        Parameters
        ----------
        checkpoint_path : str, default="google/timesfm-3.0-pytorch"
            Path or Hugging Face identifier of the pretrained TimesFM 3.0
            checkpoint. Ignored when `forecaster` is provided.
        device : str, default="cuda"
            Device used for model inference, such as "cuda" or "cpu".
            Ignored when `forecaster` is provided.
        per_core_batch_size : int, default=16
            Batch size used by the TimesFM evaluator during inference.
            Ignored when `forecaster` is provided.
        date_col : str, default="date"
            Name of the column containing observation dates. The same
            name is used for the datetime index after preprocessing.
        target_col : str, default="casos"
            Name of the column containing the target time series values
            to be forecast.
        frequency : str, default="W"
            Pandas-compatible frequency string defining the temporal
            resolution of the time series. By default, weekly frequency
            is assumed.
        forecaster : TimesFM3Evaluator or None, default=None
            Optional pretrained TimesFM 3.0 evaluator. When provided,
            this instance is used directly instead of initializing
            a new model from `checkpoint_path`.

        Attributes
        ----------
        date_col : str
            Name of the date column.
        target_col : str
            Name of the target column.
        frequency : str
            Temporal frequency of the time series.
        forecaster : TimesFM3Evaluator
            Initialized or externally provided TimesFM 3.0 evaluator."""

        self.date_col = date_col
        self.target_col = target_col
        self.frequency = frequency

        if forecaster is None:
            config = ModelConfig(
                checkpoint_path=checkpoint_path,
                per_core_batch_size=per_core_batch_size,
                device=device,
            )
            self.forecaster = TimesFM3Evaluator(config)
        else:
            self.forecaster = forecaster

    def _prepare_history(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare historical time series data for forecasting.

        The method converts the date column or index to a DatetimeIndex,
        sorts observations chronologically, and validates the target
        variable. All original columns are preserved.

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame containing historical observations. Dates must be
            provided either as a column named `self.date_col` or as the
            DataFrame index. The DataFrame must contain the target column
            specified by `self.target_col`.

        Returns
        -------
        pd.DataFrame
            Copy of the input DataFrame with a sorted DatetimeIndex
            named `self.date_col`. The target column is converted to
            numeric values and validated.

        Raises
        ------
        ValueError
            If `df_data` is not a non-empty pandas DataFrame, contains
            duplicated timestamps, does not contain the target column,
            or contains missing or non-finite target values.
        TypeError
            If the date index cannot be converted to datetime or the
            target column contains values that cannot be converted
            to numeric values.

        Notes
        -----
        The method does not resample the time series, fill missing
        timestamps, or impute missing target values. Chronological
        sorting is performed automatically.

        The original DataFrame is not modified.
        """

        if not isinstance(df_data, pd.DataFrame) or df_data.empty:
            raise ValueError("df_data must be a non-empty pandas DataFrame.")

        df = df_data.copy()
        if self.date_col in df.columns:
            df = df.set_index(self.date_col)

        df.index = pd.to_datetime(df.index)
        df.index.name = self.date_col
        df = df.sort_index()

        if df.index.has_duplicates:
            raise ValueError("The date index contains duplicated timestamps.")
        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' was not found."
            )
        if not df.index.is_monotonic_increasing:
            raise ValueError("Dates must be sorted in ascending order.")

        df[self.target_col] = pd.to_numeric(
            df[self.target_col], errors="raise"
        )
        if df[self.target_col].isna().any():
            raise ValueError("The historical target contains missing values.")
        if not np.isfinite(df[self.target_col].to_numpy(dtype=float)).all():
            raise ValueError(
                "The historical target contains non-finite values."
            )
        return df

    @staticmethod
    def _numeric_matrix(
        df: pd.DataFrame,
        columns: Sequence[str],
        expected_rows: int,
        label: str,
    ) -> Optional[np.ndarray]:
        """Return selected columns as a (n_covariates, n_timestamps) array."""
        if not columns:
            return None

        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing {label} covariate columns: {missing}")

        values = df.loc[:, list(columns)].apply(pd.to_numeric, errors="raise")
        array = values.to_numpy(dtype=np.float32).T
        if array.shape[1] != expected_rows:
            raise ValueError(
                f"{label} covariates must contain {expected_rows} timestamps; "
                f"received {array.shape[1]}."
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"{label} covariates contain missing or non-finite values."
            )
        return array

    def forecast(
        self,
        df_data: pd.DataFrame,
        predict_n: int,
        look_back: Optional[int] = None,
        past_only_covariates: Optional[Sequence[str]] = None,
        past_future_covariates: Optional[Sequence[str]] = None,
        future_covariates: Optional[pd.DataFrame] = None,
        return_quantiles: bool = True,
        use_symmetric_averaging: bool = False,
    ) -> pd.DataFrame:
        """Forecast `predict_n` steps beyond the final historical date.

        Parameters
        ----------
        df_data : pd.DataFrame
            Historical data with the target and selected historical covariates.
            Dates may be in `date_col` or in a DatetimeIndex.
        predict_n : int
            Number of future time steps to forecast.
        look_back : int, optional
            Number of most recent observations used as model context. If None,
            all available historical observations are used.
        past_only_covariates : sequence of str, optional
            Covariates available only for the historical context.
        past_future_covariates : sequence of str, optional
            Covariates with both historical and future values.
        future_covariates : pd.DataFrame, optional
            Future covariates indexed by the exact forecast dates.
        return_quantiles : bool
            Whether to return the mapped quantile columns.
        use_symmetric_averaging : bool
            Forwarded to TimesFM's predict_batch method.
        """
        if (
            not isinstance(predict_n, int)
            or isinstance(predict_n, bool)
            or predict_n < 1
        ):
            raise ValueError("predict_n must be a positive integer.")
        if look_back is not None and (
            not isinstance(look_back, int)
            or isinstance(look_back, bool)
            or look_back < 1
        ):
            raise ValueError("look_back must be a positive integer or None.")

        past_only_covariates = list(past_only_covariates or [])
        past_future_covariates = list(past_future_covariates or [])
        if set(past_only_covariates) & set(past_future_covariates):
            raise ValueError(
                "A covariate cannot be both past-only and past-and-future."
            )
        if self.target_col in past_only_covariates + past_future_covariates:
            raise ValueError("The target column cannot also be a covariate.")

        history = self._prepare_history(df_data)
        if look_back is not None:
            if look_back > len(history):
                raise ValueError(
                    f"look_back ({look_back}) exceeds the number of available "
                    f"historical observations ({len(history)})."
                )
            history = history.tail(look_back).copy()

        context_len = len(history)
        target = history[self.target_col].to_numpy(dtype=np.float32)[None, :]
        past_only = self._numeric_matrix(
            history, past_only_covariates, context_len, "past-only"
        )

        future_dates = pd.date_range(
            start=history.index[-1]
            + pd.tseries.frequencies.to_offset(self.frequency),
            periods=predict_n,
            freq=self.frequency,
        )

        past_future = None
        if past_future_covariates:
            if future_covariates is None:
                raise ValueError(
                    "future_covariates is required when past-and-future "
                    "covariates are selected."
                )

            future = future_covariates.copy()
            if self.date_col in future.columns:
                future = future.set_index(self.date_col)
            future.index = pd.to_datetime(future.index)
            future.index.name = self.date_col
            future = future.sort_index()

            if future.index.has_duplicates:
                raise ValueError(
                    "future_covariates contains duplicated dates."
                )
            if not future.index.equals(future_dates):
                raise ValueError(
                    "future_covariates index must match the forecast dates exactly. "
                    f"Expected {future_dates[0].date()} to "
                    f"{future_dates[-1].date()} ({predict_n} rows)."
                )

            historical_cov = self._numeric_matrix(
                history,
                past_future_covariates,
                context_len,
                "historical past-and-future",
            )
            future_cov = self._numeric_matrix(
                future,
                past_future_covariates,
                predict_n,
                "future",
            )
            past_future = np.concatenate([historical_cov, future_cov], axis=1)

        outputs = list(
            self.forecaster.predict_batch(
                contexts=[target],
                horizon=predict_n,
                past_only_covariates=(
                    [past_only] if past_only is not None else None
                ),
                past_future_covariates=(
                    [past_future] if past_future is not None else None
                ),
                return_quantiles=return_quantiles,
                use_symmetric_averaging=use_symmetric_averaging,
            )
        )
        if not outputs:
            raise RuntimeError("TimesFM returned no forecast.")

        forecast = np.asarray(outputs[0].forecast).reshape(-1)
        if forecast.size != predict_n:
            raise RuntimeError(
                f"Unexpected forecast size {forecast.size}; expected {predict_n}."
            )

        result = pd.DataFrame(
            {
                "date": future_dates,
                "horizon": np.arange(1, predict_n + 1),
            }
        )

        if return_quantiles:
            quantiles = np.asarray(outputs[0].quantiles)
            expected_size = predict_n * len(self.QUANTILE_LEVELS)
            if quantiles.size != expected_size:
                raise RuntimeError(
                    f"Unexpected quantile size {quantiles.size}; "
                    f"expected {expected_size}."
                )
            quantiles = quantiles.reshape(predict_n, len(self.QUANTILE_LEVELS))

            for level, column in self.OUTPUT_QUANTILES.items():
                mapped_values = [
                    map_quantile_curve(
                        np.asarray(self.QUANTILE_LEVELS),
                        quantiles[step],
                        level,
                    )
                    for step in range(predict_n)
                ]
                result[column] = np.asarray(mapped_values, dtype=float)

            result = enforce_quantile_order(result)
        else:
            result["pred"] = np.maximum(forecast.astype(float), 0.0)

        return result
