from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class ForecastXGB:
    """XGBoost forecaster with lag, rolling and calendar features."""

    def __init__(
        self,
        df_data: pd.DataFrame,
        columns: list[str] | None = None,
        date_col: str = "date",
        target_col: str = "casos",
        look_back: int = 4,
        predict_n: int = 4,
        n_estimators: int = 500,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        random_state: int = 42,
        residual: bool = False,
        **xgb_kwargs,
    ):
        """Initializes the ForecastXGB handler and validates dataset columns.

        Args:
            df_data (pd.DataFrame): Input dataframe containing time-series data.
            columns (list, optional): List of feature columns to use. Defaults to None.
            date_col (str, optional): Name of datetime column. Defaults to 'date'.
            target_col (str, optional): Target column to forecast. Defaults to 'casos'.
            look_back (int, optional): Number of past steps to look back. Defaults to 4.
            predict_n (int, optional): Number of steps ahead to predict. Defaults to 4.
            n_estimators (int, optional): Number of boosting rounds. Defaults to 500.
            max_depth (int, optional): Maximum tree depth. Defaults to 3.
            learning_rate (float, optional): Boosting learning rate. Defaults to 0.03.
            random_state (int, optional): Random seed. Defaults to 42.
            residual (bool, optional): Learn corrections over the recent-case baseline. Defaults to False.
            **xgb_kwargs: Additional parameters passed to XGBRegressor.

        Raises:
            ValueError: If `target_col` is not found within `columns`.
        """
        self.columns = columns if columns is not None else []
        self.target_col = target_col
        self.date_col = date_col
        self.look_back = look_back
        self.predict_n = predict_n
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.residual = residual
        self.xgb_kwargs = xgb_kwargs

        if self.target_col not in self.columns:
            raise ValueError(
                f"'{self.target_col}' must be one of the available columns: {self.columns}."
            )

        df_model = df_data[self.columns].copy()
        if date_col in df_data.columns and date_col not in df_model.columns:
            df_model[date_col] = df_data[date_col]

        df_model = df_model.set_index(self.date_col)
        df_model.index = pd.to_datetime(df_model.index)
        self.df_model = df_model.sort_index()

        self.model: XGBRegressor | None = None
        self.feature_names: list[str] = []
        self.X_train: pd.DataFrame | None = None
        self.Y_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.Y_test: pd.DataFrame | None = None

    def _create_features(
        self,
        df: pd.DataFrame,
        use_log: bool = True,
        require_complete_targets: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Engineers lag features, rolling statistics, cyclical features, and momentum.

        Returns (X, Y) matrices.
        """
        df_feat = df.copy()

        if use_log:
            df_feat[self.target_col] = np.log1p(
                df_feat[self.target_col].clip(lower=0)
            )

        feature_df = pd.DataFrame(index=df_feat.index)

        # Calendar features are known at forecast time.
        if isinstance(df_feat.index, pd.DatetimeIndex):
            epiweek = df_feat.index.isocalendar().week.astype(float)
            feature_df["epiweek"] = epiweek
            feature_df["month"] = df_feat.index.month.astype(float)

        # Lags for the target and any supplied predictors.
        for col in self.columns:
            for lag in range(1, self.look_back + 1):
                if lag <= len(df_feat):
                    feature_df[f"{col}_lag_{lag}"] = df_feat[col].shift(lag)

            if col == self.target_col:
                previous = df_feat[col].shift(1)
                feature_df[f"{col}_roll_mean_{self.look_back}"] = (
                    previous.rolling(window=self.look_back).mean()
                )
                feature_df[f"{col}_roll_max_{self.look_back}"] = (
                    previous.rolling(window=self.look_back).max()
                )

        # 3. Multi-step target horizon (predict_n steps ahead)
        target_df = pd.DataFrame(index=df_feat.index)
        for h in range(1, self.predict_n + 1):
            target_df[f"target_h{h}"] = df_feat[self.target_col].shift(
                -(h - 1)
            )

        # Drop rows only where target is NaN (XGBoost handles feature NaNs natively)
        valid_mask = feature_df.index.isin(df_feat.index[self.look_back :])
        if require_complete_targets:
            valid_mask &= target_df.notna().all(axis=1)

        return feature_df.loc[valid_mask], target_df.loc[valid_mask]

    def _residual_baseline(self, X_data: pd.DataFrame) -> pd.Series:
        column = f"{self.target_col}_roll_mean_{self.look_back}"
        if column not in X_data:
            raise ValueError(
                f"Residual baseline feature is missing: '{column}'."
            )
        return X_data[column].fillna(0.0)

    def train(
        self,
        ini_train_date: str = "2020-01-01",
        end_train_date: str = "2024-12-31",
        end_date: str = "2025-12-31",
        use_log: bool = True,
        early_stopping_rounds: int = 40,
        val_ratio: float = 0.15,
        verbose: bool = False,
    ) -> tuple[XGBRegressor, dict]:
        """Preprocesses dates, generates lag/calendar features, and trains the XGBoost model.

        Args:
            ini_train_date (str, optional): Start date for training window. Defaults to '2020-01-01'.
            end_train_date (str, optional): End date for training window. Defaults to '2024-12-31'.
            end_date (str, optional): Total data limit boundary date. Defaults to '2025-12-31'.
            use_log (bool, optional): Apply log1p transform to target. Defaults to True.
            early_stopping_rounds (int, optional): Rounds for early stopping. Defaults to 40.
            val_ratio (float, optional): Fraction of training set used for validation early stopping. Defaults to 0.15.
            verbose (bool, optional): Verbosity level during fitting. Defaults to False.

        Returns:
            tuple: (trained_model, history_dict)
        """
        self.ini_train_date = ini_train_date
        self.end_train_date = end_train_date
        self.end_date = end_date
        self.use_log = use_log

        start_dt = pd.to_datetime(ini_train_date)
        end_train_dt = pd.to_datetime(end_train_date)
        end_dt = pd.to_datetime(end_date)

        df_filtered = self.df_model.loc[
            (self.df_model.index >= start_dt) & (self.df_model.index <= end_dt)
        ]

        X_all, Y_all = self._create_features(
            df_filtered, use_log=use_log, require_complete_targets=False
        )
        self.feature_names = list(X_all.columns)

        # Keep every label of a training row inside the training period.  A row
        # near the cutoff otherwise carries target_h2...target_hN from the test
        # period into XGBoost's fit.
        target_end_dates = pd.Series(
            df_filtered.index, index=df_filtered.index
        ).shift(-(self.predict_n - 1))
        target_end_dates = target_end_dates.reindex(X_all.index)

        # Train / Test split
        train_mask = (
            (X_all.index <= end_train_dt)
            & target_end_dates.notna()
            & (target_end_dates <= end_train_dt)
            & Y_all.notna().all(axis=1)
        )
        test_mask = X_all.index > end_train_dt

        self.X_train = X_all.loc[train_mask]
        self.Y_train = Y_all.loc[train_mask]
        self.X_test = X_all.loc[test_mask]
        self.Y_test = Y_all.loc[test_mask]

        if self.X_train.empty:
            raise ValueError(
                "Training subset is empty after feature creation."
            )

        fit_Y_all = self.Y_train.copy()
        if self.residual:
            fit_Y_all = fit_Y_all.sub(
                self._residual_baseline(self.X_train), axis=0
            )

        # Validation split for early stopping
        n_val = max(1, int(len(self.X_train) * val_ratio))
        X_tr, X_va = self.X_train.iloc[:-n_val], self.X_train.iloc[-n_val:]
        Y_tr, Y_va = fit_Y_all.iloc[:-n_val], fit_Y_all.iloc[-n_val:]

        model_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": early_stopping_rounds,
            **self.xgb_kwargs,
        }

        self.model = XGBRegressor(**model_params)

        if not X_va.empty and early_stopping_rounds > 0:
            self.model.fit(
                X_tr,
                Y_tr,
                eval_set=[(X_va, Y_va)],
                verbose=verbose,
            )
        else:
            self.model.fit(self.X_train, fit_Y_all, verbose=verbose)

        history = {
            "best_iteration": getattr(
                self.model, "best_iteration", self.n_estimators
            ),
            "best_score": getattr(self.model, "best_score", None),
        }

        return self.model, history

    def _format_predictions(
        self, X_data: pd.DataFrame, Y_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError(
                "The model must be trained using .train() before generating predictions."
            )

        preds = self.model.predict(X_data)
        if self.residual:
            baseline = self._residual_baseline(X_data).to_numpy()
            preds = (
                preds + baseline[:, None]
                if preds.ndim > 1
                else preds + baseline
            )
        if self.use_log:
            preds = np.expm1(preds)
            preds = np.clip(preds, a_min=0, a_max=None)

        df_res = pd.DataFrame(index=X_data.index)

        if len(preds.shape) > 1 and preds.shape[1] > 1:
            df_res["pred"] = preds[:, 0]
            for h in range(1, preds.shape[1] + 1):
                df_res[f"pred_h{h}"] = preds[:, h - 1]
        else:
            df_res["pred"] = preds

        df_res["date"] = df_res.index

        if Y_data is not None and not Y_data.empty:
            actual = self.df_model.loc[X_data.index, self.target_col]
            df_res[self.target_col] = actual.values

        return df_res

    def predict_in_sample(self) -> pd.DataFrame:
        """Generates forecast predictions for the training set."""
        if self.X_train is None:
            raise RuntimeError(
                "The model must be trained using .train() before generating predictions."
            )
        return self._format_predictions(self.X_train, self.Y_train)

    def predict_out_of_sample(self) -> pd.DataFrame:
        """Generates forecast predictions for the evaluation/test split."""
        if self.X_test is None:
            raise RuntimeError(
                "The model must be trained using .train() before generating predictions."
            )
        return self._format_predictions(self.X_test, self.Y_test)

    def forecast(self, end_date: str) -> pd.DataFrame:
        """Performs multi-step forecasting up to a specified end_date."""
        if self.model is None or self.X_train is None:
            raise RuntimeError(
                "The model must be trained before performing out-of-sample forecasts."
            )

        end_dt = pd.to_datetime(end_date)
        last_date = self.df_model.index.max()

        if end_dt <= last_date:
            X_all, _Y_all = self._create_features(
                self.df_model,
                use_log=self.use_log,
                require_complete_targets=False,
            )
            forecast_mask = X_all.index > self.X_train.index.max()
            X_fc = X_all.loc[forecast_mask]
            return self._format_predictions(X_fc)

        df_curr = self.df_model.copy()
        future_preds = []

        curr_date = last_date
        while curr_date < end_dt:
            X_step, _ = self._create_features(
                df_curr, use_log=self.use_log, require_complete_targets=False
            )
            if X_step.empty:
                break

            last_X = X_step.iloc[[-1]]
            pred_step = self.model.predict(last_X)
            if self.residual:
                pred_step = (
                    pred_step
                    + self._residual_baseline(last_X).to_numpy()[:, None]
                )
            if self.use_log:
                pred_step = np.expm1(pred_step)
                pred_step = np.clip(pred_step, a_min=0, a_max=None)

            next_val = (
                pred_step[0, 0] if len(pred_step.shape) > 1 else pred_step[0]
            )
            curr_date = curr_date + timedelta(days=7)

            new_row = {col: df_curr[col].iloc[-1] for col in df_curr.columns}
            new_row[self.target_col] = next_val
            df_curr.loc[curr_date] = new_row

            future_preds.append({"date": curr_date, "pred": next_val})

        df_out = pd.DataFrame(future_preds)
        if not df_out.empty:
            df_out["date"] = pd.to_datetime(df_out["date"])
        return df_out


class ForecastXGBResidual(ForecastXGB):
    """ForecastXGB variant that learns a correction over recent cases."""

    def __init__(self, *args, **kwargs):
        kwargs["residual"] = True
        super().__init__(*args, **kwargs)
