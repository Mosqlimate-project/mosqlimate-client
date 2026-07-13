import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
from datetime import timedelta, datetime
from typing import cast, Tuple, Optional, Dict, Callable
from numpy.typing import NDArray


def get_next_n_weeks(ini_date: str, next_days: int) -> list:
    """
    Return a list of dates with the {next_days} days after ini_date.
    This function was designed to generate the dates of the forecast
    models.
    Parameters
    ----------
    ini_date : str
        Initial date.
    next_days : int
        Number of days to be included in the list after the date in
        ini_date.
    Returns
    -------
    list
        A list with the dates computed.
    """

    next_dates = []

    a = datetime.strptime(ini_date, "%Y-%m-%d")

    for i in np.arange(1, next_days + 1):
        d_i = datetime.strftime(a + timedelta(days=int(i * 7)), "%Y-%m-%d")

        next_dates.append(datetime.strptime(d_i, "%Y-%m-%d").date())

    return next_dates


def schedule(epoch, lr):
    return lr * math.exp(-0.1)


def normalize_data(
    df: pd.DataFrame,
    method: str = "max",
    ratio: Optional[float] = 0.75,
    ini_train_date: Optional[str] = None,
    end_train_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Normalize a DataFrame using either max normalization or robust (median/IQR) normalization
    computed on a training subset.

    The normalization factors are computed either:
    - From the first `ratio` portion of the dataset, OR
    - From a date range defined by `ini_train_date` and `end_train_date`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing only numeric (float) values.
    method: str, default="max"
        Method of normalization to apply. Options:
        - "max": Divides data by the column-wise maximum (x / max).
        - "robust": Subtracts the median and divides by the IQR ((x - median) / IQR).
    ratio : float or None, default=0.75
        Fraction of the dataset used for training normalization.
        Must be between 0 and 1. Ignored if `ini_train_date` and `end_train_date` are provided.
    ini_train_date : str or None, optional
        Start date (inclusive) for training period. Requires a DatetimeIndex.
    end_train_date : str or None, optional
        End date (inclusive) for training period. Requires a DatetimeIndex.

    Returns
    -------
    df_norm : pd.DataFrame
        Normalized DataFrame.
    norm_params : Dict[str, pd.Series]
        A dictionary containing the parameters used for normalization.
        - For 'max': {'max': max_values}
        - For 'robust': {'median': median_values, 'iqr': iqr_values}

    Raises
    ------
    TypeError
        If date-based normalization is requested but index is not DatetimeIndex.
    ValueError
        If ratio is invalid, training slice is empty, method is unrecognized,
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not np.issubdtype(df.dtypes.values[0], np.number):
        raise TypeError("DataFrame must contain only numeric values.")

    if method not in ["max", "robust"]:
        raise ValueError(
            "The 'method' parameter must be either 'max' or 'robust'."
        )

    df_clean = df.fillna(0)

    # --------------------------------------------------
    # Select training subset
    # --------------------------------------------------
    if ini_train_date and end_train_date:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                "DataFrame must have a DatetimeIndex for date-based normalization."
            )

        start = pd.to_datetime(ini_train_date)
        end = pd.to_datetime(end_train_date)

        if start >= end:
            raise ValueError(
                "ini_train_date must be earlier than end_train_date."
            )

        train_df = df_clean.loc[
            (df_clean.index >= start) & (df_clean.index <= end)
        ]

    else:
        if ratio is None:
            raise ValueError(
                "Either ratio or training dates must be provided."
            )

        if not 0 < ratio <= 1:
            raise ValueError("ratio must be between 0 and 1.")

        split_idx = int(len(df_clean) * ratio)
        train_df = df_clean.iloc[:split_idx]

    if train_df.empty:
        raise ValueError("Training subset is empty.")

    # --------------------------------------------------
    # Compute normalization factors and apply
    # --------------------------------------------------
    if method == "max":
        max_values = train_df.max()

        if (max_values == 0).any():
            raise ValueError(
                "At least one column has zero maximum value in training data."
            )

        df_norm = df_clean.divide(max_values, axis="columns")
        norm_params = {"max": max_values}

    elif method == "robust":
        median_values = train_df.median()
        q1 = train_df.quantile(0.25)
        q3 = train_df.quantile(0.75)
        iqr_values = q3 - q1

        if (iqr_values == 0).any():
            raise ValueError(
                "At least one column has a zero IQR in training data (division by zero)."
            )

        df_norm = df_clean.subtract(median_values, axis="columns").divide(
            iqr_values, axis="columns"
        )
        norm_params = {"median": median_values, "iqr": iqr_values}

    return df_norm, norm_params


def split_data(df, look_back=12, ratio=0.8, predict_n=5, Y_column=0):
    """
    Split the data into training and test sets
    Keras expects the input tensor to have a shape of (nb_samples, timesteps, features).
    :param df: Pandas dataframe with the data.
    :param look_back: Number of weeks to look back before predicting
    :param ratio: fraction of total samples to use for training
    :param predict_n: number of weeks to predict
    :param Y_column: Column to predict
    :return:
    """
    df = np.nan_to_num(df.values).astype("float64")
    # n_ts is the number of training samples also number of training sets
    # since windows have an overlap of n-1
    n_ts = df.shape[0] - look_back - predict_n + 1
    # data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    for i in range(n_ts):  # - predict_):
        #         print(i, df[i: look_back+i+predict_n,0])
        data[i, :, :] = df[i : look_back + i + predict_n, :]
    # train_size = int(n_ts * ratio)
    train_size = int(df.shape[0] * ratio) - look_back
    # print(train_size)

    # We are predicting only column 0
    X_train = data[:train_size, :look_back, :]
    Y_train = data[:train_size, look_back:, Y_column]
    X_test = data[train_size:, :look_back, :]
    Y_test = data[train_size:, look_back:, Y_column]

    return X_train, Y_train, X_test, Y_test


def _base_preprocess(
    df_data: Optional[pd.DataFrame],
    method: str,
    ini_train_date: Optional[str],
    end_train_date: Optional[str],
    end_date: Optional[str],
    target_col_name: str,
    use_log: bool,
    ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], int]:
    """
    Internal helper to clean, filter, log-transform, and normalize time-series data.

    This shared core utility unifies the common preprocessing steps required by both
    training and forecasting pipelines. It handles validation, date-based alignment,
    optional logarithmic scaling, and extracts the target-specific parameters
    necessary for eventual inverse scaling.

    Parameters
    ----------
    df_data : pd.DataFrame
        The source DataFrame containing historical time-series data indexed by date.
    method : str
        The normalization method to apply (e.g., "max" or "robust"). Passed down
        to `normalize_data`.
    ini_train_date : str or None
        The starting date boundary (inclusive) to filter the source dataset.
    end_train_date : str or None
        The boundary date marking the end of the training data. Used to isolate
        historical scaling factors. If None, `ratio` is utilized instead.
    end_date : str or None
        The final date boundary (inclusive) used to truncate the dataset.
    target_col_name : str
        The name of the target column to be forecasted.
    use_log : bool
        If True, applies a natural logarithmic transformation to the target column
        (handling zero values by shifting them to 1).
    ratio : float, optional
        The train/test split ratio used for normalization if `end_train_date`
        is not provided.

    Returns
    -------
    norm_df : pd.DataFrame
        The clean, filtered, and fully normalized DataFrame.
    target_params : dict of {str : float}
        Extracted normalization factors specific to the target column (e.g., max,
        median, or IQR) required to reverse the normalization later.
    target_col : int
        The integer column position index of `target_col_name` in the DataFrame.

    Raises
    ------
    ValueError
        - If `df_data` is None or resolves to empty.
        - If `target_col_name` is missing from the DataFrame columns.
        - If no rows remain after applying date boundaries.
        - If `end_train_date` evaluates to a date later than `end_date`.
    TypeError
        If the DataFrame's index cannot be coerced into a `pd.DatetimeIndex`.
    """

    if df_data is None:
        raise ValueError("df_data must be provided.")

    df = df_data.copy().dropna()
    if df.empty:
        raise ValueError("Input data is empty.")

    df.index = pd.to_datetime(df.index)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    if target_col_name not in df.columns:
        raise ValueError(f"Target column '{target_col_name}' not found.")

    target_col = cast(int, df.columns.get_loc(target_col_name))

    # Apply date filters
    if ini_train_date:
        df = df.loc[df.index >= pd.to_datetime(ini_train_date)]
    if end_date:
        df = df.loc[df.index <= pd.to_datetime(end_date)]
    if df.empty:
        raise ValueError("No data remaining after date filtering.")

    # Log transform
    if use_log:
        df.loc[df[target_col_name] == 0, target_col_name] = 1
        df[target_col_name] = np.log(df[target_col_name])

    # Normalization
    if end_train_date is None:
        norm_df, norm_params = normalize_data(
            df,
            method=method,
            ini_train_date=ini_train_date,
            end_train_date=None,
            ratio=ratio,
        )
    else:
        end_train_date_dt = pd.to_datetime(end_train_date)
        if end_date and end_train_date_dt > pd.to_datetime(end_date):
            raise ValueError("end_train_date must be earlier than end_date.")

        norm_df, norm_params = normalize_data(
            df,
            method=method,
            ratio=None,
            ini_train_date=ini_train_date,
            end_train_date=end_train_date_dt.strftime("%Y-%m-%d"),
        )

    target_params = {
        key: float(values[target_col_name])
        for key, values in norm_params.items()
    }

    return norm_df, target_params, target_col


def get_nn_data(
    df_data: pd.DataFrame,
    method: str = "max",
    ratio: float | None = 0.75,
    ini_train_date: Optional[str] = None,
    end_train_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_col_name: str = "casos",
    look_back: int = 4,
    predict_n: int = 4,
    use_log: bool = False,
) -> Tuple[
    pd.DataFrame,
    Dict[str, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Prepare normalized time-series data for neural network training and prediction.

    This function:
    1. Loads data (from DataFrame or CSV file).
    2. Filters by optional date boundaries.
    3. Normalizes features using training-only data.
    4. Splits the dataset into train/test sequences for supervised learning.
    5. Returns target parameters for inverse transformation and PyTorch tensors.

    Parameters
    ----------
    df_data : pd.DataFrame
        Input DataFrame indexed by date.
    method: str, default="max"
        Method of normalization to apply ("max" or "robust").
    ratio : float, default=0.75
        Fraction of the dataset used for training when `end_train_date` is not provided.
        Must be between 0 and 1.
    ini_train_date : str or None, optional
        Initial date (inclusive) to filter the dataset. Format: "YYYY-MM-DD".
    end_train_date : str or None, optional
        Last date (inclusive) used to compute normalization and training split.
        If provided, overrides `ratio`.
    end_date : str or None, optional
        Final date (inclusive) used to truncate the dataset.
    target_col_name : str, default="casos"
        Name of the target column used for prediction.
    look_back : int, default=4
        Number of past timesteps used as input features.
    predict_n : int, default=4
        Number of future timesteps to predict.
    use_log: bool: False
        If True, a logarithmic transformation is applied to the target column.

    Returns
    -------
    norm_df : pd.DataFrame
        Normalized dataset.
    target_params : Dict[str, float]
        Dictionary containing the normalization factors for the target column
        (e.g., {'max': value} or {'median': value, 'iqr': value}).
    X_train : torch.Tensor
        Training input sequences.
    Y_train : torch.Tensor
        Training target sequences.
    X_test : torch.Tensor
        Testing input sequences.
    Y_test : torch.Tensor
        Testing target sequences.

    Raises
    ------
    ValueError
        If `df_data` are None.
        If `ratio` is not between 0 and 1.
        If target column is missing.
    TypeError
        If index is not datetime-like.
    """

    if end_train_date is None:
        if ratio is None or not (0 < ratio <= 1):
            raise ValueError("ratio must be between 0 and 1.")

    # Chama o core compartilhado
    norm_df, target_params, target_col = _base_preprocess(
        df_data,
        method,
        ini_train_date,
        end_train_date,
        end_date,
        target_col_name,
        use_log,
        ratio,
    )

    # Divisão dos dados específica para Treino/Teste
    if end_train_date is None:
        X_train, Y_train, X_test, Y_test = split_data(
            norm_df,
            look_back=look_back,
            ratio=ratio,
            predict_n=predict_n,
            Y_column=target_col,
        )
    else:
        end_train_date_dt = pd.to_datetime(end_train_date)
        train_df = norm_df.loc[norm_df.index <= end_train_date_dt]

        X_train, Y_train, _, _ = split_data(
            train_df,
            look_back=look_back,
            ratio=1,
            predict_n=predict_n,
            Y_column=target_col,
        )
        X_test, Y_test, _, _ = split_data(
            norm_df.loc[
                norm_df.index
                > end_train_date_dt - timedelta(days=7 * look_back)
            ],
            look_back=look_back,
            ratio=1,
            predict_n=predict_n,
            Y_column=target_col,
        )

    return (
        norm_df,
        target_params,
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )


class LSTMModel(nn.Module):
    def __init__(
        self,
        hidden: int = 8,
        features: int = 100,
        predict_n: int = 4,
        look_back: int = 4,
        dropout: float = 0.2,
        num_layers: int = 3,
        output_activation: str = "relu",
        stateful: bool = False,
    ):
        super().__init__()

        self.hidden = hidden
        self.look_back = look_back
        self.stateful = stateful
        self.num_layers = num_layers

        self.lstms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):

            input_size = features if i == 0 else hidden

            self.lstms.append(
                nn.LSTM(
                    input_size=input_size, hidden_size=hidden, batch_first=True
                )
            )
            self.dropouts.append(nn.Dropout(dropout))

        self.fc = nn.Linear(hidden, predict_n)

        self.activation_fn = self._get_activation_fn(output_activation)

    def _get_activation_fn(
        self, act_str: str | None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Maps strings to PyTorch activation functions."""
        if act_str is None or act_str.lower() in ["linear", "none"]:
            return lambda x: x  # Sem ativação (saída linear pura)

        act_maps: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "relu": F.relu,
            "gelu": F.gelu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "elu": F.elu,
            "selu": torch.selu,
            "softplus": F.softplus,
        }

        chosen = act_str.lower()
        if chosen not in act_maps:
            raise ValueError(
                f"Activation '{act_str}' is not supported. Choose from: {list(act_maps.keys())} or 'linear'."
            )

        return act_maps[chosen]

    def forward(self, x, hidden_states=None):
        """
        x shape: (batch, look_back, features)
        """
        for i in range(self.num_layers):
            if i == 0:
                x, hidden_states = self.lstms[i](
                    x, None if not self.stateful else hidden_states
                )
            else:
                x, _ = self.lstms[i](x)

            if i < self.num_layers - 1:
                x = self.dropouts[i](x)

        x = x[:, -1, :]

        x = self.dropouts[-1](x)

        x = self.activation_fn(self.fc(x))

        return x


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# --------------------------------------------------
# Helper: compute loss for deterministic/prob models
# --------------------------------------------------
def compute_loss(output, target, criterion):

    # probabilistic model -> (mu, sigma)
    if isinstance(output, tuple):
        mu, sigma = output
        return criterion(mu, sigma, target)

    # deterministic model -> prediction
    return criterion(output, target)


# --------------------------------------------------
# TRAIN FUNCTION
# --------------------------------------------------
def train(
    model,
    X_train,
    Y_train,
    batch_size=1,
    epochs=10,
    patience=20,
    min_delta=0.0,
    verbose=0,
    criterion=None,
    lr=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Trains a PyTorch model using the Adam optimizer with early stopping.

    The function automatically splits the provided training data into training
    and validation sets (75/25 split), creates PyTorch DataLoaders, runs the
    training loop with gradient clipping, and monitors validation loss to
    prevent overfitting.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch neural network model to be trained.
    X_train : torch.Tensor
        The input features tensor for training.
    Y_train : torch.Tensor
        The target values tensor for training.
    batch_size : int, default=1
        Number of samples per gradient update.
    epochs : int, default=10
        Maximum number of training epochs to execute.
    patience : int, default=20
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float, default=0.0
        Minimum change in the validation loss to qualify as an improvement.
    verbose : int, default=0
        Verbosity mode. If > 0, prints epoch statistics and early stopping logs.
    criterion : callable, default=None
        The loss function to optimize (e.g., torch.nn.MSELoss()). Must be provided.
    lr : float, default=0.001
        Learning rate for the Adam optimizer.
    device : str, default="cuda" if torch.cuda.is_available() else "cpu"
        The device to run the training on ('cuda' or 'cpu').

    Returns
    -------
    model : torch.nn.Module
        The trained PyTorch model.
    history : dict
        A dictionary containing the loss history with keys:
        - "train_loss": List of average training losses per epoch.
        - "val_loss": List of average validation losses per epoch.

    Raises
    ------
    ValueError
        If `criterion` is not explicitly provided.
    """

    if criterion is None:
        raise ValueError("criterion must be provided")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------
    # LOSS HISTORY
    # -----------------------
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    # --------------------------------------------------
    # INNER TRAIN LOOP
    # --------------------------------------------------
    def run_training(train_loader, val_loader):

        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        for epoch in range(epochs):

            # ---------- TRAIN ----------
            model.train()
            train_loss = 0.0
            n_train = 0

            for X_batch, y_batch in train_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                output = model(X_batch)
                loss = compute_loss(output, y_batch, criterion)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                loss.backward()

                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                n_train += X_batch.size(0)

            train_loss /= max(n_train, 1)

            # ---------- VALIDATION ----------
            model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:

                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    output = model(X_batch)
                    loss = compute_loss(output, y_batch, criterion)

                    val_loss += loss.item()
                    n_val += 1

            val_loss /= max(n_val, 1)

            # ---------- SAVE HISTORY ----------
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f}"
                )

            early_stopping(val_loss)

            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    #

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        Y_train,
        test_size=0.25,
        random_state=7,
    )

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    run_training(train_loader, val_loader)

    return model, history


def enable_dropout(model):
    """Enable dropout layers during inference"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


class MSLELoss(nn.Module):
    r"""
    Mean Squared Logarithmic Error (MSLE) loss function.

    MSLE measures the risk corresponding to the expected value of the squared
    logarithmic error. It is particularly useful when targets have exponential
    growth or when you care more about relative percentage errors than absolute
    differences.

    The loss is computed as:
    $$L(y_{pred}, y_{true}) = \frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_{pred, i}) - \log(1 + y_{true, i}))^2$$

    Shape:
        - Input (pred): $(N, *)$ where $*$ means any number of additional dimensions.
        - Target (target): $(N, *)$ same shape as the input.
        - Output: scalar. If reduction is 'none', then same shape as input.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Safety alert: if the model predicts values less than -1, log generates NaN.
        # For safety, we clamp it.
        pred = torch.clamp(pred, min=0.0)

        log_pred = torch.log1p(pred)
        log_target = torch.log1p(target)

        return self.mse(log_pred, log_target)


def predict_mc_dropout(
    model,
    X,
    target_params: Dict[str, float],
    n_samples=100,
    device=None,
    use_log: bool = False,
):
    """
    Generates predictions using Monte Carlo (MC) Dropout to estimate model uncertainty.

    By keeping dropout layers active during inference (`model.eval()`), this function
    samples the model multiple times to create a distribution of predictions. It
    automatically handles dynamic denormalization and scales the predictions back
    to their original count scale if log transformation was used.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch neural network model. Must contain dropout layers.
    X : torch.Tensor
        The input features tensor to generate predictions for.
    target_params : dict of {str : float}
        Parameters used for target scaling. Must contain either:
        - {"max": float} for basic max scaling inversion.
        - {"median": float, "iqr": float} for RobustScaler scaling inversion.
    n_samples : int, default=100
        The number of forward passes (MC samples) to collect.
    device : torch.device or str, optional
        The device to perform inference on. If None, defaults to the device
        where the model parameters reside.
    use_log : bool, default=False
        If True, applies an exponential transformation (`torch.exp`) to reverse
        a prior logarithmic transformation.

    Returns
    -------
    preds : torch.Tensor
        A tensor of shape `(n_samples, batch_size, target_dimensions)` containing
        the denormalized, sampled predictions.

    Raises
    ------
    ValueError
        If `target_params` does not contain the required scaling keys
        ('max' or 'median' and 'iqr').
    """

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    enable_dropout(model)

    X = X.to(device)

    preds_list: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            y_hat = model(X)
            preds_list.append(y_hat.detach().cpu())

    preds = torch.stack(preds_list)  # (n_samples, batch, predict_n)

    # ------------------------------------------------------------------
    # 1. Dynamic Denormalization
    # ------------------------------------------------------------------
    if "max" in target_params:
        preds = preds * target_params["max"]

    elif "median" in target_params and "iqr" in target_params:
        # Inverso do Robust Scaling: (X_norm * IQR) + Mediana
        preds = (preds * target_params["iqr"]) + target_params["median"]

    else:
        raise ValueError(
            "The provided 'target_params' dictionary does not contain the expected keys "
            "('max' or 'median'/'iqr'). Please check the output of get_nn_data."
        )

    # ------------------------------------------------------------------
    # 2. Log Inversion (Returns predictions to the real count scale)
    # ------------------------------------------------------------------
    if use_log:

        preds = torch.exp(preds)

    return preds


def generate_forecast_df(
    model,
    X,
    target_params: Dict[str, float],
    predict_n: int,
    norm_df: Optional[pd.DataFrame] = None,
    end_date: Optional[str] = None,
    n_samples: int = 100,
    device: str = "cuda",
    use_log: bool = False,
) -> pd.DataFrame:
    """
    Generates a DataFrame with probabilistic forecasting predictions.

    This function unifies out-of-sample future forecasting (from a single cutoff date)
    and historical backtest/validation forecasting (matching multiple sequences
    back to their original reference dates).

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch neural network model.
    X : torch.Tensor
        The input tensor containing the sequences to predict.
        Shape: (num_sequences, sequence_length, num_features).
    target_params : dict of {str : float}
        Parameters used for dynamic target scaling inversion (e.g., max or median/iqr).
    predict_n : int
        The forecasting horizon (number of steps/weeks to predict).
    norm_df : pd.DataFrame, optional
        Reference DataFrame used to look up historical dates for multiple sequences.
        Required if `end_date` is None.
    end_date : str, optional
        A specific cutoff date for future forecasts. If provided, it will be
        applied to all sequences in X. Required if `norm_df` is None.
    n_samples : int, default=100
        Number of Monte Carlo forward passes to estimate uncertainty.
    device : str, default='cuda'
        The target device to execute inference on ('cuda' or 'cpu').
    use_log : bool, default=False
        If True, applies exponential transformation to reverse a log transformation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the structured forecasts with percentiles and timelines.
    """
    if norm_df is None and end_date is None:
        raise ValueError(
            "You must provide either 'norm_df' for historical matching or 'end_date' for a fixed cutoff."
        )

    # 1. Generate all Monte Carlo predictions at once
    preds = predict_mc_dropout(
        model,
        X,
        target_params=target_params,
        n_samples=n_samples,
        device=device,
        use_log=use_log,
    )

    X_numpy = X.detach().cpu().numpy()
    rows = []

    # 2. Iterate through each sequence in the batch
    for n in range(X_numpy.shape[0]):

        # Determine the reference date for this sequence
        if end_date is not None:
            dt = str(end_date)[:10]
        else:
            # Historical date matching logic
            if norm_df is not None:
                values: NDArray[np.float64] = norm_df.values
                mask = np.isclose(values, X_numpy[n, -1, :], atol=1e-4).all(
                    axis=1
                )
                if not any(mask):
                    raise ValueError(
                        f"Sequence index {n} could not be matched to any date in 'norm_df'."
                    )
                dt = str(norm_df.index[mask][0])[:10]
            else:
                raise ValueError("End date or norm_df must be provided")

        # 3. Build the dataframe for the current sequence
        df = pd.DataFrame(
            {
                "last_date": [dt] * predict_n,
                "date": get_next_n_weeks(dt, next_days=predict_n),
                "lower_95": np.percentile(preds[:, n, :], q=2.5, axis=0),
                "lower_90": np.percentile(preds[:, n, :], q=5, axis=0),
                "lower_80": np.percentile(preds[:, n, :], q=10, axis=0),
                "lower_50": np.percentile(preds[:, n, :], q=25, axis=0),
                "pred": np.percentile(preds[:, n, :], q=50, axis=0),
                "upper_50": np.percentile(preds[:, n, :], q=75, axis=0),
                "upper_80": np.percentile(preds[:, n, :], q=90, axis=0),
                "upper_90": np.percentile(preds[:, n, :], q=95, axis=0),
                "upper_95": np.percentile(preds[:, n, :], q=97.5, axis=0),
                "horizon": np.arange(1, predict_n + 1),
            }
        )

        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def split_data_for(df, look_back=12, predict_n=5):
    """
    Aggregate the predictors using only the latest available dates to forecast
    Pytorch expects the input tensor to have a shape of (nb_samples, timesteps, features).
    :param df: Pandas dataframe with the data.
    :param look_back: Number of weeks to look back before predicting
    :param predict_n: number of weeks to predict
    :return: X_for
    """

    s = get_next_n_weeks(ini_date=str(df.index[-1])[:10], next_days=predict_n)

    df = pd.concat([df, pd.DataFrame(index=s)])

    df = np.nan_to_num(df.values).astype("float64")

    n_ts = df.shape[0] - look_back - predict_n + 1
    # data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    for i in range(n_ts):  # - predict_):
        #         print(i, df[i: look_back+i+predict_n,0])
        data[i, :, :] = df[i : look_back + i + predict_n, :]

    X_for = data[
        -1:,
        :look_back,
    ]

    return X_for


def get_nn_data_for(
    df_data: Optional[pd.DataFrame] = None,
    method: str = "max",
    ini_train_date: Optional[str] = None,
    end_train_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_col_name: str = "casos",
    look_back: int = 4,
    predict_n: int = 4,
    use_log: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Prepare normalized time-series data for neural network inference/forecasting.

    Parameters
    ----------
    df_data : pd.DataFrame or None, default=None
        Input DataFrame indexed by date. If None, `filename` must be provided.
    method: str, default="max"
        Method of normalization to apply ("max" or "robust").
    ini_train_date : str or None, optional
        Initial date to filter the dataset.
    end_train_date : str or None, optional
        Last day used to compute the training normalization factors.
    end_date : str or None, optional
        Final date used to truncate the dataset.
    target_col_name : str, default="casos"
        Name of the target column.
    batch_size : int, default=1
        Batch size for the split data.
    look_back : int, default=4
        Number of past timesteps used as input features.
    predict_n : int, default=4
        Number of future timesteps to predict.

    use_log : bool, default=False
        If True, applies an exponential transformation (`torch.exp`) to reverse
        a prior logarithmic transformation.

    Returns
    -------
    X_tensor : torch.Tensor
        Input sequences for prediction.
    target_params : Dict[str, float]
        Dictionary containing the normalization factors for the target column.
    """

    if end_train_date is None:
        raise ValueError("end_train_date must be provided for forecasting.")

    # Chama o core compartilhado
    norm_df, target_params, _ = _base_preprocess(
        df_data,
        method,
        ini_train_date,
        end_train_date,
        end_date,
        target_col_name,
        use_log,
        ratio=None,
    )

    # Divisão dos dados específica para Forecast
    X_for = split_data_for(norm_df, look_back=look_back, predict_n=predict_n)

    return torch.tensor(X_for, dtype=torch.float32), target_params


def apply_forecast(
    model,
    df_data: pd.DataFrame,
    method: str = "max",  # Adicionado o parâmetro method aqui
    ini_train_date: Optional[str] = None,
    end_train_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_col_name: str = "casos",
    look_back: int = 4,
    predict_n: int = 4,
    n_samples: int = 100,
    device: str = "cuda",
    use_log: bool = True,
) -> pd.DataFrame:
    """
    Loads a trained model and generates a future forecast DataFrame from a specific cutoff date.

    This high-level function acts as an orchestrator: it extracts and formats the sequential
    features from the raw DataFrame (handling log transforms and scale parameter extraction),
    and then feeds them into the probabilistic forecasting pipeline using Monte Carlo Dropout.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch forecasting model.
    df_data : pd.DataFrame
        The historical source DataFrame containing time-series data.
    method : str, default="max"
        The normalization strategy to be utilized by `get_nn_data_for` (e.g., "max" or "robust").
    ini_train_date : str, optional
        The start date boundary of the training window, used to anchor scaling statistics.
    end_train_date : str, optional
        The end date boundary of the training window, used to anchor scaling statistics.
    end_date : str, optional
        The precise cutoff date from which the future forecast should start.
    target_col_name : str, default="casos"
        The name of the target column in `df_data` to be forecasted.
    look_back : int, default=4
        The number of historical lag steps (window size) to use as input features.
    predict_n : int, default=4
        The forecasting horizon (number of steps to predict into the future).
    n_samples : int, default=100
        The number of forward passes (MC samples) used to build the predictive distribution.
    device : str, optional
        The target device to execute inference on ('cuda' or 'cpu'). If None, defaults
        to the model's current device.
    use_log : bool, default=True
        If True, applies a logarithmic transformation during data preparation and
        flags the inference engine to reverse it upon completion.

    Returns
    -------
    pd.DataFrame
        A structured DataFrame containing the probabilistic predictions with
        percentile boundaries and forecasted calendar dates.
    """

    X_for, target_params = get_nn_data_for(
        df_data=df_data,
        method=method,
        ini_train_date=ini_train_date,
        end_train_date=end_train_date,
        end_date=end_date,
        target_col_name=target_col_name,
        look_back=look_back,
        predict_n=predict_n,
        use_log=use_log,
    )

    df_for = generate_forecast_df(
        model,
        X_for,
        target_params=target_params,
        end_date=end_date,
        predict_n=predict_n,
        n_samples=n_samples,
        device=device,
        use_log=use_log,
    )

    return df_for


class ForecastLSTM:
    """A wrapper class to handle the end-to-end lifecycle of an LSTM model
    for time-series forecasting, including data preprocessing, training,
    and evaluation.
    """

    def __init__(
        self,
        df_data: pd.DataFrame,
        columns: list | None = None,
        date_col: str = "date",
        target_col: str = "casos",
        hidden: int = 16,
        features: int = 4,
        predict_n: int = 4,
        look_back: int = 4,
        dropout: float = 0.2,
        num_layers: int = 3,
        output_activation: str = "linear",
    ):
        """Initializes the ForecastLSTMModel handler, sets up the dataframe,
        and builds the underlying LSTM network architecture.

        Args:
            df_data (pd.DataFrame): Input dataframe containing time-series data.
            columns (list, optional): List of columns to use. Defaults to None.
            date_col (str, optional): Name of the datetime column. Defaults to 'date'.
            target_col (str, optional): Target column to forecast. Defaults to 'casos'.
            hidden (int, optional): Number of hidden units in LSTM layers. Defaults to 16.
            features (int, optional): Number of input features. Defaults to 4.
            predict_n (int, optional): Number of steps ahead to predict. Defaults to 4.
            look_back (int, optional): Number of past steps to look back. Defaults to 4.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.2.
            num_layers (int, optional): Number of LSTM layers. Defaults to 3.
            output_activation (str, optional): Activation function for the output layer. Defaults to "linear".

        Raises:
            ValueError: If `target_col` is not found within the provided `columns`.
        """
        # Handling mutable default argument safely
        self.columns = columns if columns is not None else []
        self.target_col = target_col
        self.date_col = date_col
        self.look_back = look_back
        self.predict_n = predict_n

        # Validation check
        if self.target_col not in self.columns:
            raise ValueError(
                f"'{self.target_col}' must be one of the available columns: {self.columns}."
            )

        # Data preprocessing
        df_model = df_data[self.columns].set_index(self.date_col)
        df_model.index = pd.to_datetime(df_model.index)
        self.df_model = df_model

        # Base model initialization
        self.model = LSTMModel(
            hidden=hidden,
            features=features,
            predict_n=predict_n,
            look_back=look_back,
            dropout=dropout,
            num_layers=num_layers,
            output_activation=output_activation,
        )

    def train(
        self,
        ini_train_date: str = "2020-01-01",
        end_train_date: str = "2024-12-31",
        end_date: str = "2025-12-31",
        use_log: bool = True,
        method: str = "robust",
        batch_size: int = 2,
        epochs: int = 500,
        patience: int = 30,
        min_delta: float = 0.0,
        verbose: int = 1,
        criterion=MSLELoss(),
        lr: float = 0.001,
        device: str = "cuda",
    ):
        """Preprocesses dataset splits and trains the LSTM model.

        Args:
            ini_train_date (str, optional): Start date for training window. Defaults to '2020-01-01'.
            end_train_date (str, optional): End date for training window. Defaults to '2024-12-31'.
            end_date (str, optional): Total data limit boundary date. Defaults to '2025-12-31'.
            use_log (bool, optional): If True, applies log transformation to data. Defaults to True.
            method (str, optional): Normalization technique (e.g., 'robust', 'minmax'). Defaults to 'robust'.
            batch_size (int, optional): Mini-batch size for gradient descent. Defaults to 2.
            epochs (int, optional): Maximum training iterations. Defaults to 500.
            patience (int, optional): Early stopping patience window. Defaults to 30.
            min_delta (float, optional): Early stopping loss threshold delta. Defaults to 0.0.
            verbose (int, optional): Training log verbosity level. Defaults to 1.
            criterion (Loss, optional): PyTorch/Custom loss function. Defaults to MSLELoss().
            lr (float, optional): Learning rate for optimization. Defaults to 0.001.
            device (str, optional): Compute accelerator target ('cuda' or 'cpu'). Defaults to 'cuda'.

        Returns:
            tuple: (trained_model, training_history)
        """
        self.device = device
        self.ini_train_date = ini_train_date
        self.end_train_date = end_train_date
        self.end_date = end_date
        self.use_log = use_log
        self.method = method

        # Data preparation using state attributes
        (
            self.norm_df,
            self.target_params,
            self.X_train,
            self.Y_train,
            self.X_test,
            self.Y_test,
        ) = get_nn_data(
            df_data=self.df_model,
            method=method,
            ratio=None,
            ini_train_date=ini_train_date,
            end_train_date=end_train_date,
            end_date=end_date,
            target_col_name=self.target_col,
            look_back=self.look_back,
            predict_n=self.predict_n,
            use_log=use_log,
        )

        # Executing model training
        model, hist = train(
            self.model,
            self.X_train,
            self.Y_train,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            verbose=verbose,
            criterion=criterion,
            lr=lr,
            device=device,
        )

        self.model = model
        return model, hist

    def _generate_and_merge_prediction(
        self, X_data, n_samples: int = 100
    ) -> pd.DataFrame:
        """Internal helper method to calculate forecasts on a given subset
        and merge them back with actual target records.

        Args:
            X_data (np.array/torch.Tensor): Feature subset data to run inference on.
            n_samples (int, optional): Number of simulation paths/samples. Defaults to 100.

        Raises:
            RuntimeError: If called before running the `.train()` routine.

        Returns:
            pd.DataFrame: Merged historical actuals and generated predictions.
        """
        if self.X_train is None:
            raise RuntimeError(
                "The model must be trained using .train() before generating predictions."
            )

        df_preds = generate_forecast_df(
            self.model,
            X_data,
            norm_df=self.norm_df,
            target_params=self.target_params,
            predict_n=self.predict_n,
            n_samples=n_samples,
            device=self.device,
            use_log=self.use_log,
        )

        df_preds["date"] = pd.to_datetime(df_preds["date"])
        return df_preds.merge(
            self.df_model[self.target_col], left_on="date", right_index=True
        )

    def predict_in_sample(self, n_samples: int = 100) -> pd.DataFrame:
        """Generates backtesting forecast predictions using the training split features.

        Args:
            n_samples (int, optional): Number of simulation samples. Defaults to 100.

        Returns:
            pd.DataFrame: Dataframe containing matched training targets and predictions.
        """
        return self._generate_and_merge_prediction(
            self.X_train, n_samples=n_samples
        )

    def predict_out_of_sample(self, n_samples: int = 100) -> pd.DataFrame:
        """Generates validation forecast predictions using the test split features.

        Args:
            n_samples (int, optional): Number of simulation samples. Defaults to 100.

        Returns:
            pd.DataFrame: Dataframe containing matched evaluation targets and predictions.
        """
        return self._generate_and_merge_prediction(
            self.X_test, n_samples=n_samples
        )

    def forecast(self, end_date: str, n_samples: int = 100) -> pd.DataFrame:
        """Performs multi-step-ahead forecasting into the unseen future up to a specified end date.

        Args:
            end_date (str): Target final date limit for the prospective forecast.
            n_samples (int): Number of Monte Carlo or stochastic simulation samples.

        Raises:
            RuntimeError: If called before running the `.train()` routine.

        Returns:
            pd.DataFrame: Dataframe containing future timestamp dates and forecast values.
        """
        if self.method is None:
            raise RuntimeError(
                "The model must be trained before performing out-of-sample forecasts."
            )

        df_for = apply_forecast(
            self.model,
            df_data=self.df_model,
            method=self.method,
            ini_train_date=self.ini_train_date,
            end_train_date=self.end_train_date,
            end_date=end_date,
            target_col_name=self.target_col,
            look_back=self.look_back,
            predict_n=self.predict_n,
            n_samples=n_samples,
            device=self.device,
            use_log=self.use_log,
        )
        return df_for
