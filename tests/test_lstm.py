"""Lightweight contract tests for the LSTM forecasting utilities.

Tests use small CPU tensors and monkeypatched forecasting functions so they
never require GPU access or long neural-network training runs.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from mosqlient.forecast import lstm as lstm_module
from mosqlient.forecast.lstm import (
    EarlyStopping,
    ForecastLSTM,
    LSTMModel,
    MSLELoss,
    _base_preprocess,
    apply_forecast,
    compute_loss,
    enable_dropout,
    generate_forecast_df,
    get_next_n_weeks,
    get_nn_data,
    get_nn_data_for,
    normalize_data,
    predict_mc_dropout,
    split_data,
    split_data_for,
    train,
)

QUANTILE_COLUMNS = [
    "lower_95", "lower_90", "lower_80", "lower_50", "pred",
    "upper_50", "upper_80", "upper_90", "upper_95",
]


@pytest.fixture
def sample_forecast_df():
    dates = pd.date_range("2020-01-05", periods=20, freq="W-SUN")
    return pd.DataFrame({
        "date": dates,
        "casos": np.arange(1, 21, dtype=float),
        "temp": np.linspace(20, 25, 20),
    })


@pytest.fixture
def train_forecast_df():
    dates = pd.date_range("2020-01-05", periods=40, freq="W-SUN")
    return pd.DataFrame({
        "date": dates,
        "casos": np.arange(1, 41, dtype=float),
        "temp": np.linspace(20, 25, 40),
    })


def make_model(df, **kwargs):
    """Include date in `columns` because the wrapper indexes it afterwards."""
    return ForecastLSTM(
        df,
        columns=["date", "casos", "temp"],
        date_col="date",
        target_col="casos",
        features=2,
        hidden=4,
        num_layers=1,
        look_back=4,
        predict_n=2,
        **kwargs,
    )


def _dated(df):
    return df.set_index("date")


def _fake_forecast_dataframe(X, predict_n, *, last_date="2020-02-02"):
    """A deterministic forecast in the real function's output schema."""
    n = len(X)
    output = pd.DataFrame({
        "last_date": [last_date] * (n * predict_n),
        "date": pd.date_range("2020-02-09", periods=predict_n, freq="W-SUN").tolist() * n,
        "horizon": list(range(1, predict_n + 1)) * n,
    })
    for i, col in enumerate(QUANTILE_COLUMNS):
        output[col] = float(i + 1)
    return output


def test_lstm_contract(sample_forecast_df):
    model = make_model(sample_forecast_df)
    assert model.target_col == "casos"
    assert model.look_back == 4
    assert model.predict_n == 2
    assert model.df_model.index.name == "date"
    assert list(model.df_model.columns) == ["casos", "temp"]
    assert isinstance(model.model, LSTMModel)


def test_invalid_target_is_rejected(sample_forecast_df):
    with pytest.raises(ValueError, match="must be one of the available columns"):
        ForecastLSTM(sample_forecast_df, columns=["date", "temp"])


def test_constructor_currently_requires_date_in_columns(sample_forecast_df):
    # Documents the present implementation: df_data[self.columns].set_index(date_col).
    with pytest.raises(KeyError):
        ForecastLSTM(sample_forecast_df, columns=["casos", "temp"])


def test_get_next_n_weeks():
    assert get_next_n_weeks("2020-01-05", 3) == [
        date(2020, 1, 12), date(2020, 1, 19), date(2020, 1, 26)
    ]
    assert get_next_n_weeks("2020-01-05", 0) == []


def test_normalize_max_uses_training_only(sample_forecast_df):
    df = _dated(sample_forecast_df)[["casos", "temp"]]
    scaled, params = normalize_data(df, method="max", ratio=0.5)
    assert params["max"]["casos"] == 10
    assert scaled.iloc[9]["casos"] == pytest.approx(1)
    assert scaled.iloc[-1]["casos"] == pytest.approx(2)


def test_normalize_robust_uses_training_subset(sample_forecast_df):
    df = _dated(sample_forecast_df)[["casos", "temp"]]
    scaled, params = normalize_data(df, method="robust", ratio=0.5)
    train_values = df.iloc[:10]
    expected_median = train_values["casos"].median()
    expected_iqr = train_values["casos"].quantile(0.75) - train_values["casos"].quantile(0.25)
    assert params["median"]["casos"] == expected_median
    assert params["iqr"]["casos"] == expected_iqr
    assert scaled.iloc[0]["casos"] == pytest.approx((1 - expected_median) / expected_iqr)


def test_normalize_dates_and_missing_values(sample_forecast_df):
    df = _dated(sample_forecast_df)[["casos", "temp"]].copy()
    df.iloc[0, 0] = np.nan
    scaled, params = normalize_data(
        df, method="max", ini_train_date="2020-01-05", end_train_date="2020-02-02"
    )
    assert params["max"]["casos"] == 5
    assert scaled.iloc[0]["casos"] == 0


@pytest.mark.parametrize("ratio", [0, -0.2, 1.1, None])
def test_normalize_invalid_ratio(sample_forecast_df, ratio):
    with pytest.raises(ValueError, match="ratio"):
        normalize_data(_dated(sample_forecast_df)[["casos"]], ratio=ratio)


def test_normalize_invalid_method(sample_forecast_df):
    with pytest.raises(ValueError, match="method"):
        normalize_data(_dated(sample_forecast_df)[["casos"]], method="minmax")


def test_normalize_invalid_dates(sample_forecast_df):
    df = sample_forecast_df[["casos"]]
    with pytest.raises(TypeError, match="DatetimeIndex"):
        normalize_data(df, ini_train_date="2020-01-01", end_train_date="2020-03-01")
    with pytest.raises(ValueError, match="earlier"):
        normalize_data(_dated(sample_forecast_df)[["casos"]], ini_train_date="2020-03-01", end_train_date="2020-01-01")
    with pytest.raises(ValueError, match="empty"):
        normalize_data(_dated(sample_forecast_df)[["casos"]], ini_train_date="1990-01-01", end_train_date="1990-03-01")


def test_normalize_rejects_zero_max_and_iqr():
    df = pd.DataFrame({"casos": [0.0] * 8})
    with pytest.raises(ValueError, match="zero maximum"):
        normalize_data(df, method="max")
    with pytest.raises(ValueError, match="zero IQR"):
        normalize_data(df, method="robust")


def test_split_data_uses_past_windows_and_future_targets(sample_forecast_df):
    df = sample_forecast_df[["casos", "temp"]]
    x_train, y_train, x_test, y_test = split_data(
        df, look_back=4, predict_n=2, ratio=0.7, Y_column=0
    )
    assert x_train.shape[1:] == (4, 2)
    assert y_train.shape[1:] == (2,)
    np.testing.assert_allclose(x_train[0, :, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(y_train[0], [5, 6])
    assert len(x_train) + len(x_test) == len(df) - 4 - 2 + 1
    assert len(y_test) == len(x_test)


def test_base_preprocess_log_scaling_and_target_column(sample_forecast_df):
    df = _dated(sample_forecast_df)[["temp", "casos"]]
    norm, params, col = _base_preprocess(
        df, "max", "2020-01-05", "2020-03-01", "2020-04-05", "casos", True
    )
    assert col == 1
    assert norm.index.max() == pd.Timestamp("2020-04-05")
    assert params["max"] == pytest.approx(np.log(9))
    assert norm.iloc[0]["casos"] == pytest.approx(0)


def test_base_preprocess_invalid_data(sample_forecast_df):
    df = _dated(sample_forecast_df)
    with pytest.raises(ValueError, match="provided"):
        _base_preprocess(None, "max", None, None, None, "casos", False, 0.5)
    with pytest.raises(ValueError, match="not found"):
        _base_preprocess(df, "max", None, None, None, "missing", False, 0.5)
    with pytest.raises(ValueError, match="No data remaining"):
        _base_preprocess(df, "max", "2030-01-01", None, None, "casos", False, 0.5)
    with pytest.raises(ValueError, match="earlier"):
        _base_preprocess(df, "max", "2020-01-05", "2020-05-01", "2020-03-01", "casos", False)


def test_get_nn_data_shapes_and_types(train_forecast_df):
    df = _dated(train_forecast_df)
    norm, params, x_train, y_train, x_test, y_test = get_nn_data(
        df, ratio=0.75, look_back=4, predict_n=2, use_log=False
    )
    assert len(norm) == 40
    assert "max" in params
    for tensor in (x_train, y_train, x_test, y_test):
        assert tensor.dtype == torch.float32
    assert x_train.shape[1:] == (4, 2)
    assert y_train.shape[1:] == (2,)
    assert x_test.shape[1:] == (4, 2)


def test_get_nn_data_with_date_split(train_forecast_df):
    norm, params, x_train, y_train, x_test, y_test = get_nn_data(
        _dated(train_forecast_df), method="max", ratio=None,
        ini_train_date="2020-01-05", end_train_date="2020-06-28",
        end_date="2020-10-04", look_back=4, predict_n=2,
    )
    assert "max" in params
    assert x_train.shape[0] > 0
    assert x_test.shape[0] > 0
    assert y_train.shape[-1] == 2
    assert y_test.shape[-1] == 2


def test_get_nn_data_rejects_invalid_ratio(train_forecast_df):
    with pytest.raises(ValueError, match="ratio"):
        get_nn_data(_dated(train_forecast_df), ratio=0)


def test_split_data_for_selects_last_context(train_forecast_df):
    df = _dated(train_forecast_df)[["casos", "temp"]]
    result = split_data_for(df, look_back=4, predict_n=3)
    assert result.shape == (1, 4, 2)
    np.testing.assert_allclose(result[0, :, 0], [37, 38, 39, 40])


def test_get_nn_data_for_context_and_params(train_forecast_df):
    x, params = get_nn_data_for(
        _dated(train_forecast_df), ini_train_date="2020-01-05",
        end_train_date="2020-06-28", end_date="2020-10-04",
        look_back=4, predict_n=2,
        use_log=False,
    )
    assert x.shape == (1, 4, 2)
    assert x.dtype == torch.float32
    assert "max" in params


def test_get_nn_data_for_requires_train_end_date(train_forecast_df):
    with pytest.raises(ValueError, match="end_train_date"):
        get_nn_data_for(_dated(train_forecast_df))


@pytest.mark.parametrize("activation", ["relu", "gelu", "sigmoid", "tanh", "elu", "selu", "softplus", "linear", "none", None])
def test_lstm_model_activations(activation):
    model = LSTMModel(features=2, hidden=4, predict_n=3, num_layers=2,
                      dropout=0, output_activation=activation)
    output = model(torch.randn(5, 4, 2))
    assert output.shape == (5, 3)
    assert torch.isfinite(output).all()
    if activation in ("relu", "sigmoid", "softplus"):
        assert torch.all(output >= 0)


def test_lstm_model_rejects_invalid_activation():
    with pytest.raises(ValueError, match="not supported"):
        LSTMModel(output_activation="invalid")


def test_early_stopping_patience_and_improvement():
    callback = EarlyStopping(patience=2, min_delta=0.1)
    callback(1.0)
    assert callback.best_loss == 1.0
    callback(0.95)
    assert callback.counter == 1
    callback(0.8)
    assert callback.counter == 0
    callback(0.85)
    callback(0.9)
    assert callback.early_stop


def test_compute_loss_supports_deterministic_and_probabilistic_models():
    target = torch.tensor([[1.0]])
    output = torch.tensor([[2.0]])
    assert compute_loss(output, target, nn.MSELoss()).item() == pytest.approx(1)
    criterion = lambda mu, sigma, y: ((mu - y) ** 2 + sigma).mean()
    assert compute_loss((output, torch.tensor([[0.5]])), target, criterion).item() == pytest.approx(1.5)


def test_msle_loss_clamps_negative_predictions():
    criterion = MSLELoss()
    pred = torch.tensor([[-3.0, 3.0]])
    target = torch.tensor([[0.0, 3.0]])
    assert criterion(pred, target).item() == pytest.approx(0)


def test_enable_dropout_only_changes_dropout_modules():
    model = LSTMModel(features=2, hidden=4, num_layers=2)
    model.eval()
    enable_dropout(model)
    assert all(layer.training for layer in model.dropouts)
    assert all(not layer.training for layer in model.lstms)


def test_train_requires_criterion():
    model = LSTMModel(features=2, hidden=4, num_layers=1, predict_n=2)
    with pytest.raises(ValueError, match="criterion must be provided"):
        train(model, torch.zeros(8, 4, 2), torch.zeros(8, 2), device="cpu")


def test_train_returns_history_on_cpu():
    torch.manual_seed(7)
    model = LSTMModel(features=2, hidden=4, num_layers=1, predict_n=2,
                      dropout=0, output_activation="linear")
    x = torch.randn(12, 4, 2)
    y = torch.randn(12, 2)
    fitted, history = train(model, x, y, epochs=2, batch_size=4,
                            criterion=nn.MSELoss(), device="cpu")
    assert fitted is model
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert np.isfinite(history["train_loss"]).all()


def test_predict_mc_dropout_max_and_robust_scaling():
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            return self.weight.expand(x.size(0), 2)

    model = ConstantModel()
    x = torch.ones(3, 4, 2)
    pred = predict_mc_dropout(model, x, {"max": 5.0}, n_samples=3, device="cpu")
    assert pred.shape == (3, 3, 2)
    torch.testing.assert_close(pred, torch.full((3, 3, 2), 10.0))

    pred = predict_mc_dropout(model, x, {"median": 2.0, "iqr": 3.0},
                               n_samples=2, device="cpu")
    torch.testing.assert_close(pred, torch.full((2, 3, 2), 8.0))

    pred = predict_mc_dropout(model, x, {"max": 1.0}, n_samples=2,
                               device="cpu", use_log=True)
    torch.testing.assert_close(pred, torch.full((2, 3, 2), np.exp(2.0)))


def test_predict_mc_dropout_rejects_missing_scaling():
    model = LSTMModel(features=2, hidden=4, num_layers=1, predict_n=2)
    with pytest.raises(ValueError, match="target_params"):
        predict_mc_dropout(model, torch.zeros(1, 4, 2), {}, n_samples=2, device="cpu")


def test_generate_forecast_df_future_dates_and_quantiles(monkeypatch):
    predictions = torch.tensor([
        [[1.0, 2.0]], [[2.0, 3.0]], [[3.0, 4.0]], [[4.0, 5.0]],
    ])
    monkeypatch.setattr(lstm_module, "predict_mc_dropout", lambda *a, **kw: predictions)
    df = generate_forecast_df(
        model=object(), X=torch.zeros(1, 4, 2), target_params={"max": 1.0},
        predict_n=2, end_date="2020-01-05", device="cpu", n_samples=4,
    )
    assert len(df) == 2
    assert df["date"].tolist() == [date(2020, 1, 12), date(2020, 1, 19)]
    assert df["horizon"].tolist() == [1, 2]
    assert df["pred"].tolist() == pytest.approx([2.5, 3.5])
    assert (np.diff(df[QUANTILE_COLUMNS].to_numpy(), axis=1) >= 0).all()


def test_generate_forecast_df_matches_historical_reference(monkeypatch, sample_forecast_df):
    reference = _dated(sample_forecast_df)[["casos", "temp"]]
    x = torch.tensor(reference.iloc[2:6].values[None, :, :], dtype=torch.float32)
    monkeypatch.setattr(lstm_module, "predict_mc_dropout", lambda *a, **kw: torch.ones(3, 1, 2))
    df = generate_forecast_df(
        model=object(), X=x, target_params={"max": 1.0}, predict_n=2,
        norm_df=reference, device="cpu",
    )
    assert df["last_date"].tolist() == ["2020-02-09", "2020-02-09"]


def test_generate_forecast_df_rejects_missing_reference():
    with pytest.raises(ValueError, match="either 'norm_df'"):
        generate_forecast_df(None, torch.zeros(1, 4, 2), {"max": 1}, 2, device="cpu")


def test_generate_forecast_df_rejects_unmatched_sequence(monkeypatch, sample_forecast_df):
    monkeypatch.setattr(lstm_module, "predict_mc_dropout", lambda *a, **kw: torch.ones(3, 1, 2))
    with pytest.raises(ValueError, match="could not be matched"):
        generate_forecast_df(
            model=object(), X=torch.full((1, 4, 2), 999),
            target_params={"max": 1}, predict_n=2,
            norm_df=_dated(sample_forecast_df)[["casos", "temp"]], device="cpu",
        )


def test_apply_forecast_forwards_arguments(monkeypatch, train_forecast_df):
    observed = {}

    def fake_get_nn_data_for(**kwargs):
        observed["preprocess"] = kwargs
        return torch.zeros(1, 4, 2), {"max": 10.0}

    def fake_generate_forecast_df(model, X, **kwargs):
        observed["forecast"] = kwargs
        return _fake_forecast_dataframe(X, kwargs["predict_n"])

    monkeypatch.setattr(lstm_module, "get_nn_data_for", fake_get_nn_data_for)
    monkeypatch.setattr(lstm_module, "generate_forecast_df", fake_generate_forecast_df)
    result = apply_forecast(
        model=object(), df_data=_dated(train_forecast_df),
        end_train_date="2020-06-28", end_date="2020-10-04",
        predict_n=2, look_back=4, n_samples=5, device="cpu", use_log=False,
    )
    assert len(result) == 2
    assert observed["preprocess"]["look_back"] == 4
    assert observed["forecast"]["n_samples"] == 5
    assert observed["forecast"]["target_params"] == {"max": 10.0}


def test_wrapper_train_and_forecast_contract(monkeypatch, train_forecast_df):
    model = make_model(train_forecast_df)

    def fake_train(network, x, y, **kwargs):
        assert x.shape[1:] == (4, 2)
        assert y.shape[1:] == (2,)
        return network, {"train_loss": [0.1], "val_loss": [0.2]}

    monkeypatch.setattr(lstm_module, "train", fake_train)
    fitted, history = model.train(
        ini_train_date="2020-01-05", end_train_date="2020-06-28",
        end_date="2020-10-04", method="max", use_log=False,
        epochs=1, device="cpu", verbose=0,
    )
    assert fitted is model.model
    assert history["train_loss"] == [0.1]
    assert model.X_train.shape[0] > 0
    assert model.X_test.shape[0] > 0

    def fake_generate(network, X, **kwargs):
        # For historical forecasts, choose a last_date whose future dates
        # overlap with the observed values for the merge.
        return _fake_forecast_dataframe(X, kwargs["predict_n"])

    monkeypatch.setattr(lstm_module, "generate_forecast_df", fake_generate)
    in_sample = model.predict_in_sample(n_samples=3)
    out_sample = model.predict_out_of_sample(n_samples=3)
    for df in (in_sample, out_sample):
        assert not df.empty
        assert set(QUANTILE_COLUMNS + ["date", "casos"]).issubset(df.columns)

    monkeypatch.setattr(
        lstm_module, "apply_forecast",
        lambda *a, **kw: _fake_forecast_dataframe(torch.zeros(1, 4, 2), kw["predict_n"]),
    )
    future = model.forecast("2020-10-04", n_samples=3)
    assert len(future) == 2
    assert (future["lower_95"] <= future["pred"]).all()
    assert (future["pred"] <= future["upper_95"]).all()


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="ForecastLSTM does not initialize X_train before train()")
def test_predict_before_training_should_raise_runtime_error(sample_forecast_df):
    model = make_model(sample_forecast_df)
    with pytest.raises(RuntimeError, match="trained"):
        model.predict_in_sample()


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="ForecastLSTM does not initialize method before train()")
def test_forecast_before_training_should_raise_runtime_error(sample_forecast_df):
    model = make_model(sample_forecast_df)
    with pytest.raises(RuntimeError, match="trained"):
        model.forecast("2020-05-31")
