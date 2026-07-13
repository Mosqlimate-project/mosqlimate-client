import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Definir a cor das bordas (spines) como cinza
mpl.rcParams["axes.edgecolor"] = "gray"
# Definir a cor das linhas dos ticks maiores e menores como cinza
mpl.rcParams["xtick.color"] = "gray"
mpl.rcParams["ytick.color"] = "gray"
mpl.rcParams["xtick.labelcolor"] = "black"
mpl.rcParams["ytick.labelcolor"] = "black"


def plot_forecasts(
    df_preds: pd.DataFrame,
    data: pd.DataFrame = None,
    last_obs: int | None = None,
    conf_levels=[0.90],
    target_col: str = "data",
    model_col: str | None = None,
    date_col: str = "date",
    pred_col: str = "pred",
    color_palette: str = "Set2",
    linestyle: str = "-",
    fill_alpha: float = 0.1,
    figsize: tuple = (10, 5),
    title: str | None = None,
    label: str | None = None,
    ax=None,
    date_format: str | None = None,
    connect_history: bool = False,
    default_color: str = "tab:orange",
) -> tuple:
    """Plot observed data and time-series model forecasts..

    Parameters
    ----------
    df_preds : pd.DataFrame
        DataFrame containing the forecast results. Must include at least the
        columns specified in `date_col` and `pred_col`, as well as interval
        columns (e.g., 'lower_95', 'upper_95').
    data : pd.DataFrame, optional
        DataFrame containing the observed historical data. Can use dates in the
        Index or in a specific column. If None, the function will attempt to
        fetch observed data directly from `df_preds[target_col]`.
    last_obs : int, optional
        Number of recent historical observations (the last N rows) to display
        on the plot to avoid visual clutter. If None, the full history is shown.
    conf_levels : list or float, default=[0.90]
        List or single value of decimal confidence levels (e.g., 0.95 or
        [0.50, 0.80, 0.95]). The function automatically converts these into
        integer suffixes (e.g., 'lower_95').
    target_col : str, default='data'
        Name of the column storing the actual observed numerical values.
    model_col : str, optional
        Name of the column in `df_preds` that identifies different models (e.g., 'model_id').
        If None, the function treats all rows as a single model forecast.
    date_col : str, default='date'
        Name of the datetime column used in the `df_preds` DataFrame.
    pred_col : str, default='pred'
        Name of the column storing the point forecast estimation (mean/median).
    color_palette : str, default='Set2'
        Name of the Seaborn/Matplotlib color palette used when multiple models
        are present in `model_col`.
    linestyle : str, default='-'
        Line style for the forecast curve (e.g., '-', '--', ':').
    fill_alpha : float, default=0.1
        Base opacity (transparency) for the shaded confidence interval areas.
        In Fan Charts, opacities accumulate across layers.
    figsize : tuple, default=(10, 5)
        Dimensions of the figure (width, height) in inches, if a new `ax` is created.
    title : str, optional
        Custom title for the plot. If None, an automatic title based on the
        confidence intervals will be generated.
    ax : matplotlib.axes.Axes, optional
        An existing Matplotlib Axes object. If provided, the plot will be drawn
        directly onto it instead of creating a new figure.
    date_format : str, optional
        Date formatting string for the X-axis (e.g., "%b\n%y" or "%b-%d\n%Y").
    connect_history : bool, default=False
        If True, draws a black dashed line connecting the last point of the
        observed history to the first point of the predicted series.
    default_color : str, default='tab:orange'
        Color used for the forecast line if `model_col` is None and there are
        no multiple models.

    Returns
    -------
    tuple
        A tuple containing (fig, ax), where `fig` is the matplotlib.figure.Figure
        object and `ax` is the modified matplotlib.axes.Axes object.

    Raises
    ------
    ValueError
        If `df_preds` is passed as None.
    """
    if df_preds is None:
        raise ValueError("df_preds cannot be None")

    if isinstance(conf_levels, (int, float)):
        conf_levels = [conf_levels]

    conf_ints = [
        int(round(cl * 100)) if cl < 1.0 else int(cl) for cl in conf_levels
    ]
    conf_ints.sort(reverse=True)

    # 2. Initialize Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    last_hist_x, last_hist_y = None, None

    # 3. Plot History (Observed Data)
    if data is not None:
        if last_obs is not None:
            data = data.tail(last_obs)

        if date_col in data.columns:
            hist_x = data[date_col]
        else:
            hist_x = data.index

        hist_y = data[target_col]
        ax.plot(hist_x, hist_y, color="black", linewidth=2, label="Data")

        if len(data) > 0:
            last_hist_x = (
                hist_x.iloc[-1]
                if hasattr(hist_x, "iloc")
                else hist_x.values[-1]
            )
            last_hist_y = (
                hist_y.iloc[-1]
                if hasattr(hist_y, "iloc")
                else hist_y.values[-1]
            )

    elif target_col in df_preds.columns:
        obs_df = (
            df_preds[[date_col, target_col]]
            .drop_duplicates()
            .sort_values(date_col)
        )
        if last_obs is not None:
            obs_df = obs_df.tail(last_obs)

        ax.plot(
            obs_df[date_col],
            obs_df[target_col],
            color="black",
            marker=".",
            label="Data",
        )

        if len(obs_df) > 0:
            last_hist_x = obs_df[date_col].iloc[-1]
            last_hist_y = obs_df[target_col].iloc[-1]

    # 4. Identify Multiple Models / Colors
    if model_col in df_preds.columns and model_col is not None:
        models = sorted(df_preds[model_col].unique())
        colors = sns.color_palette(color_palette, n_colors=len(models))
        color_map = dict(zip(models, colors))
    else:
        models = ["Forecast"]
        color_map = {
            "Forecast": (
                sns.color_palette(color_palette, n_colors=1)[0]
                if color_palette and model_col
                else default_color
            )
        }

    # 5. Plot Forecasts and Shaded Intervals
    for model in models:
        if model_col in df_preds.columns and model_col is not None:
            preds_ = df_preds.loc[df_preds[model_col] == model].sort_values(
                date_col
            )
            model_label = str(model)
        else:
            preds_ = df_preds.sort_values(date_col)
            model_label = label if label else "Forecast"

        color = color_map[model]

        ax.plot(
            preds_[date_col],
            preds_[pred_col],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=model_label,
        )

        for ci in conf_ints:
            lower_col = f"lower_{ci}"
            upper_col = f"upper_{ci}"

            if lower_col in preds_.columns and upper_col in preds_.columns:
                ax.fill_between(
                    preds_[date_col],
                    preds_[lower_col],
                    preds_[upper_col],
                    color=color,
                    alpha=fill_alpha,
                    label=(
                        f"{ci}% CI"
                        if len(conf_ints) == 1 and not model_col
                        else None
                    ),
                )

        # 6. Dashed Connection Line
        if connect_history and last_hist_x is not None and len(preds_) > 0:
            first_pred_x = preds_[date_col].iloc[0]
            first_pred_y = preds_[pred_col].iloc[0]
            ax.plot(
                [last_hist_x, first_pred_x],
                [last_hist_y, first_pred_y],
                ls="--",
                color="black",
            )

    # Aesthetics
    ax.set_xlabel("Date")
    ax.set_ylabel("New cases")

    if title is not None:
        ax.set_title(title)
    elif len(conf_ints) == 1:
        ax.set_title(f"Forecasts with {conf_ints[0]}% prediction interval")
    else:
        ax.set_title("Forecasts with Multiple Prediction Intervals")

    if date_format is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    return fig, ax


def plot_single_forecast(
    df_for: pd.DataFrame,
    df_train: pd.DataFrame,
    last_obs: int = 20,
    target_col="data",
    date_col="date",
    conf_levels=[0.5, 0.90, 0.95],
    title="Forecast ARIMA",
    label="Arima",
    xlabel="Date",
    ylabel="New Cases",
    figsize=(7, 4),
) -> tuple:
    """Generate a classic red Fan Chart for ARIMA models.

    A simplified wrapper around `plot_forecasts` configured to display multiple
    stacked confidence intervals in a gradient, history truncation with dates
    located in the index, and a dashed connection line linking history to the forecast.

    Parameters
    ----------
    df_for : pd.DataFrame
        DataFrame containing the ARIMA prospective projections. Must include
        interval columns in the format `lower_XX` and `upper_XX` matching the
        calculated alphas.
    df_train : pd.DataFrame
        DataFrame containing the historical training data. The date series must
        be set as the Index, and the actual values should be in a column named 'data'.
    last_obs : int, default=20
        Number of historical records from the training data to display on the plot.

    Returns
    -------
    tuple
        (fig, ax) Modified Matplotlib figure and axes objects.
    """

    if date_col in df_train.columns:

        df_train = df_train.loc[
            df_train[date_col] < pd.to_datetime(df_for[date_col].min())
        ]
    else:
        df_train = df_train.loc[
            df_train.index < pd.to_datetime(df_for[date_col].min())
        ]

    fig, ax = plot_forecasts(
        df_preds=df_for,
        data=df_train,
        last_obs=last_obs,
        conf_levels=conf_levels,
        target_col=target_col,
        date_col=date_col,
        label=label,
        default_color="tab:red",
        connect_history=True,
        date_format="%b-%d\n%Y",
        title=title,
        figsize=figsize,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_model_comparison(
    df_preds: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "data",
    label: str = "Arima",
    conf_levels=[0.5, 0.90, 0.95],
    title: str = "In sample predictions",
    xlabel: str = "Date",
    ylabel: str = "New cases",
    figsize=(7, 4),
) -> tuple:
    """Generate a comparative plot separating curves for multiple ML models.

    A simplified wrapper around `plot_forecasts` structured to separate and
    color forecasts dynamically using a model identifier ID column and applying
    a categorical palette.

    Parameters
    ----------
    df_multi_preds : pd.DataFrame
        Long-format DataFrame containing stacked predictions.
    date_col : str, default='date'
        Name of the datetime column used in the `df_preds` DataFrame.
    target_col : str, default='data'
        Name of the column storing the actual observed numerical values.
    conf_levels : list or float, default=[0.90]
        List or single value of decimal confidence levels (e.g., 0.95 or
        [0.50, 0.80, 0.95]). The function automatically converts these into
        integer suffixes (e.g., 'lower_95').

    Returns
    -------
    tuple
        (fig, ax) Modified Matplotlib figure and axes objects.
    """

    fig, ax = plot_forecasts(
        df_preds=df_preds,
        conf_levels=conf_levels,
        target_col=target_col,
        date_col=date_col,
        color_palette="Set2",
        figsize=figsize,
        label=label,
        title=title,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def plot_training_model_loss(hist, figsize=(8, 4)):

    _, ax = plt.subplots(figsize=figsize)

    ax.plot(hist["train_loss"], label="Training")
    ax.plot(hist["val_loss"], label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")

    ax.legend()
    ax.grid(True)
    plt.show()

    return _, ax
