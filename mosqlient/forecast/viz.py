import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# Definir a cor das bordas (spines) como cinza
mpl.rcParams["axes.edgecolor"] = "gray"
# Definir a cor das linhas dos ticks maiores e menores como cinza
mpl.rcParams["xtick.color"] = "gray"
mpl.rcParams["ytick.color"] = "gray"
mpl.rcParams["xtick.labelcolor"] = "black"
mpl.rcParams["ytick.labelcolor"] = "black"


def plot_preds(
    data=None,
    df_preds=None,
    conf_level=0.90,
    data_col="casprov",
    model_col="model_id",
    date_col="date",
    pred_col="pred",
    color_palette="Set2",
    linestyle="-",
    alpha=0.15,
    figsize=(10, 5),
    title=None,
    ax=None,
):
    """
    Plot observed data and model forecasts.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Observed data. Index must contain dates.

    df_preds : pd.DataFrame
        Forecast dataframe containing prediction intervals.

    conf_level : float, default=0.90
        Confidence interval level.

    data_col : str
        Column containing observed values.

    model_col : str
        Column identifying the model.

    date_col : str
        Column containing forecast dates.

    pred_col : str
        Column containing point forecasts.

    color_palette : str
        Seaborn/matplotlib palette name.

    linestyle : str
        Forecast line style.

    alpha : float
        Confidence interval transparency.

    figsize : tuple
        Figure size.

    title : str, optional
        Plot title.

    ax : matplotlib.axes.Axes, optional
        Existing axis.

    Returns
    -------
    fig, ax
    """

    if df_preds is None:
        raise ValueError("df_preds cannot be None")

    ci = int(conf_level * 100)
    lower_col = f"lower_{ci}"
    upper_col = f"upper_{ci}"

    required_cols = [
        date_col,
        pred_col,
        model_col,
        lower_col,
        upper_col,
    ]

    missing = [c for c in required_cols if c not in df_preds.columns]
    if missing:
        raise ValueError(f"Missing columns in df_preds: {missing}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    models = sorted(df_preds[model_col].unique())

    colors = sns.color_palette(color_palette, n_colors=len(models))
    color_map = dict(zip(models, colors))

    # Observed data
    if data is not None:
        ax.plot(
            data[date_col],
            data[data_col],
            color="black",
            linewidth=2,
            label="Observed",
        )

    # Forecasts
    for model in models:

        preds_ = df_preds.loc[df_preds[model_col] == model].sort_values(
            date_col
        )

        color = color_map[model]

        ax.plot(
            preds_[date_col],
            preds_[pred_col],
            color=color,
            linestyle=linestyle,
            label=str(model),
        )

        ax.fill_between(
            preds_[date_col],
            preds_[lower_col],
            preds_[upper_col],
            color=color,
            alpha=alpha,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("New cases")

    if title is None:
        title = f"Forecasts with {ci}% prediction interval"

    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()

    return fig, ax
