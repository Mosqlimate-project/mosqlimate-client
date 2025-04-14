import matplotlib as mpl
import matplotlib.pyplot as plt
from mosqlient.forecast.ensemble import validate_df_preds

# Definir a cor das bordas (spines) como cinza
mpl.rcParams["axes.edgecolor"] = "gray"
# Definir a cor das linhas dos ticks maiores e menores como cinza
mpl.rcParams["xtick.color"] = "gray"
mpl.rcParams["ytick.color"] = "gray"
mpl.rcParams["xtick.labelcolor"] = "black"
mpl.rcParams["ytick.labelcolor"] = "black"


def plot_preds(
    data,
    df_preds,
    conf_level=0.9,
    data_col="casprov",
    color_palette="Set2",
    linestyle="-",
):
    """
    Plot data against predictions. If `data` is None, only the predictions will be plotted.

    Parameters
    ----------
    data: pd.DataFrame or None
        A DataFrame with a datetime index and a column containing the case values.

    df_preds: pd.DataFrame
        A DataFrame containing the predictions. It must include the following columns:
        'date', 'lower', 'pred', 'upper', and 'model_id'.

    data_col: str
        The name of the column containing the case values in the `data` DataFrame.

    color_palette: str
        The name of the color palette used for coloring the predictions.

    linestyle: str
        The linestyle for the prediction plots.

    Returns
    -------
    ax.Subplot
        A figure displaying the predictions.

    """

    validate_df_preds(df_preds, conf_level=conf_level)

    models = df_preds.model_id.unique()
    colors = plt.get_cmap(color_palette).colors[: len(models)]
    color_map = dict(zip(models, colors))

    _, ax = plt.subplots()

    if data is not None:
        ax.plot(data.index, data[data_col], color="black", label="Data")

    for model in models:

        preds_ = df_preds.loc[df_preds.model_id == model]

        ax.plot(
            preds_.date,
            preds_.pred,
            linestyle=linestyle,
            color=color_map[model],
            label=f"{model}",
        )

        ax.fill_between(
            preds_.date,
            preds_[f"lower_{int(100*conf_level)}"],
            preds_[f"upper_{int(100*conf_level)}"],
            color=color_map[model],
            alpha=0.1,
        )

    ax.grid()
    ax.legend()

    ax.set_xlabel("Date")

    ax.set_ylabel("New cases")

    return ax
