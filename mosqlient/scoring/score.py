import numpy as np
import pandas as pd
import altair as alt
from typing import Optional
from numpy.typing import NDArray
from scipy.stats import lognorm
from mosqlient import get_prediction_by_id
from scoringrules import crps_normal, crps_lognormal, logs_normal
from mosqlient.prediction_optimize import get_df_pars
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_point_metrics(y_true, y_pred, metric):
    """
    Evaluate multiple sklearn metrics on given true and predicted values.

    Parameters:
    -------------
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    metrics (str): Options: ['MAE', 'MSE'] .

    Returns:
    Scores.
    """

    if metric == "MAE":

        m = mean_absolute_error

    if metric == "MSE":

        m = mean_squared_error

    score = m(y_true, y_pred)

    return score


def compute_interval_score(
    lower_bound, upper_bound, observed_value, alpha=0.05
):
    """
    Calculate the interval score for a given prediction interval and observed value.

    Parameters:
    ------------------
    lower_bound: float | np.array
        The lower bound of the prediction interval.
    upper_bound: float | np.array
        The upper bound of the prediction interval.
    observed_value: float | np.array
        The observed value.
    alpha: float
        The significance level of the interval. Default is 0.05 (for 95% prediction intervals).

    Returns:
    -----------
    float or np.array: The interval score.
    """

    interval_width = upper_bound - lower_bound

    # Compute penalties
    penalty_lower = 2 / alpha * np.maximum(0, lower_bound - observed_value)
    penalty_upper = 2 / alpha * np.maximum(0, observed_value - upper_bound)

    penalty = penalty_lower + penalty_upper

    return interval_width + penalty


def compute_wis(
    df: pd.DataFrame,
    observed_value: NDArray[np.float64],
    w_0: float = 1 / 2,
    w_k: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Calculate the weighted interval score for a given prediction dataframe and observed value. In the dataframe the column `pred``
    must represent the median and each prediction interval must be enconded as `lower_{1-alpha}*100` and `upper_{1-alpha}*100`,
    where alpha is the significance level of the interval.

    Parameters:
    ------------------
    df:  pd.DataFrame
        The lower bound of the prediction interval.
    observed_value: float | np.array
        The observed value.
    w_0: float
        Initial weight.
    w_k: Optional | np.array
        Weights for each prediction interval, if None the weights are computed based on the
        prediction intervals (w_k = alpha_k/2).

    Returns:
    -----------
    float or np.array:
        The weighted interval score.
    """
    observed_value = np.asarray(observed_value)
    if observed_value.ndim == 0:
        observed_value = observed_value.reshape(1)

    lower_cols = [col for col in df.columns if col.startswith("lower_")]
    alphas = (
        1 - (np.array([float(col.split("_")[-1]) for col in lower_cols])) / 100
    )
    K = len(alphas)

    if w_k is None:
        w_k = alphas / 2
    elif len(w_k) != K:
        raise ValueError(
            f"Weights length {len(w_k)} doesn't match intervals count {K}"
        )

    interval_scores = np.zeros_like(observed_value, dtype=np.float64)

    for alpha, weight in zip(alphas, w_k):
        level = int((1 - alpha) * 100)
        interval_scores += weight * compute_interval_score(
            lower_bound=df[f"lower_{level}"].values,
            upper_bound=df[f"upper_{level}"].values,
            observed_value=observed_value,
            alpha=alpha,
        )

    median_error = np.abs(observed_value - df["pred"].values.reshape(-1))
    return (w_0 * median_error + interval_scores) / (K + 0.5)


def plot_bar_score(data: pd.DataFrame, score: str) -> alt.Chart:
    """
    Function to plot a bar chart based on scorer.summary dataframe

    Parameters:
    --------------
    data: pd.DataFrame
    score: str
        Valid options are: ['mae', 'mse', 'crps', 'log_score']
    """
    data = data.reset_index()
    data["id"] = data["id"].astype(str)

    bar_chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("id:N", axis=alt.Axis(labelAngle=360)).title("Model"),
            y=alt.Y(f"{score}:Q").title(score),
            color=alt.Color("id", legend=alt.Legend(title="Model")),
        )
        .properties(
            title=f"{score} score",
            width=400,
            height=300,
        )
    )

    return bar_chart


def plot_score(
    data: pd.DataFrame, df_melted: pd.DataFrame, score: str = "CRPS"
) -> alt.VConcatChart:
    """
    Function that returns an Altair panel with the time series of cases and the
    time series of the score for each model.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame with the time series of cases must contain the columns
        `date` and `casos`.
    df_melted : pd.DataFrame
        The DataFrame must contains the columns:
        * date: with the date';
            * variable: with the models name;
        * '{score}_score': with the score value
    score: str
        Name of the score metric. Available options include: ['CRPS','interval','wis','log']
    """

    if score == "CRPS":
        title = "CRPS score"
        subtitle = "Lower is better"

    if score == "interval":
        title = "Interval score"
        subtitle = "Lower is better"

    if score == "wis":
        title = "WIS"
        subtitle = "Lower is better"

    if score == "log":
        title = "Log score"
        subtitle = "Bigger is better"

    timedata = (
        alt.Chart(data)
        .mark_line()
        .encode(x="date", y="casos", color=alt.value("black"))
        .properties(width=400, height=300)  # Set the width  # Set the height
    )

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["date"], empty=False
    )

    graph_score = (
        alt.Chart(df_melted)
        .mark_point(filled=False)
        .encode(
            x="date",
            y=f"{score}_score",
            color=alt.Color("variable", legend=alt.Legend(legendX=100)),
        )
        .properties(width=400, height=250)  # Set the width  # Set the height
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = (
        alt.Chart(df_melted)
        .mark_point()
        .encode(  # TODO: Not used
            x="date",
            opacity=alt.value(0),
        )
        .add_params(nearest)
    )

    # Draw points on the line, and highlight based on selection
    points = graph_score.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw a rule at the location of the selection
    columns = list(df_melted.variable.unique())
    tooltip = [
        alt.Tooltip(c, type="quantitative", format=".2f") for c in columns
    ]
    tooltip.insert(0, alt.Tooltip("date:T", title="Date"))
    rules = (
        alt.Chart(df_melted)
        .transform_pivot("variable", value=f"{score}_score", groupby=["date"])
        .mark_rule(color="gray")
        .encode(
            x="date",
            opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
            tooltip=tooltip,
        )
        .add_params(nearest)
    )

    return timedata.properties(
        width=400, height=150, title="New cases"
    ) & alt.layer(  # Set the width  # Set the height
        graph_score, points, rules
    ).properties(
        title={"text": title, "subtitle": subtitle}
    )


class Scorer:
    """
    A class to compare the score of the models.

    Attributes
    ----------

    df_true: pd.DataFrame
        DataFrame of the cases provided by the user.


    filtered_df_true: pd.DataFrame
        DataFrame of the cases provided by the user filtered according
        to the interval of the predictions or with the `set_date_range` method .

    ids: Optional[list[int]]
        The list of the predictions id that will be compared


    dict_df_ids: dict[pd.DataFrame]
        A dict of DataFrames of the predictions. If the key is int it refers
        to the ids passed in the init. If it is `pred` it refers to the
        dataframe of the predictions provided by the user.

    filtered_dict_df_ids: dict[pd.DataFrame]
        A dict of DataFrames of the predictions. If the key is int it refers to
        the ids passed in the init. If it is `pred` it refers to the dataframe
        of the predictions provided by the user. The DataFrames are filtered
        according to the interval of the predictions or with the
        `set_date_range` method.

    min_date: str
        Min date that will include the information of the df_true and predictions.

    max_date: str
        Max date that will include the information of the df_true and predictions.

    mae : dict
        Dict where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the values of
        the dict are the mean absolute error.

    mse: dict
        Dict where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the values of the
        dict are the mean squared error.

    crps: tuple of dicts
        Dict where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the values of the
        dict are the scores computed.

        The first dict contains the CRPS score computed for every predicted
        point, and the second one contains the mean values of the CRPS score
        for all the points.

        The CRPS computed assumes a normal distribution.

    log_score: tuple of dicts
        Dict where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the values of the
        dict are the scores computed.

        The first dict contains the log score computed for every predicted
        point, and the second one contains the mean values of the log score for
        all the points.

        The log score computed assumes a normal distribution.


    wis: tuple of dicts
        Dict where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the values of the
        dict are the scores computed.

        The first dict contains the weighted interval score computed for every predicted
        point, and the second one contains the mean values of the weighted interval score
        for all the points.

    summary: pd.DataFrame
        DataFrame where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the columns are
        the scores: mae, mse, and the mean of crps, log_score, interval score
        and weighted interval score.

    Methods
    -------
    start_date_range():
        Train the model.
    plot_predictions():
        Function that returns an Altair panel (alt.Chart) with the time series
        of cases and the predictions for each model.
    plot_crps():
        alt.Chart: Method that returns an Altair panel with the time series of
        cases and the time series of the CRPS score for each model.
    plot_log_score():
        alt.Chart: Method that returns an Altair panel with the time series of
        cases and the time series of the log score for each model.
    plot_interval_score():
        alt.Chart: Method that returns an Altair panel with the time series of
        cases and the time series of the interval score for each model.
    plot_wis():
        alt.Chart: Method that returns an Altair panel with the time series of
        cases and the time series of the weighted interval score for each model.
    plot_mae():
        alt.Chart : Bar chart of the MAE score for each prediction.
    plot_mse():
        alt.Chart : Bar chart of the MSE score for each prediction.
    """

    def __init__(
        self,
        api_key: str,
        df_true: pd.DataFrame,
        ids: Optional[list[int] | list[str]] = None,
        pred: Optional[pd.DataFrame] = None,
        dist: str = "log_normal",
        fn_loss: str = "median",
        conf_level: float = 0.90,
    ):
        """
        Parameters
        ----------
        df_true: pd.DataFrame
            DataFrame with the columns `date` and `casos`.
        ids : list[int]
            List of the predictions ids that it will be compared.
        pred: pd.DataFrame
            Pandas Dataframe already in the format accepted by the platform
            that will be computed the score.
        dist : {'normal', 'log_normal'}, optional, default='log_normal'
            The type of distribution used for parameter estimation.
        fn_loss : {'median', 'lower'}, optional, default='median'
            Specifies the method for parameter estimation:
            - 'median': Fits the log-normal distribution by minimizing `pred` and `upper` columns.
            - 'lower': Fits the log-normal distribution by minimizing `lower` and `upper` columns.
        conf_level: float.
            The confidence level of the predictions of the columns upper and lower.
        """

        # input validation data
        cols_df_true = ["date", "casos"]

        if not set(cols_df_true).issubset(set(list(df_true.columns))):
            raise ValueError(
                "Missing required keys in the df_true:"
                f"{set(cols_df_true).difference(set(list(df_true.columns)))}"
            )

        df_true.date = pd.to_datetime(df_true.date)
        # Ensure all the dates has the same lenght
        min_dates = [min(df_true.date)]
        max_dates = [max(df_true.date)]

        dict_df_ids = {}

        if pred is not None:
            cols_preds = [
                "date",
                f"lower_{int(100*conf_level)}",
                "pred",
                f"upper_{int(100*conf_level)}",
            ]
            if not set(cols_preds).issubset(set(list(pred.columns))):
                raise ValueError(
                    "Missing required keys in the pred:"
                    f"{set(cols_preds).difference(set(list(pred.columns)))}"
                )

            pred = get_df_pars(
                pred.copy(), conf_level=conf_level, dist=dist, fn_loss=fn_loss
            )

            dict_df_ids["pred"] = pred
            pred.date = pd.to_datetime(pred.date)
            min_dates.append(min(pred.date))
            max_dates.append(max(pred.date))

        if (ids is None or len(ids) == 0) and (pred is None):
            raise ValueError(
                "It must be provide and id or DataFrame to be compared"
            )

        if ids is not None:
            ids = [str(id_) for id_ in ids]
            for id_ in ids:
                prediction = get_prediction_by_id(api_key=api_key, id=int(id_))

                if not prediction:
                    raise ValueError(f"No Prediction found for id: {id_}")

                df_ = prediction.to_dataframe()
                df_ = df_.dropna(axis=1)
                df_ = df_.sort_values(by="date")
                df_.date = pd.to_datetime(df_.date)
                df_ = get_df_pars(
                    df_.copy(),
                    conf_level=conf_level,
                    dist=dist,
                    fn_loss=fn_loss,
                )
                dict_df_ids[id_] = df_
                min_dates.append(min(df_.date))
                max_dates.append(max(df_.date))

        min_dates = pd.to_datetime(min_dates)
        max_dates = pd.to_datetime(max_dates)
        min_date = max(min_dates)
        max_date = min(max_dates)

        # updating the dates interval
        df_true = df_true.loc[
            (df_true.date >= min_date) & (df_true.date <= max_date)
        ]
        df_true = df_true.sort_values(by="date")
        df_true.reset_index(drop=True, inplace=True)

        for id_ in dict_df_ids.keys():
            df_id = dict_df_ids[id_]
            df_id = df_id.loc[
                (df_id.date >= min_date) & (df_id.date <= max_date)
            ]
            df_id = df_id.sort_values(by="date")
            dict_df_ids[id_] = df_id

        self.df_true = df_true
        self.filtered_df_true = df_true
        self.ids = ids
        self.dict_df_ids = dict_df_ids
        self.filtered_dict_df_ids = dict_df_ids
        self.min_date = min_date
        self.max_date = max_date
        self.dist = dist
        self.conf_level = conf_level

    def set_date_range(self, start_date: str, end_date: str) -> None:
        """
         This method will redefine the interval of dates used to compute the
         scores.
         The new dates provided must be in the interval defined by the
         `__init__` method that ensures the df_true and predictions are in the
         same interval. You can access these values by score.min_date and
         score.max_date.

        Parameters
        --------------
        start_date: str
            The new start date used to compute the scores.
        end_date: str
            The new end date used to compute the scores.
        """

        if (self.min_date > pd.to_datetime(start_date)) or (
            self.max_date < pd.to_datetime(start_date)
        ):
            raise ValueError(
                "The start and end date must be between "
                + f"{self.min_date} and {self.max_date}."
            )

        df_true = self.df_true
        dict_df_ids = self.dict_df_ids

        self.filtered_df_true = df_true.loc[
            (df_true.date >= pd.to_datetime(start_date))
            & (df_true.date <= pd.to_datetime(end_date))
        ]

        for id_ in dict_df_ids.keys():
            df_id = dict_df_ids[id_]
            df_id = df_id.loc[
                (df_id.date >= pd.to_datetime(start_date))
                & (df_id.date <= pd.to_datetime(end_date))
            ]
            dict_df_ids[id_] = df_id

        self.filtered_dict_df_ids = dict_df_ids

        return None

    @property
    def mae(
        self,
    ):
        """
        dict: Dict, where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the values of the
        dict are the mean absolute error.
        """
        ids = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true

        scores = {}

        for id_ in dict_df_ids.keys():

            scores[id_] = evaluate_point_metrics(
                df_true.casos, y_pred=dict_df_ids[id_].pred, metric="MAE"
            )

        return scores

    @property
    def mse(
        self,
    ):
        """
        dict: Dict, where the keys are the id of the models or `pred` when a
        dataframe of predictions is provided by the user, and the values of the
        dict are the mean squared error.
        """

        ids = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true

        scores = {}

        for id_ in dict_df_ids.keys():

            scores[id_] = evaluate_point_metrics(
                df_true.casos, y_pred=dict_df_ids[id_].pred, metric="MSE"
            )
        return scores

    @property
    def crps(
        self,
    ):
        """
        tuple of dict: Dict where the keys are the id of the models or `pred`
        when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the CRPS score computed for every predicted
        point, and the second one contains the mean values of the CRPS score
        for all the points.

        The CRPS computed assumes a normal distribution.
        """

        ids = self.ids
        dist = self.dist
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true

        scores_curve = {}

        scores_mean = {}

        for id_ in dict_df_ids.keys():

            df_id_ = dict_df_ids[id_]

            if dist == "normal":
                score = crps_normal(
                    df_true.casos,
                    df_id_.mu,
                    df_id_.sigma,
                )
            if dist == "log_normal":
                score = crps_lognormal(
                    df_true.casos,
                    df_id_.mu,
                    df_id_.sigma,
                )

            scores_curve[id_] = pd.Series(score, index=df_true.date)

            scores_mean[id_] = np.mean(score)

        self.crps_curve = scores_curve

        return scores_curve, scores_mean

    @property
    def log_score(
        self,
    ):
        """
        tuple of dict: Dict where the keys are the id of the models or `pred`
        when a dataframe of predictions is provided by the user, and the values
        of the dict are the scores computed.

        The first dict contains the log score computed for every predicted
        point, and the second one contains the mean values of the log score
        for all the points.

        The log score computed assumes a normal distribution.
        """

        ids = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true
        dist = self.dist

        scores_curve = {}
        scores_mean = {}

        for id_ in dict_df_ids.keys():

            df_id_ = dict_df_ids[id_]

            if dist == "normal":
                score = logs_normal(
                    df_true.casos,
                    df_id_.mu,
                    df_id_.sigma,
                    negative=False,
                )
            if dist == "log_normal":
                score = lognorm.logpdf(
                    df_true.casos.values,
                    s=df_id_.sigma.values,
                    scale=np.exp(df_id_.mu.values),
                )

            # truncated the output
            score = np.maximum(score, np.repeat(-100, len(score)))

            scores_curve[id_] = pd.Series(score, index=df_true.date)
            scores_mean[id_] = np.mean(score)

        self.log_curve = scores_curve

        return scores_curve, scores_mean

    @property
    def interval_score(
        self,
    ):
        """
        tuple of dict: Dict where the keys are the id of the models or `pred`
        when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the interval score computed for every predicted
        point, and the second one contains the mean values of the interval score
        for all the points.
        """

        ids = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true
        conf_level = self.conf_level

        scores_curve = {}

        scores_mean = {}

        for id_ in dict_df_ids.keys():

            df_id_ = dict_df_ids[id_]

            score = compute_interval_score(
                df_id_[f"lower_{int(100*conf_level)}"].values,
                df_id_[f"lower_{int(100*conf_level)}"].values,
                df_true.casos.values,
                alpha=1 - conf_level,
            )

            scores_curve[id_] = pd.Series(score, index=df_true.date)

            scores_mean[id_] = np.mean(score)

        self.interval_score_curve = scores_curve

        return scores_curve, scores_mean

    @property
    def wis(self, w_0=0.5, w_k=None):
        """
        tuple of dict: Dict where the keys are the id of the models or `pred`
        when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the weighted interval score computed for every predicted
        point, and the second one contains the mean values of the weighted interval score
        for all the points.
        """

        ids = self.ids
        dict_df_ids = self.filtered_dict_df_ids
        df_true = self.filtered_df_true

        scores_curve = {}

        scores_mean = {}

        for id_ in dict_df_ids.keys():

            df_id_ = dict_df_ids[id_]

            score = compute_wis(
                df=df_id_,
                observed_value=df_true.casos.values,
                w_0=w_0,
                w_k=w_k,
            )

            scores_curve[id_] = pd.Series(score, index=df_true.date)

            scores_mean[id_] = np.mean(score)

        self.wis_score_curve = scores_curve

        return scores_curve, scores_mean

    @property
    def summary(
        self,
    ):
        """
        pd.DataFrame: DataFrame where the keys are the id of the models or
        `pred` when a dataframe of predictions is provided by the user, and
        the columns are the scores: mae, mse, and the mean of crps, log_score,
        interval_score and weighted interval score.
        """
        sum_scores = {}

        sum_scores["mae"] = self.mae

        sum_scores["mse"] = self.mse

        sum_scores["crps"] = self.crps[1]

        sum_scores["log_score"] = self.log_score[1]

        sum_scores["interval_score"] = self.interval_score[1]

        sum_scores["wis"] = self.wis[1]

        df_score = pd.DataFrame.from_dict(sum_scores, orient="columns")

        df_score.index.name = "id"

        return df_score

    def plot_mae(
        self,
    ) -> alt.Chart:
        """
        Bar chart of the MAE score for each prediction.
        """

        return plot_bar_score(self.summary, "mae")

    def plot_mse(
        self,
    ) -> alt.Chart:
        """
        Bar chart of the MSE score for each prediction.
        """

        return plot_bar_score(self.summary, "mse")

    def plot_crps(
        self,
    ) -> alt.VConcatChart:
        """
        alt.Chart: Function that returns an Altair panel with the time series
        of cases and the time series of the CRPS score for each model
        """

        crps_ = self.crps_curve

        df_crps = pd.DataFrame()

        for v in crps_.keys():

            df_crps[str(v)] = crps_[v]

        df_crps.reset_index(inplace=True)

        df_melted = pd.melt(
            df_crps, id_vars="date", value_vars=list(map(str, crps_.keys()))
        )
        df_melted = df_melted.rename(columns={"value": "CRPS_score"})

        return plot_score(self.df_true, df_melted, score="CRPS")

    def plot_log_score(
        self,
    ) -> alt.VConcatChart:
        """
        alt.Chart: Function that returns an Altair panel with the time series
        of cases and the time series of the Log score for each model
        """

        crps_ = self.log_curve

        df_crps = pd.DataFrame()

        for v in crps_.keys():

            df_crps[str(v)] = crps_[v]

        df_crps.reset_index(inplace=True)

        df_melted = pd.melt(
            df_crps, id_vars="date", value_vars=list(map(str, crps_.keys()))
        )
        df_melted = df_melted.rename(columns={"value": "log_score"})

        return plot_score(self.df_true, df_melted, score="log")

    def plot_interval_score(
        self,
    ) -> alt.VConcatChart:
        """
        alt.Chart: Function that returns an Altair panel with the time series
        of cases and the time series of the CRPS score for each model
        """

        interval_ = self.interval_score_curve

        df_interval = pd.DataFrame()

        for v in interval_.keys():

            df_interval[str(v)] = interval_[v]

        df_interval.reset_index(inplace=True)

        df_melted = pd.melt(
            df_interval,
            id_vars="date",
            value_vars=list(map(str, interval_.keys())),
        )
        df_melted = df_melted.rename(columns={"value": "interval_score"})

        return plot_score(self.df_true, df_melted, score="interval")

    def plot_wis(
        self,
    ) -> alt.VConcatChart:
        """
        alt.Chart: Function that returns an Altair panel with the time series
        of cases and the time series of the wis score for each model
        """

        wis_ = self.wis_score_curve

        df_wis = pd.DataFrame()

        for v in wis_.keys():

            df_wis[str(v)] = wis_[v]

        df_wis.reset_index(inplace=True)

        df_melted = pd.melt(
            df_wis,
            id_vars="date",
            value_vars=list(map(str, wis_.keys())),
        )
        df_melted = df_melted.rename(columns={"value": "wis_score"})

        return plot_score(self.df_true, df_melted, score="wis")

    def plot_predictions(
        self, show_ci: bool = True, width: int = 400, height: int = 300
    ) -> alt.Chart:
        """
        Function that returns an Altair panel (alt.Chart) with the time series
        of cases and the predictions for each model

        Parameters
        ---------------
        show_ci :bool
            If True it shows the confidence interval.
        width: int
            width of the plot
        width: int
            height of the plot
        """

        dict_df_ids = self.filtered_dict_df_ids
        df_true_ = self.filtered_df_true
        df_true_.loc[:, "legend"] = "Data"

        if show_ci:
            title = "Median and 95% confidence interval"
        else:
            title = "Median of predictions"

        df_to_plot = pd.DataFrame()

        for id_ in dict_df_ids.keys():

            df_ = dict_df_ids[id_]

            df_.loc[:, "model"] = id_

            df_to_plot = pd.concat([df_to_plot, df_])

        df_to_plot["model"] = df_to_plot["model"].astype(str)

        data = (
            alt.Chart(df_true_)
            .mark_circle(size=60)
            .encode(
                x="date:T",
                y="casos:Q",
                color=alt.Color(
                    "legend:N",
                    scale=alt.Scale(range=["black"]),
                    legend=alt.Legend(title=None),
                ),
            )
            .properties(
                width=width, height=height
            )  # Set the width  # Set the height
        )

        # here we define the plot of the right figure
        timeseries = (
            alt.Chart(df_to_plot, title=title)
            .mark_line()
            .encode(
                x=alt.X("date:T").title("Dates"),
                y=alt.Y("pred:Q").title("New cases"),
                color=alt.Color("model", legend=alt.Legend(title="Model")),
            )
        )

        # here we create the area that represent the confidence interval of the
        # predicitions
        timeseries_conf = timeseries.mark_area(
            opacity=0.25,
        ).encode(
            x="date:T",
            y="lower:Q",
            y2="upper:Q",
            color=alt.Color("model", legend=None),
        )

        nearest = alt.selection_point(
            nearest=True, on="pointerover", fields=["date"], empty=False
        )

        # Draw points on the line, and highlight based on selection
        points = timeseries.mark_point().encode(
            color=alt.Color("model", legend=None),
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        )

        df_true_ = df_true_.rename(columns={"casos": "pred"})

        df_true_["model"] = "cases"

        df_to_plot = pd.concat([df_to_plot, df_true_])

        columns = list(df_to_plot.model.unique())
        tooltip = [
            alt.Tooltip(c, type="quantitative", format=".0f") for c in columns
        ]
        tooltip.insert(0, alt.Tooltip("date:T", title="Date"))

        rules = (
            alt.Chart(df_to_plot)
            .transform_pivot("model", value="pred", groupby=["date"])
            .mark_rule(color="gray")
            .encode(
                x="date",
                opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
                tooltip=tooltip,
            )
            .add_params(nearest)
        )

        if show_ci:

            final = (
                data + timeseries + timeseries_conf + points + rules
            ).resolve_scale(color="independent")

        else:
            final = alt.layer(data, timeseries, points, rules).resolve_scale(
                color="independent"
            )

        return final
