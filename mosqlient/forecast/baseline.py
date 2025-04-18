from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pmdarima.arima import auto_arima
from pmdarima import preprocessing as ppc


def get_next_n_weeks(ini_date: str, next_days: int) -> list:
    """
    Return a list of dates with the {next_weeks} weeks after ini_date.
    This function was designed to generate the dates of the forecast
    models.
    Parameters
    ----------
    ini_date : str
        Initial date.
    next_weeks : int
        Number of weeks to be included in the list after the date in
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


def get_prediction_dataframe(
    model, date, boxcox, horizon=None, alphas=[0.05, 0.1, 0.2, 0.5]
) -> pd.DataFrame:
    """
    Function to organize the predictions of the ARIMA model in a pandas DataFrame.

    Parameters
    ----------
    horizon: int
        The number of weeks forecasted by the model
    end_date: str
        Last week of the out of the sample evaluation. The first week is after the last training observation.
    plot: bool
        If true the plot of the model out of the sample is returned
    """

    dfs = []
    for alpha in alphas:
        if horizon is None:
            preds_ = model.predict_in_sample(return_conf_int=True, alpha=alpha)
        else:
            preds_ = model.predict(
                n_periods=horizon, return_conf_int=True, alpha=alpha
            )
        df_ = pd.DataFrame(
            preds_[1],
            columns=[
                f"lower_{int((1-alpha)*100)}",
                f"upper_{int((1-alpha)*100)}",
            ],
        )
        dfs.append(df_)

    df_preds = pd.concat(dfs, axis=1)

    try:
        df_preds["pred"] = preds_[0].values
    except:
        df_preds["pred"] = preds_[0]

    if df_preds["pred"].values[0] == 0:
        df_preds = df_preds.iloc[1:]
        date = date[1:]

    for col in df_preds.columns:
        df_preds[col] = boxcox.inverse_transform(df_preds[col])[0]

    df_preds["date"] = date

    return df_preds


def plot_predictions(
    df_preds: pd.DataFrame, title: str = "", alphas=[0.05, 0.1, 0.2, 0.5]
) -> None:
    """
    Function to plot the predictions of the model.

    Parameters
    ----------
    df_preds: pd.DataFrame
        Dataframe with the columns: ['date', 'data', 'pred', 'lower', 'upper'].
    title: str
        Title of the plot.
    """

    fig, ax = plt.subplots(1, figsize=(6, 4))

    ax.plot(df_preds.date, df_preds.data, color="black", label="Data")

    ax.plot(df_preds.date, df_preds.pred, color="tab:orange", label="ARIMA")

    for alpha in alphas:

        ax.fill_between(
            df_preds.date,
            df_preds[f"lower_{int((1-alpha)*100)}"],
            df_preds[f"upper_{int((1-alpha)*100)}"],
            color="tab:orange",
            alpha=0.1,
        )

    ax.legend()

    ax.grid()

    ax.set_title(title)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))

    ax.set_xlabel("Date")

    ax.set_ylabel("New cases")


def plot_forecast(
    df_for: pd.DataFrame,
    df_train: pd.DataFrame,
    last_obs: int,
    alphas=[0.05, 0.1, 0.2, 0.5],
) -> None:
    """
    Function to plot the forecast of the model.

    Parameters
    ----------
    df_for: pd.DataFrame
        Dataframe with the forecast results, with the columns: ['date', 'pred', 'lower', 'upper']
    df_preds: pd.DataFrame
        Dataframe with the columns: ['data'] and a datetime index.
    last_obs: int
        Number of previous observations of the data included.
    """

    df_train = df_train.tail(last_obs)

    fig, ax = plt.subplots(1, figsize=(6, 4))

    ax.plot(df_train.index, df_train.data, color="black", label="Data")

    ax.plot(df_for.date, df_for.pred, color="tab:red", label="Forecast")

    for alpha in alphas:

        ax.fill_between(
            df_for.date,
            df_for[f"lower_{int((1-alpha)*100)}"],
            df_for[f"upper_{int((1-alpha)*100)}"],
            color="tab:red",
            alpha=0.1,
        )

    ax.plot(
        [df_train.index[-1], df_for.date[0]],
        [df_train[f"data"].values[-1], df_for.pred.values[0]],
        ls="--",
        color="black",
    )

    ax.legend()

    ax.grid()

    ax.set_title("Forecast ARIMA")

    ax.set_xlabel("Date")

    ax.set_ylabel("New cases")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%Y"))


class InvalidDataFrameError(Exception):
    """Custom exception for invalid DataFrame."""

    pass


class Arima:
    """
    A class to implement a ARIMA model as baseline for forecast cases in some city.

    Attributes
    ----------
    df : pd.DataFrame
        A pandas dataframe with the columns y and a datetime index

    Methods
    -------
    train():
        Train the model.
    predict_in_sample():
        Predictions of the model in sample.
    predict_out_of_sample():
        Predictions of the model out of sample.
    forecast():
        Forecast models
    """

    def __init__(self, df: pd.DataFrame, **auto_arima_kwargs):
        """
        Constructs all the necessary attributes for the Arima object.

        Parameters
        ----------
            df : pd.DataFrame
            A pandas dataframe with the column y and a datetime index
            auto_arima_kwargs : dict
            All parameters that can be passed to pmdarima.arima.auto_arima.

        """
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise InvalidDataFrameError(
                "The DataFrame's index is not of datetime type."
            )

        if df.shape[1] != 1:
            raise InvalidDataFrameError(
                "The DataFrame must have one single column."
            )

        if df.columns[0] != "y":
            raise InvalidDataFrameError("The column must be named `y`.")

        df["y"] = df["y"].astype(float)

        self.df = df

        default_auto_arima_kwargs = {
            "seasonal": False,
            "trace": True,
            "maxiter": 100,
            "error_action": "ignore",
            "information_criterion": "aic",
            "suppress_warnings": True,
            "stepwise": True,
        }

        default_auto_arima_kwargs.update(auto_arima_kwargs)

        self.auto_arima_kwargs = default_auto_arima_kwargs

    def train(self, train_ini_date: str, train_end_date: str):
        """
        Train the ARIMA model

        Parameters
        ----------
            train_ini_date: str
                Initial date for model training
            train_end_date: str
                End date for model training
        """

        df_train = self.df.copy()

        df_train = df_train.loc[
            (df_train.index >= pd.to_datetime(train_ini_date))
            & (df_train.index <= pd.to_datetime(train_end_date))
        ]

        self.df_train = df_train

        boxcox = ppc.BoxCoxEndogTransformer().fit(df_train.y)

        self.boxcox = boxcox

        df_train.loc[:, "y"] = boxcox.transform(df_train.y)[0]

        model = auto_arima(df_train.y, **self.auto_arima_kwargs)

        model.fit(df_train.y)

        self.model = model

        return model

    def predict_in_sample(self, plot: bool = True) -> pd.DataFrame:
        """
        Returns the model performance in the sample.

        Parameters
        ----------
            plot: bool
                If true the plot of the model in the sample is returned

        """

        df_train = self.df_train.copy()

        df_in_sample = get_prediction_dataframe(
            self.model, df_train.index, self.boxcox
        )

        df_in_sample = df_in_sample.merge(
            df_train, left_on="date", right_index=True
        )

        df_in_sample = df_in_sample.rename(columns={"y": "data"})

        df_in_sample["data"] = self.boxcox.inverse_transform(
            df_in_sample["data"]
        )[0]

        if plot:
            plot_predictions(df_in_sample, title="In sample predictions")

        return df_in_sample

    def predict_out_of_sample(
        self, horizon: int, end_date: str, plot=True
    ) -> pd.DataFrame:
        """
        Returns the model performance out of the sample.
        The predictions are returned by windows of {horizon} observations. After each window
        the model is updated with the data of the last observations forecasted.

        Parameters
        ----------
            horizon: int
                The number of observations forecasted by the model
            end_date: str
                Last week of the out of sample evaluation. The first week is after the last training observation.
            plot: bool
                If true the plot of the model out of the sample is returned
        """

        df = self.df.copy()

        df.loc[:, "y"] = self.boxcox.transform(df.y)[0]

        model = self.model

        date = get_next_n_weeks(
            self.df_train.index[-1].strftime("%Y-%m-%d"), horizon
        )

        df_preds = get_prediction_dataframe(
            model, date, self.boxcox, horizon=horizon
        )

        while pd.Timestamp(date[-1]) < pd.to_datetime(end_date):

            date = get_next_n_weeks(date[-1].strftime("%Y-%m-%d"), horizon)

            df_preds = pd.concat(
                [
                    df_preds,
                    get_prediction_dataframe(
                        model.update(df.loc[date[0] : date[-1]]),
                        date,
                        self.boxcox,
                        horizon=horizon,
                    ),
                ]
            )

        df_preds.date = pd.to_datetime(df_preds.date)

        df_preds = df_preds.merge(df, left_on="date", right_index=True)

        df_preds = df_preds.loc[df_preds.date <= end_date]

        df_preds = df_preds.dropna()

        df_preds.loc[:, "y"] = self.boxcox.inverse_transform(df_preds["y"])[0]

        df_preds = df_preds.rename(columns={"y": "data"})

        if plot:

            plot_predictions(df_preds, title="Out of sample predictions")

        return df_preds

    def forecast(
        self, horizon: int, plot: bool, last_obs: int
    ) -> pd.DataFrame:
        """
        Returns the forecast of the model.
        Before applying this method is necessary to call the `train()` method.
        The `forecast()` method will forecast {horizon} observations ahead of the last observation
        used to train the model in the `train()` method.

        Parameters
        ----------
            horizon: int
                The number of observations forecasted by the model
            plot: bool
                If true return a figure with the forecasted values.
            last_obs: bool
                The number of last observations plotted in the figure
        """

        df_train = self.df_train.copy()

        df_train = df_train.rename(columns={"y": "data"})

        df_train["data"] = self.boxcox.inverse_transform(df_train["data"])[0]

        model = self.model

        date = get_next_n_weeks(
            df_train.index[-1].strftime("%Y-%m-%d"), horizon
        )

        df_preds = get_prediction_dataframe(
            model, date, self.boxcox, horizon=horizon
        )

        if plot:

            plot_forecast(df_preds, df_train, last_obs)

        return df_preds
