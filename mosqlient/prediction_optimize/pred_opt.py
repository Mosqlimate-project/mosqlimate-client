import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize


def get_lognormal_pars(
    med: float,
    lwr: float,
    upr: float,
    conf_level: float = 0.90,
    fn_loss: str = "median",
) -> tuple:
    """
    Estimate the parameters of a log-normal distribution based on forecasted median,
    lower, and upper bounds.

    This function estimates the mu and sigma parameters of a log-normal distribution
    given a forecast's known median (`med`), lower (`lwr`), and upper (`upr`) confidence
    interval bounds. The optimization minimizes the discrepancy between the theoretical
    quantiles of the log-normal distribution and the provided forecast values.

    Parameters
    ----------
    med : float
        The median of the forecast distribution.
    lwr : float
        The lower bound of the forecast (corresponding to `(1 - alpha)/2` quantile).
    upr : float
        The upper bound of the forecast (corresponding to `(1 + alpha)/2` quantile).
    Conf_level : float, optional, default=0.90
        Confidence level used to define the lower and upper bounds.
    fn_loss : {'median', 'lower'}, optional, default='median'
        The optimization criterion for fitting the log-normal distribution:
        - 'median': Minimizes the error in estimating `med` and `upr`.
        - 'lower': Minimizes the error in estimating `lwr` and `upr`.

    Returns
    -------
    tuple
        A tuple `(mu, sigma)`, where:
        - `mu` is the estimated location parameter of the log-normal distribution.
        - `sigma` is the estimated scale parameter.

    Notes
    -----
    - The function uses the Nelder-Mead optimization method to minimize the loss function.
    - If `fn_loss='median'`, the optimization prioritizes minimizing the difference
      between the estimated and actual median (`med`) and upper bound (`upr`).
    - If `fn_loss='lower'`, the optimization prioritizes minimizing the difference
      between the estimated lower bound (`lwr`) and upper bound (`upr`).
    """

    if fn_loss not in {"median", "lower"}:
        raise ValueError(
            "Invalid value for fn_loss. Choose 'median' or 'lower'."
        )

    if any(x < 0 for x in [med, lwr, upr]):
        raise ValueError("med, lwr, and upr must be non-negative.")

    def loss_lower(theta):
        tent_qs = st.lognorm.ppf(
            [(1 - conf_level) / 2, (1 + conf_level) / 2],
            s=theta[1],
            scale=np.exp(theta[0]),
        )
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = (
                abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
            )
        return attained_loss

    def loss_median(theta):
        tent_qs = st.lognorm.ppf(
            [0.5, (1 + conf_level) / 2], s=theta[1], scale=np.exp(theta[0])
        )
        if med == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = (
                abs(med - tent_qs[0]) / med + abs(upr - tent_qs[1]) / upr
            )
        return attained_loss

    if med == 0:
        mustar = np.log(0.1)
    else:
        mustar = np.log(med)

    if fn_loss == "median":
        result = minimize(
            loss_median,
            x0=[mustar, 0.5],
            bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],
            method="Nelder-mead",
            options={
                "xatol": 1e-6,
                "fatol": 1e-6,
                "maxiter": 1000,
                "maxfev": 1000,
            },
        )
    if fn_loss == "lower":
        result = minimize(
            loss_lower,
            x0=[mustar, 0.5],
            bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],
            method="Nelder-mead",
            options={
                "xatol": 1e-8,
                "fatol": 1e-8,
                "maxiter": 5000,
                "maxfev": 5000,
            },
        )

    return result.x


def get_normal_pars(
    med: float,
    lwr: float,
    upr: float,
    conf_level: float = 0.90,
    fn_loss="median",
) -> tuple:
    """
    Estimate the parameters of a normal (Gaussian) distribution given forecasted median,
    lower, and upper bounds.

    This function estimates the mean (`mu`) and standard deviation (`sigma`) of a normal
    distribution that best fits the given forecasted median (`med`), lower (`lwr`), and
    upper (`upr`) confidence interval bounds. The optimization minimizes the discrepancy
    between the theoretical quantiles of the normal distribution and the provided forecast values.

    Parameters
    ----------
    med : float
        The median of the forecast distribution.
    lwr : float
        The lower bound of the forecast (corresponding to `(1 - alpha)/2` quantile).
    upr : float
        The upper bound of the forecast (corresponding to `(1 + alpha)/2` quantile).
    conf_level : float, optional, default=0.90
        Confidence level used to define the lower and upper bounds.
        fn_loss : {'median', 'lower'}, optional, default='median'
        The optimization criterion for fitting the log-normal distribution:
        - 'median': Minimizes the error in estimating `med` and `upr`.
        - 'lower': Minimizes the error in estimating `lwr` and `upr`.

    Returns
    -------
    tuple
        A tuple `(mu, sigma)`, where:
        - `mu` is the estimated mean of the normal distribution.
        - `sigma` is the estimated standard deviation of the normal distribution.

    Notes
    -----
    - The function uses the Nelder-Mead optimization method to find the best-fitting parameters.
    - The optimization minimizes the difference between the provided bounds (`lwr`, `upr`) and
      the theoretical quantiles of the estimated normal distribution.
    - If `lwr == 0`, only the upper bound (`upr`) is used in the optimization to prevent
      division by zero.
    """

    def loss_lower(theta):
        tent_qs = st.norm.ppf(
            [(1 - conf_level) / 2, (1 + conf_level) / 2],
            loc=theta[0],
            scale=theta[1],
        )
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = (
                abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
            )
        return attained_loss

    def loss_median(theta):
        tent_qs = st.norm.ppf(
            [0.5, (1 + conf_level) / 2], loc=theta[0], scale=theta[1]
        )
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = (
                abs(med - tent_qs[0]) / med + abs(upr - tent_qs[1]) / upr
            )
        return attained_loss

    sigmastar = max((upr - lwr) / 4, 1e-4)

    if fn_loss == "lower":
        result = minimize(
            loss_lower,
            x0=[med, sigmastar],
            bounds=[(-5 * abs(med), 5 * abs(med)), (0, 100000)],
            method="Nelder-mead",
        )

    if fn_loss == "median":
        result = minimize(
            loss_median,
            x0=[med, sigmastar],
            bounds=[(-5 * abs(med), 5 * abs(med)), (0, 100000)],
            method="Nelder-mead",
        )

    return result.x


def get_df_pars(
    preds_: pd.DataFrame,
    conf_level: float = 0.9,
    dist: str = "log_normal",
    fn_loss: str = "median",
    return_estimations: bool = False,
) -> pd.DataFrame:
    """
    Compute distribution parameters and optionally return estimated confidence intervals.

    This function processes a DataFrame containing prediction intervals and computes the
    parameters of a specified probability distribution ('normal' or 'log_normal').
    Additional columns for the estimated median, lower, and upper bounds are returned
    if `return_estimations` is set to True.

    Parameters
    ----------
    preds_ : pd.DataFrame
        DataFrame with columns: 'date', 'pred', 'lower', 'upper', and 'model_id'.
    conf_level: float, optional, default=0.9
        Confidence level used for computing the confidence intervals. Valid options are
        [0.5, 0.8, 0.9, 0.95]
    dist : {'normal', 'log_normal'}, optional, default='log_normal'
        The type of distribution used for parameter estimation.
    fn_loss : {'median', 'lower'}, optional, default='median'
        Specifies the method for parameter estimation:
        - 'median': Fits the log-normal distribution by minimizing `pred` and `upper` columns.
        - 'lower': Fits the log-normal distribution by minimizing `lower` and `upper` columns.
    return_estimations : bool, optional, default=False
        If True, returns additional columns with estimated median ('fit_med'), lower bound ('fit_lwr'),
        and upper bound ('fit_upr').

    Returns
    -------
    pd.DataFrame
        The input DataFrame augmented with the following columns:
        - 'mu', 'sigma': Parameters of the specified distribution.
        - If `return_estimations=True`, also includes: 'fit_med', 'fit_lwr', 'fit_upr'.

    Notes
    -----
    - The function applies `get_lognormal_pars` or `get_normal_pars` row-wise to estimate
      the distribution parameters.
    - When `return_estimations=True`, the function also computes the theoretical quantiles
      based on the estimated distribution parameters.
    """

    if dist == "log_normal":
        preds_[["mu", "sigma"]] = preds_.apply(
            lambda row: get_lognormal_pars(
                med=row["pred"],
                lwr=row[f"lower_{int(100*conf_level)}"],
                upr=row[f"upper_{int(100*conf_level)}"],
                fn_loss=fn_loss,
            ),
            axis=1,
            result_type="expand",
        )
    elif dist == "normal":
        preds_[["mu", "sigma"]] = preds_.apply(
            lambda row: get_normal_pars(
                med=row["pred"],
                lwr=row[f"lower_{int(100*conf_level)}"],
                upr=row[f"upper_{int(100*conf_level)}"],
                fn_loss=fn_loss,
            ),
            axis=1,
            result_type="expand",
        )

    if not return_estimations:
        return preds_

    if dist == "log_normal":
        theo_pred_df = preds_.apply(
            lambda row: st.lognorm.ppf(
                [0.5, (1 - conf_level) / 2, (1 + conf_level) / 2],
                s=row["sigma"],
                scale=np.exp(row["mu"]),
            ),
            axis=1,
            result_type="expand",
        )
    elif dist == "normal":
        theo_pred_df = preds_.apply(
            lambda row: st.norm.ppf(
                [0.5, (1 - conf_level) / 2, (1 + conf_level) / 2],
                loc=row["mu"],
                scale=row["sigma"],
            ),
            axis=1,
            result_type="expand",
        )

    theo_pred_df.columns = ["fit_med", "fit_lwr", "fit_upr"]
    preds_ = pd.concat([preds_, theo_pred_df], axis=1)

    return preds_


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

quantile_levels = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]


def _fit_ln_least_squares(p, Q):
    """
    Fit a lognormal distribution from quantile estimates using least squares.

    This function estimates the parameters of a lognormal distribution
    (`mu` and `sigma`) from a set of quantile values and their associated
    cumulative probabilities. The estimation is performed by transforming
    the quantiles into log-space and fitting a linear relationship between
    the standard normal quantiles and the logarithm of the observed quantiles.

    Parameters
    ----------
    p : array-like of float
        Cumulative probability levels associated with the quantiles.
        Values must be in the interval ``(0, 1)``.

    Q : array-like of float
        Quantile values corresponding to the probabilities in `p`.
        Non-positive values are replaced by ``0.01`` before applying
        the logarithmic transformation.

    Returns
    -------
    mu : float
        Estimated mean parameter of the underlying normal distribution
        in log-space.

    sigma : float
        Estimated standard deviation parameter of the underlying normal
        distribution in log-space.

    Notes
    -----
    The method assumes that the data follow a lognormal distribution:

    .. math::

        Y \\sim \\text{LogNormal}(\\mu, \\sigma)

    such that:

    .. math::

        \\log(Y) \\sim \\mathcal{N}(\\mu, \\sigma^2)

    The parameters are estimated by solving a simple linear regression:

    .. math::

        \\log(Q_p) = \\mu + \\sigma z_p

    where :math:`z_p` are the standard normal quantiles associated with
    probabilities `p`.

    Examples
    --------
    >>> p = [0.1, 0.5, 0.9]
    >>> Q = [2.3, 5.0, 10.2]
    >>> mu, sigma = _fit_ln_least_squares(p, Q)
    """
    Q = np.where(Q <= 0, 0.01, Q)

    z = st.norm.ppf(p)
    y = np.log(Q)

    z_bar = z.mean()
    y_bar = y.mean()

    sigma = np.sum((z - z_bar) * (y - y_bar)) / np.sum((z - z_bar) ** 2)
    mu = y_bar - sigma * z_bar

    return mu, sigma


def fit_row(row):
    """
    Estimate lognormal distribution parameters for a single dataframe row.

    This function extracts quantile values from a dataframe row using the
    global variable `quantile_cols`, fits a lognormal distribution using
    `_fit_ln_least_squares`, and returns the estimated parameters.

    Parameters
    ----------
    row : pandas.Series
        Row containing quantile columns defined in `quantile_cols`.

    Returns
    -------
    pandas.Series
        Series containing the estimated lognormal parameters:

        - ``mu`` : float
            Mean parameter in log-space.
        - ``sigma`` : float
            Standard deviation parameter in log-space.

    Notes
    -----
    This function depends on the following global variables:

    - `quantile_cols` : list of str
        Column names containing quantile values.
    - `QUANTILES_LEVELS` : array-like of float
        Probability levels associated with the quantiles.

    Examples
    --------
    >>> df[['mu', 'sigma']] = df.apply(fit_row, axis=1)
    """
    Q = row[quantile_cols].values.astype(float)

    mu, sigma = _fit_ln_least_squares(quantile_levels, Q)

    return pd.Series({"mu": mu, "sigma": sigma})


def get_df_pars_ls(df):
    """
    Estimate lognormal distribution parameters for all rows in a dataset.

    This function converts the input object to a pandas DataFrame,
    applies the `fit_row` function to each row, and appends the
    estimated lognormal parameters (`mu` and `sigma`) as new columns.

    Parameters
    ----------
    df : xarray.Dataset or pandas.DataFrame
        Input dataset containing quantile columns defined in the global
        variable `quantile_cols`. If an xarray object is provided,
        it must implement the ``to_dataframe()`` method.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all original columns plus:

        - ``mu`` : float
            Estimated mean parameter of the underlying normal
            distribution in log-space.
        - ``sigma`` : float
            Estimated standard deviation parameter of the underlying
            normal distribution in log-space.

    """

    df[["mu", "sigma"]] = df.apply(fit_row, axis=1)

    return df
