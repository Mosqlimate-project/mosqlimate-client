import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm, norm


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
        tent_qs = lognorm.ppf(
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
        tent_qs = lognorm.ppf(
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
        tent_qs = norm.ppf(
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
        tent_qs = norm.ppf(
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
            lambda row: lognorm.ppf(
                [0.5, (1 - conf_level) / 2, (1 + conf_level) / 2],
                s=row["sigma"],
                scale=np.exp(row["mu"]),
            ),
            axis=1,
            result_type="expand",
        )
    elif dist == "normal":
        theo_pred_df = preds_.apply(
            lambda row: norm.ppf(
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
