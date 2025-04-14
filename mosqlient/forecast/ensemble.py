from typing import Union, cast, List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scoringrules import crps_lognormal, crps_normal
from mosqlient.prediction_optimize import get_df_pars


def validate_df_preds(df_preds: pd.DataFrame, conf_level=0.9):
    """
    Validade if the predictions dataframe contains the necessary columns

    Parameters
    ------------

    df_preds: pd.DataFrame

    conf_level : float, optional, default=0.90
        Confidence level used to define the lower and upper bounds.

    Returns
    ---------
    Returns an error if df_preds is missing the required columns.
    """
    expected_cols = {
        "date",
        f"lower_{int(100*conf_level)}",
        "pred",
        f"upper_{int(100*conf_level)}",
        "model_id",
    }

    if not expected_cols.issubset(df_preds.columns):
        raise ValueError(
            f"df_preds must contain the following columns: {expected_cols}. "
            f"Missing: {expected_cols - set(df_preds.columns)}"
        )


def invlogit(y: float) -> float:

    return 1 / (1 + np.exp(-y))


def alpha_01(alpha_inv: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Function that maps from R^n to the open simplex.

    Parameters
    -----------
    alpha_inv: array of float

    Returns
    --------
    array
        Vector on the (n+1) open simplex.
    """
    K = len(alpha_inv) + 1
    z = np.full(K - 1, np.nan)  # Equivalent to rep(NA, K-1)
    alphas = np.zeros(K)  # Equivalent to rep(0, K)

    for k in range(K - 1):
        z[k] = invlogit(alpha_inv[k] + np.log(1 / (K - (k + 1))))
        alphas[k] = (1 - np.sum(alphas[:k])) * z[k]

    alphas[K - 1] = 1 - np.sum(alphas[:-1])
    return alphas


def pool_par_gauss(
    alpha: NDArray[np.float64], m: NDArray[np.float64], v: NDArray[np.float64]
) -> tuple:
    """
    Function to get the output distribution from a logarithmic pool of lognormal (or normal) distrutions

    Parameters
    ----------
    alpha : array of float
        Weigths assigned to each distribution in the pool.
    m : array of float
        mu parameter
    v : array of float
        variance parameter
    Returns
    -------
    tuple
        A tuple containing two elements. The first one is the mu and the second one the sd parameter of the distribution.

    Notes
    ------
    The logarithmic pooling method is based on the work of Carvalho, L. M., Villela, D. A., Coelho, F. C., & Bastos, L. S. (2023).
    Bayesian inference for the weights in logarithmic pooling. Bayesian Analysis, 18(1), 223-251.
    """
    if not (len(alpha) == len(m) == len(v)):
        raise ValueError(
            "The arrays 'alpha', 'm', and 'v' must have the same length."
        )

    ws = alpha / v
    vstar = 1 / np.sum(ws)
    mstar = np.sum(ws * m) * vstar
    return mstar, np.sqrt(vstar)


def linear_mix(
    weights: NDArray[np.float64],
    ms: NDArray[np.float64],
    vs: NDArray[np.float64],
) -> tuple:
    """
    Computes the mean (mu) and standard deviation (sd) of a linear mixture of normal distributions
    weighted by `weights`.

    Parameters
    ----------
    weights : np.array
        Array of weights for the linear mixture. Should sum to 1.
    ms : np.array
        Array of mean values of the normal distributions.
    vs : np.array
        Array of variance values of the normal distributions.

    Returns
    -------
    tuple (mu, sd)
        mu : float
            Mean of the resulting normal distribution.
        sd : float
            Standard deviation of the resulting normal distribution.
    """

    mu = np.dot(weights, ms)

    sd = np.sqrt(np.dot(weights**2, vs))

    return mu, sd


def get_score(
    obs: float,
    mu: float,
    sd: float,
    dist: str = "log_normal",
    metric: str = "crps",
) -> float:
    """
    Function to compute the score given a distribution
    and a predefined metric.

    Parameters
    -----------
    obs: float
        The real observation.

    mu:float
        The mu parameter of the distribution

    sd: float
        The sd parameter associated with the distribution

    dist: str ['normal', 'log_normal']
        Distribution type, either 'normal' or 'log_normal'.

    metric: str ['crps', 'log_score']
        Scoring metric, either 'crps' or 'log_score'.

    Returns
    --------
    float
        The computed score based on the given metric and distribution.
    """

    if metric == "log_score":
        if dist == "log_normal":
            return -lognorm.logpdf(obs, s=sd, scale=np.exp(mu))

        elif dist == "normal":
            return -np.log(norm.pdf(obs, loc=mu, scale=sd))

    if metric == "crps":
        if dist == "log_normal":
            return crps_lognormal(observation=obs, mulog=mu, sigmalog=sd)

        elif dist == "normal":
            return crps_normal(obs, mu, sd)

    raise ValueError(f"Invalid distribution '{dist}' and metric '{metric}'")


def find_opt_weights_log(
    obs: pd.DataFrame,
    preds: pd.DataFrame,
    order_models: list,
    dist: str = "log_normal",
    metric: str = "crps",
    bounds: tuple = (-100, 100),
) -> dict:
    """
    Function that generate the weights of the ensemble minimizing the metric selected.

    Parameters
    -----------------
    obs: pd.dataframe
        Dataframe with columns date and casos;

    preds: pd.dataframe
        Dataframe with columns date, mu, sigma, and model_id

    order_models: list
        Order of the different models in the model_id column

    dist: str ['log_normal', 'normal']
        Distribution used to represent the forecast

    metric: str ['crps', 'log_score']
        Metric used to optimize the weights

    bounds: tuple
        Tuple where the first element represents the minimum value and the second
        represents the maximum value for the bounds.

    Returns
    --------
    dict
    The dict contains the keys:
    - weights: the optmize weights by the loss
    - loss: loss function value
    """

    K = len(order_models)

    def loss(eta):

        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.drop(["date"], axis=1).reset_index(drop=True)
            ms = preds_["mu"]
            vs = preds_.sigma**2

            if not len(ms) == len(vs) == K:
                raise ValueError("n_models and vs are not the same size!")

            mu, sd = pool_par_gauss(alpha=ws, m=ms, v=vs)

            score = score + get_score(
                obs=obs.loc[obs.date == date].casos,
                mu=mu,
                sd=sd,
                dist=dist,
                metric=metric,
            )

        return score

    initial_guess = np.random.normal(size=K - 1)
    bounds_ = [bounds] * (K - 1)
    opt_result = minimize(
        loss, initial_guess, method="Nelder-mead", bounds=bounds_
    )

    optimal_weights = alpha_01(opt_result.x)

    return {"weights": optimal_weights, "loss": opt_result.fun}


def get_epiweek(date):
    """
    Function to capture the epidemiological year and week from the date
    """
    epiweek = Week.fromdate(date)
    return (epiweek.year, epiweek.week)


def get_ci_columns(p):
    """
    Function that given the confidence interval return the columns names

    Parameters
    -----------
    p: NDArray[np.float64]
    percentile values

    Returns
    --------
    List of columns name
    """

    columns = []

    for value in p:
        if value < 0.5:
            columns.append(f"lower_{int((1 - 2*value) * 100)}")
        elif value == 0.5:
            columns.append("pred")
        else:
            columns.append(f"upper_{int(2*value * 100)-100}")

    return columns


class Ensemble:
    """
    A class to compute the weights and apply the ensemble of multiple models.

    Attributes
    ------------
    df : pd.DataFrame
        Processed DataFrame containing model predictions.
    dist : str
        The distribution type used for modeling ('log_normal' or 'normal').
    order_models : list
        List of models in a specific order for weight computation.

    Methods
    ---------
    compute_weights(df_obs: pd.DataFrame, metric: str = 'crps')
        Computes the weights for the ensemble based on observed data and a specified metric.

    apply_ensemble(weights: dict = None)
        Computes the final ensemble distribution using either precomputed or provided weights.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        order_models: list,
        mixture: str = "log",
        dist: str = "log_normal",
        fn_loss: str = "median",
        conf_level: float = 0.9,
    ):
        """
        Initializes the Ensemble class by processing the input DataFrame and defining key attributes.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns `date`, `pred`, `lower`, `upper`, and `model_id`.
        order_models : list
            List defining the order of models for weight computation.
        mixture: str
            Determine how the predictions are combined. Choose `linear` for a weighted
            linear mixture or `log` for logarithmic pooling.
        dist : str, optional
            The distribution type used for parameterizing predictions ('log_normal' or 'normal'). Default is 'log_normal'.
        fn_loss : str, optional
            Loss function used for estimation ('median' or 'lower'). Default is 'median'.
        conf_level : float, optional, default=0.9
            Confidence level used for computing the confidence intervals.

        Raises
        ------
        ValueError
            If the input DataFrame does not contain the required columns.
        """

        try:
            df = df[
                [
                    "date",
                    "pred",
                    f"lower_{int(100*conf_level)}",
                    f"upper_{int(100*conf_level)}",
                    "model_id",
                ]
            ]

        except:
            raise ValueError(
                f"The input dataframe must contain the columns: 'date', 'pred', 'lower_{int(100*conf_level)}', 'upper_{int(100*conf_level)}', 'model_id'"
            )

        df = get_df_pars(
            df.copy(), conf_level=conf_level, dist=dist, fn_loss=fn_loss
        )

        # organize the dataframe:
        df["model_id"] = pd.Categorical(
            df["model_id"], categories=order_models, ordered=True
        )
        df = df.sort_values(by=["model_id", "date"])

        self.df = df
        self.dist = dist
        self.mixture = mixture
        self.order_models = order_models

    def compute_weights(
        self,
        df_obs: pd.DataFrame,
        metric: str = "crps",
        bounds: tuple = (-100, 100),
    ) -> dict:
        """
        Computes the optimal weights for the ensemble based on observed data and a specified metric.

        Parameters
        ------------
        df_obs : pd.DataFrame
            DataFrame containing observed values with columns `date` and `casos`.
        metric : str, optional
            Scoring metric used for optimization. Options: ['crps', 'log_score']. Default is 'crps'.
        bounds: tuple
            Tuple where the first element represents the minimum value and the second
            represents the maximum value for the bounds.

        Returns
        -------
        dict
            Dictionary containing the computed weights for each model and the loss value.
        """

        preds = self.df[["date", "mu", "sigma", "model_id"]]

        if self.mixture == "linear":
            weights = find_opt_weights_linear(
                df_obs,
                preds,
                self.order_models,
                dist=self.dist,
                metric=metric,
                bounds=bounds,
            )

        if self.mixture == "log":
            weights = find_opt_weights_log(
                obs=df_obs,
                preds=preds,
                order_models=self.order_models,
                dist=self.dist,
                metric=metric,
                bounds=bounds,
            )

        self.weights = weights

        return weights

    def apply_ensemble(
        self,
        weights: Union[None, NDArray[np.float64]] = None,
        p: NDArray[np.float64] = np.array(
            [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        ),
    ) -> pd.DataFrame:
        """
        Computes the final ensemble distribution using either precomputed or user-provided weights.

        Parameters
        ----------
        weights : np.array
            Array containing weights for each model. If None, uses precomputed weights.

        p: np.array
            Returned percentile values

        Returns
        -------
        pd.DataFrame
            DataFrame containing the ensemble predictions with quantiles (`pred`, `lower`, `upper`).
        """

        if weights is None:
            try:
                weights = self.weights["weights"]
            except:
                raise ValueError(
                    "Weights must be computed first using `compute_weights`, or provided explicitly."
                )

        weights = cast(NDArray[np.float64], weights)

        columns = get_ci_columns(p)

        preds = self.df

        df_for = pd.DataFrame()

        for d in preds.date.unique():
            preds_ = preds.loc[preds.date == d]

            if self.mixture == "log":
                quantiles = get_quantiles_log(
                    self.dist,
                    weights=weights,
                    ms=preds_.mu,
                    vs=preds_.sigma**2,
                    p=p,
                )

            if self.mixture == "linear":
                quantiles = get_quantiles_linear(
                    self.dist, weights=weights, preds=preds_, p=p
                )

            df_ = pd.DataFrame([quantiles], columns=columns)

            df_["date"] = d

            df_for = pd.concat([df_for, df_], axis=0).reset_index(drop=True)

        df_for.date = pd.to_datetime(df_for.date)

        return df_for


def dlnorm_mix(
    obs: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    weights: NDArray[np.float64],
    log=False,
) -> float:
    """
    Compute the PDF or log-PDF of a mixture of lognormal distributions for omega values.

    Parameters
    ------------
        obs: np.array or float
            Values where the mixture density is evaluated. Can be a single value or an array.
        mu: np.array
            Mu parameter (in log-space) for the lognormal components.
        sigma: np.array
            Standard deviations (in log-space) for the lognormal components.
        weight: array-like
            Mixture weights (must sum to 1).
        log: bool
            Whether to return the log-density.

    Returns
    ------------
        array: The mixture density or log-density evaluated at obs.
    """
    obs = np.atleast_1d(obs)  # Ensure `obs` is an array
    lw = np.log(weights)  # Log of weights
    K = len(mu)  # Number of components

    if len(sigma) != K or len(weights) != K:
        raise ValueError("mu, sigma, and weights must have the same length")

    # Compute log-PDFs for each component in a vectorized manner
    ldens = np.array(
        [
            lognorm.logpdf(obs, s=sigma[i], scale=np.exp(mu[i]))
            for i in range(K)
        ]
    ).T  # Transpose to align with obs dimensions

    # Combine using logsumexp for numerical stability
    if log:
        ans = logsumexp(lw + ldens, axis=1)
    else:
        ans = np.exp(logsumexp(lw + ldens, axis=1))

    return (
        ans if ans.size > 1 else ans.item()
    )  # Return scalar if input was scalar


def compute_ppf(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    weights: NDArray[np.float64],
    p: NDArray[np.float64] = np.array([0.5, 0.05, 0.95]),
) -> NDArray[np.float64]:
    """
    Compute the Percent-Point Function (PPF), which is the inverse of the CDF,
    for a mixture of lognormal distributions.

    The function takes the parameters of a lognormal mixture (mean, standard deviation, and weights)
    and returns the mixture values for the 5th, 50th, and 95th percentiles.

    Parameters
    ------------
        mu: np.array
            Mean values (in log-space) for the lognormal components of the mixture.
        sigma: np.array
            Standard deviation values (in log-space) for the lognormal components of the mixture.
        weights: np.array
            Weights of each component in the lognormal mixture. These should sum to 1.

    Returns
    ---------
        np.array
        The x-values corresponding to the 5th, 50th, and 95th percentiles.
    """
    x = np.linspace(1e-6, 10**5, 10**5)

    pdf_values = dlnorm_mix(x, mu, sigma, weights, log=False)

    # Normalize the PDF using the trapezoidal rule
    dx = np.diff(x)  # Compute spacing between consecutive x-values
    dx = np.append(dx, dx[-1])  # Ensure length matches the x array
    area = np.sum(pdf_values * dx)  # Approximate the area under the PDF
    pdf_values_normalized = (
        pdf_values / area
    )  # Normalize the PDF to ensure total area is 1

    cdf_values = cumulative_trapezoid(pdf_values_normalized, x, initial=0)

    # Invert the CDF to obtain the PPF
    ppf_function = interp1d(
        cdf_values, x, bounds_error=False, fill_value="extrapolate"
    )

    x_for_p = ppf_function(
        p
    )  # Get x-values corresponding to the probabilities

    return x_for_p


def crps_lognormal_mix(
    obs: Union[float, NDArray[np.float64]],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """
    Compute the score of a mix of lognormal distributions.

    Parameters
    ------------
        obs: np.array or float
            Values where the mixture score is evaluated.
        mu: np.array
            Mu parameter (in log-space) for the lognormal components.
        sigma: np.array
            Standard deviations (in log-space) for the lognormal components.
        weight: array-like
            Mixture weights (must sum to 1).

    Returns
    ------------
        float
        The score evaluated.
    """
    K = len(mu)

    if len(sigma) != K:
        print("mu and sigma should be the same lenght")

    crpsdens = list(np.zeros(K))

    for i in np.arange(K):

        crpsdens[i] = crps_lognormal(
            observation=obs, mulog=mu[i], sigmalog=sigma[i]
        )

    return np.dot(np.array(weights), np.array(crpsdens))  # , crpsdens


def find_opt_weights_linear(
    obs: pd.DataFrame,
    preds: pd.DataFrame,
    order_models: list,
    dist: str,
    metric: str,
    bounds: tuple = (-100, 100),
) -> dict:
    """
    Find the weights of a linear mix distributions that minimizes the metric selected.

    Parameters
    -----------
    obs: pd.Dataframe
        Dataframe with the columns: `date` and `casos`
    preds: pd.Dataframe
        Dataframe with the columns: `date`, `mu`, `sigma`, `model_id`
    order_models : list
        List defining the order of models for weight computation.
    dist : str, optional
        The distribution type used for parameterizing predictions ('log_normal' or 'normal'). Default is 'log_normal'.
    metric : str, optional
        Metric used for optimization. Options: `crps`, `log_score`.
    bounds: tuple
        Tuple where the first element represents the minimum value and the second
        represents the maximum value for the bounds.

    Return
    -------
    dict
        A dictionary containing:
        - `weights`: The optimized weights for the models.
        - `loss`: The minimized loss value based on the selected metric.
    """

    if dist == "log_normal":
        weights = find_opt_weights_linear_mix_log(
            obs, preds, order_models, metric=metric, bounds=bounds
        )

    if dist == "normal":
        weights = find_opt_weights_linear_mix_norm(
            obs, preds, order_models, metric=metric, bounds=bounds
        )

    return weights


def find_opt_weights_linear_mix_log(
    obs: pd.DataFrame,
    preds: pd.DataFrame,
    order_models: list,
    metric: str,
    bounds: tuple,
) -> dict:
    """
    Find the weights of a lognormal linear mix distributions that minimizes the metric selected.

    Parameters
    -----------
    obs: pd.Dataframe
        Dataframe with the columns: `date` and `casos`
    preds: pd.Dataframe
        Dataframe with the columns: `date`, `mu`, `sigma`, `model_id`
    order_models: list
        Order of the different models in the model_id column
    metric: str ['crps', 'log_score']
        Metric used to optimize the weights
    bounds: tuple
        Tuple where the first element represents the minimum value and the second
        represents the maximum value for the bounds.

    Return
    -------
    dict
        A dictionary containing:
        - `weights`: The optimized weights for the models.
        - `loss`: The minimized loss value based on the selected metric.
    """
    K = len(order_models)

    def loss(eta):
        """
        Computes the loss function based on the selected metric.

        Parameters
        ----------
        eta : array-like
            Parameterization of the weights, transformed via `alpha_01`.

        Returns
        -------
        float
            The computed loss value.
        """
        ws = alpha_01(eta)
        ws = np.where(ws < 1e-6, 1e-6, ws)

        score = 0
        for date in obs.date:
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.drop(["date"], axis=1).reset_index(drop=True)

            if metric == "log_score":
                score = score - dlnorm_mix(
                    obs.loc[obs.date == date].casos,
                    preds_["mu"].to_numpy(),
                    preds_["sigma"].to_numpy(),
                    weights=ws,
                    log=True,
                )

            if metric == "crps":
                score = score + crps_lognormal_mix(
                    obs.loc[obs.date == date].casos,
                    preds_["mu"].to_numpy(),
                    preds_["sigma"].to_numpy(),
                    weights=ws,
                )

        return score

    initial_guess = np.random.normal(size=K - 1)
    bounds_ = [bounds] * (K - 1)
    opt_result = minimize(
        loss, initial_guess, method="Nelder-mead", bounds=bounds_
    )

    optimal_weights = alpha_01(opt_result.x)

    return {"weights": optimal_weights, "loss": opt_result.fun}


def find_opt_weights_linear_mix_norm(
    obs: pd.DataFrame,
    preds: pd.DataFrame,
    order_models: list,
    metric: str,
    bounds: tuple,
) -> dict:
    """
    Find the weights of a lognormal linear mix distributions that minimizes the metric selected.

    Parameters
    -----------
    obs: pd.Dataframe
        Dataframe with the columns: `date` and `casos`
    preds: pd.Dataframe
        Dataframe with the columns: `date`, `mu`, `sigma`, `model_id`
    order_models: list
        Order of the different models in the model_id column
    metric: str ['crps', 'log_score']
        Metric used to optimize the weights
    bounds: tuple
        Tuple where the first element represents the minimum value and the second
        represents the maximum value for the bounds.

    Return
    -------
    dict
        A dictionary containing:
        - `weights`: The optimized weights for the models.
        - `loss`: The minimized loss value based on the selected metric.
    """
    K = len(order_models)

    def loss(eta):
        """
        Computes the loss function based on the selected metric.

        Parameters
        ----------
        eta : array-like
            Parameterization of the weights, transformed via `alpha_01`.

        Returns
        -------
        float
            The computed loss value.
        """
        ws = alpha_01(eta)
        ws = np.where(ws < 1e-6, 1e-6, ws)

        score = 0
        for date in obs.date:
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.drop(["date"], axis=1).reset_index(drop=True)

            ms = preds_["mu"]
            vs = preds_.sigma**2

            if not len(ms) == len(vs) == K:
                raise ValueError("n_models and vs are not the same size!")

            mu, sd = linear_mix(weights=ws, ms=ms, vs=vs)

            score = score + get_score(
                obs=obs.loc[obs.date == date].casos,
                mu=mu,
                sd=sd,
                dist="norm",
                metric=metric,
            )

        return score

    initial_guess = np.random.normal(size=K - 1)
    bounds_ = [bounds] * (K - 1)
    opt_result = minimize(
        loss, initial_guess, method="Nelder-mead", bounds=bounds_
    )

    optimal_weights = alpha_01(opt_result.x)

    return {"weights": optimal_weights, "loss": opt_result.fun}


def get_quantiles_log(
    dist: str,
    weights: NDArray[np.float64],
    ms: NDArray[np.float64],
    vs: NDArray[np.float64],
    p: NDArray[np.float64] = np.array([0.5, 0.05, 0.95]),
):
    """
    Function to get the quantiles of a logarithmic pooling.

    Parameters
    ------------
    dist : str, optional
        The distribution type used for parameterizing predictions ('log_normal' or 'normal'). Default is 'log_normal'.
    weights: np.array
        The weights assigned to each prediction.
    ms: np.array
        The mu parameter of each prediction.
    vs: np.array
        The variance parameter of each prediction.
    p: np.array
        Returned percentile values

    Returns
    --------
    quantiles: np.array
        The quantiles obtained according to p.
    """
    pool = pool_par_gauss(alpha=weights, m=ms, v=vs)

    if dist == "log_normal":
        quantiles = lognorm.ppf(p, s=pool[1], scale=np.exp(pool[0]))

    elif dist == "normal":
        quantiles = norm.ppf(p, loc=pool[0], scale=pool[1])

    return quantiles


def get_quantiles_linear(
    dist: str,
    weights: NDArray[np.float64],
    preds: pd.DataFrame,
    p: NDArray[np.float64] = np.array([0.5, 0.05, 0.95]),
):
    """
    Function to get the quantiles of the linear mixture.

    Parameters
    ------------
    dist : str, optional
        The distribution type used for parameterizing predictions ('log_normal' or 'normal'). Default is 'log_normal'.
    weights: np.array
        The weights assigned to each prediction.
    preds: pd.DataFrame
        The Dataframe with the predictions.
    p: np.array
        Returned percentile values

    Returns
    --------
    quantiles: np.array
        The quantiles obtained according to p.
    """

    weights = np.where(weights < 1e-6, 1e-6, weights)

    if dist == "normal":
        pool = linear_mix(weights=weights, ms=preds.mu, vs=preds.sigma**2)

        quantiles = norm.ppf(p, loc=pool[0], scale=pool[1])

    if dist == "log_normal":
        quantiles = compute_ppf(
            mu=preds["mu"].values,
            sigma=preds["sigma"].values,
            weights=weights,
            p=p,
        )

    return quantiles
