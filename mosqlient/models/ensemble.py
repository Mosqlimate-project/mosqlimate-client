import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scoringrules import crps_lognormal, crps_normal,  interval_score  
from numpy.typing import NDArray
from typing import Union, cast 

def validate_df_preds(df_preds:pd.DataFrame):
    expected_columns = {'date', 'lower', 'pred', 'upper', 'model_id'}
    
    if not expected_columns.issubset(df_preds.columns):
        raise ValueError(f"df_preds must contain the following columns: {expected_columns}. "
                         f"Missing: {expected_columns - set(df_preds.columns)}")
        
def invlogit(y:float) -> float:

    return 1 / (1 + np.exp(-y))

def alpha_01(alpha_inv: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Function that maps from R^n to the open simplex.

    Parameters
    -----------
    alpha_inv: array of float

    Returns
    --------
    array
        Vector on the (n+1) open simplex.
    '''
    K = len(alpha_inv) + 1
    z = np.full(K-1, np.nan)  # Equivalent to rep(NA, K-1)
    alphas = np.zeros(K)      # Equivalent to rep(0, K)
    
    for k in range(K-1):
        z[k] = invlogit(alpha_inv[k] + np.log(1 / (K - (k+1))))
        alphas[k] = (1 - np.sum(alphas[:k])) * z[k]
    
    alphas[K-1] = 1 - np.sum(alphas[:-1])
    return alphas

def pool_par_gauss(alpha: NDArray[np.float64], m: NDArray[np.float64], v: NDArray[np.float64]) -> tuple:
    '''
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
    '''
    if not (len(alpha) == len(m) == len(v)):
        raise ValueError("The arrays 'alpha', 'm', and 'v' must have the same length.")
    
    ws = alpha / v
    vstar = 1 / np.sum(ws)
    mstar = np.sum(ws * m) * vstar
    return mstar, np.sqrt(vstar)
    
def get_lognormal_pars(med:float, lwr:float, upr:float, alpha:float=0.90, fn_loss:str = 'median')->tuple:
    '''
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
    alpha : float, optional, default=0.90
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
    '''

    if fn_loss not in {'median', 'lower'}:
        raise ValueError("Invalid value for fn_loss. Choose 'median' or 'lower'.")
    
    if any(x < 0 for x in [med, lwr, upr]):
        raise ValueError("med, lwr, and upr must be non-negative.")
    
    def loss_lower(theta):
        tent_qs = lognorm.ppf([(1 - alpha)/2, (1 + alpha)/2], s=theta[1], scale=np.exp(theta[0]))
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
        return attained_loss
    
    def loss_median(theta):
        tent_qs = lognorm.ppf([0.5, (1 + alpha)/2], s=theta[1], scale=np.exp(theta[0]))
        if  med == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(med - tent_qs[0]) / med + abs(upr - tent_qs[1]) / upr
        return attained_loss
    
    if med == 0:
        mustar = np.log(0.1)
    else: 
        mustar = np.log(med)

    if fn_loss == 'median':
        result = minimize(loss_median, x0=[mustar, 0.5], bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],method = "Nelder-mead", 
                          options={'xatol': 1e-6, 'fatol': 1e-6, 
                           'maxiter': 1000, 
                           'maxfev':1000})
    if fn_loss == 'lower':
            result = minimize(loss_lower, x0=[mustar, 0.5], bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],method = "Nelder-mead",
                            options={'xatol': 1e-8, 'fatol': 1e-8, 
                           'maxiter': 5000,
                           'maxfev':5000})

    return result.x

def get_normal_pars(med:float, lwr:float, upr:float, alpha:float=0.90, fn_loss = 'median')->tuple:
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
    alpha : float, optional, default=0.90
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
        tent_qs = norm.ppf([(1 - alpha)/2, (1 + alpha)/2], loc=theta[0],  scale=theta[1])
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
        return attained_loss
    
    def loss_median(theta):
        tent_qs = norm.ppf([0.5, (1 + alpha)/2], loc=theta[0],  scale=theta[1])
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(med - tent_qs[0]) / med + abs(upr - tent_qs[1]) / upr
        return attained_loss

    sigmastar = max((upr - lwr) / 4, 1e-4)

    if fn_loss == 'lower':
        result = minimize(loss_lower, x0=[med, sigmastar], bounds=[(-5 * abs(med), 5 * abs(med)), (0, 100000)],method = "Nelder-mead")

    if fn_loss == 'median':
        result = minimize(loss_median, x0=[med, sigmastar], bounds=[(-5 * abs(med), 5 * abs(med)), (0, 100000)],method = "Nelder-mead")

    return result.x

def linear_mix(weights:NDArray[np.float64], ms:NDArray[np.float64], vs:NDArray[np.float64]) ->tuple: 
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

    mu = np.dot(weights,ms)

    sd = np.sqrt(np.dot(weights**2,vs))

    return mu, sd


def get_df_pars(preds_: pd.DataFrame, alpha: float = 0.9, dist: str = 'log_normal', fn_loss: str = 'median', return_estimations: bool = False) -> pd.DataFrame:
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
    alpha : float, optional, default=0.9
        Confidence level used for computing the confidence intervals.
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

    if dist == 'log_normal':
        preds_[["mu", "sigma"]] = preds_.apply(lambda row: get_lognormal_pars(
            med=row["pred"], lwr=row["lower"], upr=row["upper"], fn_loss=fn_loss
        ), axis=1, result_type="expand")
    elif dist == 'normal': 
        preds_[["mu", "sigma"]] = preds_.apply(lambda row: get_normal_pars(
            med=row["pred"], lwr=row["lower"], upr=row["upper"],  fn_loss=fn_loss
        ), axis=1, result_type="expand")
    
    if not return_estimations:
        return preds_
    
    if dist == 'log_normal':
        theo_pred_df = preds_.apply(lambda row: lognorm.ppf(
            [0.5, (1 - alpha) / 2, (1 + alpha) / 2], s=row["sigma"], scale=np.exp(row["mu"])
        ), axis=1, result_type="expand")
    elif dist == 'normal':
        theo_pred_df = preds_.apply(lambda row: norm.ppf(
            [0.5, (1 - alpha) / 2, (1 + alpha) / 2], loc=row["mu"], scale=row["sigma"]
        ), axis=1, result_type="expand")
    
    theo_pred_df.columns = ["fit_med", "fit_lwr", "fit_upr"]
    preds_ = pd.concat([preds_, theo_pred_df], axis=1)
    
    return preds_

def get_score(obs:float, mu:float, sd:float,
              dist:str = 'log_normal', metric:str = 'crps') ->float:
    '''
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
    '''

    if metric == 'log_score':
        if dist == 'log_normal':
            return -lognorm.logpdf(obs, s=sd, scale=np.exp(mu))
                
        elif dist == 'normal':
            return -np.log(norm.pdf(obs,loc =mu, scale = sd))

    if metric == 'crps':
        if dist == 'log_normal':
            return crps_lognormal(observation = obs,
                                    mulog = mu, sigmalog = sd)

        elif dist == 'normal':
            return crps_normal(obs, mu, sd)

    raise ValueError(f"Invalid distribution '{dist}' and metric '{metric}'")


def find_opt_weights_log(obs:pd.DataFrame,
                     preds:pd.DataFrame,
                     order_models:list,
                     dist:str = 'log_normal', metric:str = 'crps') -> dict:
    '''
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

    Returns 
    --------
    dict
    The dict contains the keys:
    - weights: the optmize weights by the loss
    - loss: loss function value 
    '''

    K = len(order_models)

    def loss(eta):

        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)
            ms = preds_['mu']
            vs = preds_.sigma**2

            if not len(ms) == len(vs) == K:
                raise ValueError("n_models and vs are not the same size!")
            
            mu, sd = pool_par_gauss(alpha=ws, m=ms, v=vs) 

            score = score + get_score(obs=obs.loc[obs.date == date].casos,
                                      mu=mu, sd=sd,
                                      dist = dist,
                                      metric = metric)

            return score  

    initial_guess = np.random.normal(size=K-1)
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }

def get_epiweek(date):
    '''
    Function to capture the epidemiological year and week from the date 
    '''
    epiweek = Week.fromdate(date)
    return (epiweek.year, epiweek.week)


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
        order_models:list, 
        mixture:str = 'log', 
        dist: str = 'log_normal',
        fn_loss:str = 'median', 
        alpha:float=0.9

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
        alpha : float, optional, default=0.9
            Confidence level used for computing the confidence intervals.

        Raises
        ------
        ValueError
            If the input DataFrame does not contain the required columns.
        """

        try: 
            df = df[['date', 'pred', 'lower', 'upper', 'model_id']]

        except:
            raise ValueError(
                "The input dataframe must contain the columns: 'date', 'pred', 'lower', 'upper', 'model_id'"
            )
        
        df = get_df_pars(df, alpha = alpha, dist = dist, fn_loss=fn_loss)

        #organize the dataframe:
        df['model_id'] = pd.Categorical(df['model_id'], categories = order_models,ordered=True)
        df = df.sort_values(by = ['model_id','date'])

        self.df = df 
        self.dist = dist 
        self.mixture = mixture
        self.order_models = order_models
    
    def compute_weights(self, df_obs:pd.DataFrame, metric:str = 'crps') ->dict: 
        '''
        Computes the optimal weights for the ensemble based on observed data and a specified metric.

        Parameters 
        ------------
        df_obs : pd.DataFrame
            DataFrame containing observed values with columns `date` and `casos`.
        metric : str, optional
            Scoring metric used for optimization. Options: ['crps', 'log_score']. Default is 'crps'.

        Returns
        -------
        dict
            Dictionary containing the computed weights for each model and the loss value.
        '''

        preds = self.df[['date', 'mu', 'sigma','model_id']]

        if self.mixture == 'linear': 
            weights = find_opt_weights_linear(df_obs, preds, self.order_models, dist = self.dist, metric = metric)
            
        if self.mixture == 'log':
            weights = find_opt_weights_log(obs=df_obs,
                                       preds=preds,
                                       order_models=self.order_models,
                                       dist = self.dist,
                                       metric = metric)

        self.weights = weights 

        return  weights
    
    def apply_ensemble(self, weights:Union[None, NDArray[np.float64]]=None, 
                       p:NDArray[np.float64] = np.array([0.5, 0.05, 0.95])) -> pd.DataFrame: 
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
                weights = self.weights['weights']
            except: 
                raise ValueError("Weights must be computed first using `compute_weights`, or provided explicitly.")
         
        
        weights = cast(NDArray[np.float64], weights) 

        preds = self.df

        df_for = pd.DataFrame()

        for d in preds.date.unique():
            preds_ = preds.loc[preds.date == d]
            
            if self.mixture == 'log':
                quantiles = get_quantiles_log(self.dist,                   
                            weights = weights, 
                            ms= preds_.mu,
                            vs= preds_.sigma**2,
                            p=p)

            if self.mixture == 'linear':
                quantiles = get_quantiles_linear(self.dist,
                                                weights=weights, 
                                                preds= preds_,
                                                p = p)
            
            df_ = pd.DataFrame([quantiles], columns = ['pred', 'lower', 'upper'])
        
            df_['date'] = d
            
            df_for = pd.concat([df_for, df_], axis =0).reset_index(drop = True)

        df_for.date = pd.to_datetime(df_for.date)
        
        return df_for
    

def dlnorm_mix(obs:  NDArray[np.float64], 
               mu:  NDArray[np.float64],
               sigma: NDArray[np.float64],
               weights: NDArray[np.float64], log=False) -> float:
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
    ldens = np.array([
        lognorm.logpdf(obs, s=sigma[i], scale=np.exp(mu[i]))
        for i in range(K)
    ]).T  # Transpose to align with obs dimensions

    # Combine using logsumexp for numerical stability
    if log:
        ans = logsumexp(lw + ldens, axis=1)
    else:
        ans = np.exp(logsumexp(lw + ldens, axis=1))

    return ans if ans.size > 1 else ans.item()  # Return scalar if input was scalar


def compute_ppf(mu: NDArray[np.float64], sigma:NDArray[np.float64],
                 weights:NDArray[np.float64],
                p:NDArray[np.float64] = np.array([0.5, 0.05, 0.95])) -> NDArray[np.float64]:
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
    pdf_values_normalized = pdf_values / area  # Normalize the PDF to ensure total area is 1

    cdf_values = cumulative_trapezoid(pdf_values_normalized, x, initial=0)

    # Invert the CDF to obtain the PPF
    ppf_function = interp1d(cdf_values, x, bounds_error=False, fill_value="extrapolate")
    
    x_for_p = ppf_function(p)  # Get x-values corresponding to the probabilities

    return x_for_p


def crps_lognormal_mix(obs: Union[float, NDArray[np.float64]],
                       mu:NDArray[np.float64], sigma:NDArray[np.float64],
                       weights:NDArray[np.float64]) -> float:
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
        print('mu and sigma should be the same lenght')

    crpsdens = list(np.zeros(K))
    
    for i in np.arange(K):

        crpsdens[i] = crps_lognormal(observation = obs, mulog = mu[i], sigmalog = sigma[i])

    return np.dot(np.array(weights), np.array(crpsdens))#, crpsdens 

def find_opt_weights_linear(obs:pd.DataFrame, preds:pd.DataFrame, order_models:list, dist:str, metric:str) -> dict:
    '''
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
    
    Return
    -------
    dict
        A dictionary containing:
        - `weights`: The optimized weights for the models.
        - `loss`: The minimized loss value based on the selected metric. 
    '''

    if dist == 'log_normal':
        weights = find_opt_weights_linear_mix_log(obs, preds, order_models, metric=metric)
        
    if dist == 'normal':
        weights = find_opt_weights_linear_mix_norm(obs, preds, order_models, metric=metric)
        
    return weights

def find_opt_weights_linear_mix_log(obs:pd.DataFrame, preds:pd.DataFrame, order_models:list, metric:str) -> dict:
    '''
    Find the weights of a lognormal linear mix distributions that minimizes the metric selected.

    Parameters
    -----------
    obs: pd.Dataframe
        Dataframe with the columns: `date` and `casos`
    preds: pd.Dataframe
        Dataframe with the columns: `date`, `mu`, `sigma`, `model_id`
    
    Return
    -------
    dict
        A dictionary containing:
        - `weights`: The optimized weights for the models.
        - `loss`: The minimized loss value based on the selected metric. 
    '''
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
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)

            if metric == 'log_score':
                score = score - dlnorm_mix(obs.loc[obs.date == date].casos,
                                       preds_["mu"].to_numpy(), preds_["sigma"].to_numpy(), weights =ws, log = True)
            
            if metric == 'crps': 
                score = score + crps_lognormal_mix(obs.loc[obs.date == date].casos,
                                       preds_["mu"].to_numpy(), preds_["sigma"].to_numpy(), weights =ws)
        
        return score  
    
    initial_guess = np.random.normal(size=K-1)
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }

def find_opt_weights_linear_mix_norm(obs:pd.DataFrame, preds:pd.DataFrame, order_models:list, metric:str) -> dict:
    '''
    Find the weights of a lognormal linear mix distributions that minimizes the metric selected.

    Parameters
    -----------
    obs: pd.Dataframe
        Dataframe with the columns: `date` and `casos`
    preds: pd.Dataframe
        Dataframe with the columns: `date`, `mu`, `sigma`, `model_id`
    
    Return
    -------
    dict
        A dictionary containing:
        - `weights`: The optimized weights for the models.
        - `loss`: The minimized loss value based on the selected metric. 
    '''
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
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)

            ms = preds_['mu']
            vs = preds_.sigma**2

            if not len(ms) == len(vs) == K:
                raise ValueError("n_models and vs are not the same size!")
             
            mu, sd =  linear_mix(weights=ws, ms=ms, vs=vs)

            score = score + get_score(obs=obs.loc[obs.date == date].casos,
                                      mu=mu, sd=sd,
                                      dist = 'norm',
                                      metric = metric)

        return score  
    
    initial_guess = np.random.normal(size=K-1)
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }

def get_quantiles_log(dist:str, weights:NDArray[np.float64], 
                      ms: NDArray[np.float64],
                      vs:NDArray[np.float64],
                      p:NDArray[np.float64] = np.array([0.5, 0.05, 0.95])):
    '''
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
    '''
    pool = pool_par_gauss(alpha = weights, m = ms,
                    v = vs)
                    
    if dist == 'log_normal':
        quantiles = lognorm.ppf(p, s=pool[1], scale=np.exp(pool[0]))

    elif dist == 'normal':
        quantiles = norm.ppf(p, loc=pool[0], scale=pool[1])

    return quantiles 

def get_quantiles_linear(dist:str, weights:NDArray[np.float64], 
                      preds: pd.DataFrame,
                      p:NDArray[np.float64] = np.array([0.5, 0.05, 0.95])):
    '''
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
    '''

    weights = np.where(weights < 1e-6, 1e-6, weights)
        
    if dist =='normal':
        pool = linear_mix(weights = weights, ms = preds.mu,
                    vs = preds.sigma**2)
                
        quantiles = norm.ppf(p, loc=pool[0], scale=pool[1])
            
    if dist =='log_normal':
       quantiles = compute_ppf(mu = preds['mu'].values, sigma = preds['sigma'].values,
                                        weights = weights)
       
    return quantiles   