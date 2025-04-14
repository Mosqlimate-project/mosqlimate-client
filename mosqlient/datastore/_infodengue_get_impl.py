__all__ = ["get_infodengue"]

from typing import Optional, Literal
from datetime import date

import pandas as pd

from mosqlient import types
from .models import Infodengue


def get_infodengue(
    api_key: str,
    disease: Literal["dengue", "zika", "chikungunya"],
    start_date: date | str,
    end_date: date | str,
    uf: Optional[types.UF] = None,
    geocode: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch InfoDengue Data from Mosqlimate API for dengue, zika, or chikungunya.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        disease : {'dengue', 'zika', 'chikungunya'}
            The arbovirus to retrieve data for.
        start_date : date or str
            Start date of the data range (as a `datetime.date` or ISO format string).
        end_date : date or str
            End date of the data range (as a `datetime.date` or ISO format string).
        uf : types.UF, optional
            The Brazilian state abbreviation (e.g., 'SP', 'RJ'). If provided without `geocode`, filters by state.
        geocode : int, optional
            IBGE geocode of a municipality. If provided, overrides `uf`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the epidemiological time series data for the specified region and time period.
        Detailed descriptions of each column in the DataFrame can be found in the official API documentation:
        https://api.mosqlimate.org/docs/datastore/GET/infodengue/

    Notes
    -----
    Either `uf` or `geocode` must be provided to specify the target geographic area.

    Examples
    --------
    >>> get_infodengue(
    ...     api_key="your_api_key",
    ...     disease="dengue",
    ...     start_date="2023-01-01",
    ...     end_date="2023-03-01",
    ...     uf="RJ"
    ... )
    """

    return pd.DataFrame(
        Infodengue.get(
            api_key=api_key,
            disease=disease,
            start=start_date,
            end=end_date,
            uf=uf,
            geocode=geocode,
        )
    )
