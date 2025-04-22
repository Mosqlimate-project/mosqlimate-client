__all__ = ["get_episcanner"]


from typing import Optional
from datetime import date

import pandas as pd

from mosqlient import types
from .models import Episcanner


def get_episcanner(
    api_key: str,
    uf: types.UF,
    disease: types.Disease = "dengue",
    year: Optional[int] = date.today().year,
) -> pd.DataFrame:
    """
    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        disease : types.Disease
            Default: "dengue". Options: dengue, chikungunya or zika
        uf : types.UF
            The Brazilian state abbreviation (e.g., 'SP', 'MG').
        year : int, optional
            Default value is the current year.

    Returns
    -------
        pandas.DataFrame
            DataFrame containing Episcanner data. Detailed
            descriptions of each column in the DataFrame can be found in the
            official API documentation:
            https://api.mosqlimate.org/docs/datastore/GET/episcanner/

    Examples
    --------
    >>> get_episcanner(
    ...     api_key="your_api_key",
    ...     uf="SP",
    ... )
    """

    return pd.DataFrame(
        Episcanner.get(
            api_key=api_key,
            disease=disease,
            year=year,
            uf=uf,
        )
    )
