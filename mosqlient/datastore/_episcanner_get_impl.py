__all__ = ["get_episcanner"]


from typing import Optional
from datetime import date

import pandas as pd

from mosqlient import types
from .models import Episcanner


def get_episcanner(
    api_key: str | None = None,
    uf: Optional[types.UF] = None,
    disease: types.DiseaseName = "dengue",
    year: Optional[int] = date.today().year,
    adm_1: Optional[types.ADM1] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        disease : types.DiseaseName
            Default: "dengue". Accepts disease names or ICD-10 codes.
        uf : types.UF
            The Brazilian state abbreviation (e.g., 'SP', 'MG') or state IBGE
            code (e.g., 33). Defaults to 'RJ'.
        year : int, optional
            Default value is the current year.
        adm_1 : types.ADM1, optional
            Alias for `uf`. Accepts a state abbreviation or IBGE state code.

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

    uf = uf if uf is not None else adm_1
    if uf is None:
        uf = "RJ"

    return pd.DataFrame(
        Episcanner.get(
            api_key=api_key,
            disease=disease,
            year=year,
            uf=uf,
        )
    )
