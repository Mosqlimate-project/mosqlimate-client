__all__ = ["get_vegetation"]


from typing import Optional
from datetime import date

import pandas as pd

from mosqlient import types
from .models import Vegetation


def get_vegetation(
    api_key: str | None = None,
    start_date: date | str = "2024-01-01",
    end_date: date | str = "2024-02-01",
    uf: Optional[types.UF] = None,
    geocode: Optional[int] = None,
    collection: Optional[str] = None,
    attribute: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retrieve historical vegetation metrics from the Mosqlimate API for a specific region and date range.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        start_date : date or str
            Start date of the data range (as a `datetime.date` or ISO format string).
        end_date : date or str
            End date of the data range (as a `datetime.date` or ISO format string).
        uf : types.UF, optional
            The Brazilian state abbreviation (e.g., 'SP', 'MG').
        geocode : int, optional
            IBGE geocode of a municipality. If provided, overrides `uf`.
        collection : str, optional
            Specific satellite dataset collection identifier (e.g., 'modis').
        attribute : str, optional
            Specific vegetation index metric or band name (e.g., 'ndvi', 'evi').

    Returns
    -------
        pandas.DataFrame
            DataFrame containing daily vegetation index data metrics. Detailed descriptions of each column in the DataFrame can be found in the official API documentation:
            https://api.mosqlimate.org/docs/datastore/GET/vegetation/#output_items

    Notes
    -----
    - Either `uf` or `geocode` must be provided to define the target location filtering parameters.

    Examples
    --------
    >>> get_vegetation(
    ...     api_key="your_api_key",
    ...     start_date="2024-01-01",
    ...     end_date="2024-02-01",
    ...     geocode=3304557,
    ...     attribute="ndvi"
    ... )
    """

    return pd.DataFrame(
        Vegetation.get(
            api_key=api_key,
            start=start_date,
            end=end_date,
            uf=uf,
            geocode=geocode,
            collection=collection,
            attribute=attribute,
        )
    )
