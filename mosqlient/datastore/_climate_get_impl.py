__all__ = ["get_climate", "get_climate_weekly"]


from typing import Optional
from datetime import date

import pandas as pd

from mosqlient import types
from .models import Climate, ClimateWeekly


def get_climate(
    api_key: str,
    start_date: date | str,
    end_date: date | str,
    uf: Optional[types.UF] = None,
    geocode: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve historical climate data from the Mosqlimate API for a specific region and date range.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        start_date : date or str
            Start date of the data range (as a `datetime.date` or ISO format string).
        end_date : date or str
            End date of the data range (as a `datetime.date` or ISO format string).
        uf : types.UF, optional
            The Brazilian state abbreviation (e.g., 'SP', 'MG'). If provided and `geocode` is not, filters by state.
        geocode : int, optional
            IBGE geocode of a municipality. If provided, overrides `uf`.

    Returns
    -------
        pandas.DataFrame
            DataFrame containing daily climate data. Detailed descriptions of each column in the DataFrame can be found in the official API documentation:
            https://api.mosqlimate.org/docs/datastore/GET/climate/#output_items

    Notes
    -----
    - Either `uf` or `geocode` must be provided to define the target location.

    Examples
    --------
    >>> get_climate(
    ...     api_key="your_api_key",
    ...     start_date="2023-01-01",
    ...     end_date="2023-01-31",
    ...     geocode=3550308
    ... )
    """

    return pd.DataFrame(
        Climate.get(
            api_key=api_key,
            start=start_date,
            end=end_date,
            uf=uf,
            geocode=geocode,
        )
    )


def get_climate_weekly(
    api_key: str,
    start: str,
    end: str,
    uf: Optional[types.UF] = None,
    geocode: Optional[int] = None,
    macro_health_code: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve historical climate data weekly aggregated

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        start : str
            Start epiweek in YYYYWW format (example: 202501).
        end : str
            End epiweek in YYYYWW format (example: 202501).
        uf : str, optional
            The Brazilian state abbreviation (e.g., 'SP', 'MG').
        geocode : int, optional
            IBGE geocode of a municipality.
        macro_health_code : int, optional
            Macro Health IBGE geocode. Example: 1101

    Returns
    -------
        pandas.DataFrame
            DataFrame containing daily climate data. Detailed descriptions of
            each column in the DataFrame can be found in the official API
            documentation:
            https://api.mosqlimate.org/docs/datastore/GET/climate-weekly/

    Notes
    -----
    - Either `uf` or `geocode` or `macro_health_code` must be provided.

    Examples
    --------
    >>> get_climate_weekly(
    ...     api_key="your_api_key",
    ...     start="202401",
    ...     end="202402",
    ...     geocode=3550308
    ... )
    """

    return pd.DataFrame(
        ClimateWeekly.get(
            api_key=api_key,
            start=start,
            end=end,
            uf=uf,
            geocode=geocode,
            macro_health_code=macro_health_code,
        )
    )
