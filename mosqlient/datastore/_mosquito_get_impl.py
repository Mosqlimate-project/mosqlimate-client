__all__ = ["get_mosquito"]


from typing import Optional

import pandas as pd

from .models import Mosquito


def get_mosquito(
    api_key: str,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    state: Optional[str] = None,
    municipality: Optional[str] = None,
    page: Optional[int] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        date_start : str, optional
            Format: YYYY-MM-dd
        date_end : str, optional
            Format: YYYY-MM-dd
        state : str, optional
            Example: MG
        municipality : str, optional
            Name of the municipality. Example: "Ponta Porã"
        page : int, optional

    Returns
    -------
        pandas.DataFrame
            DataFrame containing ContaOvos data. Detailed
            descriptions of each column in the DataFrame can be found in the
            official API documentation:
            https://api.mosqlimate.org/docs/datastore/GET/mosquito/

    Examples
    --------
    >>> get_mosquito(
    ...     api_key="your_api_key",
    ...     date_start="2024-01-01",
    ...     date_end="2024-12-31",
    ...     municipality="Ponta Porã",
    ... )
    """

    return pd.DataFrame(
        Mosquito.get(
            api_key=api_key,
            date_start=date_start,
            date_end=date_end,
            state=state,
            municipality=municipality,
            page=page,
        )
    )
