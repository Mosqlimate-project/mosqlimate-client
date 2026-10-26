__all__ = ["get_mosquito"]


from typing import Optional

import pandas as pd

from mosqlient import types
from .models import Mosquito


def get_mosquito(
    api_key: str | None = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    state: Optional[types.UF] = None,
    municipality: Optional[str] = None,
    page: Optional[int] = None,
    uf: Optional[types.UF] = None,
    adm_1: Optional[types.ADM1] = None,
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
        state : str or int, optional
            State abbreviation (e.g., 'MG') or IBGE state code (e.g., 31).
        municipality : str, optional
            Name of the municipality. Example: "Ponta Porã"
        page : int, optional
        uf : str or int, optional
            Alias for `state`.
        adm_1 : types.ADM1, optional
            Alias for `state`. Accepts a state abbreviation or IBGE state code.

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

    state = state if state is not None else uf
    if state is None:
        state = adm_1

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
