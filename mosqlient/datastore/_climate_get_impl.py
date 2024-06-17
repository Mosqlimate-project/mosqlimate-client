__all__ = ["get_climate"]


from typing import Optional, Literal
from datetime import date

import pandas as pd

from .climate import Climate


def get_climate(
    start_date: date | str,
    end_date: date | str,
    uf: Optional[Literal[
        "AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG",
        "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP",
        "SE", "TO", "DF"
    ]] = None,
    geocode: Optional[int] = None
) -> pd.DataFrame:
    params = {
        "uf": uf,
        "geocode": geocode
    }
    return pd.DataFrame(Climate.get(start=start_date, end=end_date, **params))
