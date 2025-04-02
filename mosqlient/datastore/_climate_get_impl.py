__all__ = ["get_climate"]


from typing import Optional
from datetime import date

import pandas as pd

from mosqlient import types
from .models import Climate


def get_climate(
    api_key: str,
    start_date: date | str,
    end_date: date | str,
    uf: Optional[types.UF] = None,
    geocode: Optional[int] = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        Climate.get(
            api_key=api_key,
            start=start_date,
            end=end_date,
            uf=uf,
            geocode=geocode,
        )
    )
