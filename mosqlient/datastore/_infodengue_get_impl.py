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
