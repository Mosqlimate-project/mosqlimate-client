__all__ = ["get_infodengue"]

from typing import Optional, Literal
from datetime import date

import pandas as pd

from .models import Infodengue


def get_infodengue(
    api_key: str,
    disease: Literal["dengue", "zika", "chikungunya"],
    start_date: date | str,
    end_date: date | str,
    # fmt: off
    uf: Optional[
        Literal[
            "AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS",
            "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR",
            "SC", "SP", "SE", "TO", "DF",
        ]
    ] = None,
    # fmt: on
    geocode: Optional[int] = None,
) -> pd.DataFrame:
    params = {"uf": uf, "geocode": geocode}

    return pd.DataFrame(
        Infodengue.get(
            api_key=api_key,
            disease=disease,
            start=start_date,
            end=end_date,
            **params
        )
    )
