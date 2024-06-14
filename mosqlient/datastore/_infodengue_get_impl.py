__all__ = ["get_infodengue"]

from typing import Optional, Literal
from datetime import date

from .infodengue import Infodengue


def get_infodengue(
    disease: Literal["dengue", "zika", "chikungunya"],
    start_date: date | str,
    end_date: date | str,
    uf: Optional[Literal[
        "AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG",
        "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP",
        "SE", "TO", "DF"
    ]] = None,
    geocode: Optional[int] = None
):
    params = {
        "disease": disease,
        "start": start_date,
        "end": end_date,
        "uf": uf,
        "geocode": geocode
    }
    return Infodengue.get(**params)
