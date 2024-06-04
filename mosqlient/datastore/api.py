from datetime import datetime
from typing import Any

import pandas as pd

from mosqlient.requests import get_datastore


def validate_date(date_str: str, par_name: str) -> None:
    try:
        # Validate the date format
        valid_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format for '{par_name}', should be YYYY-mm-dd")


def _params(**kwargs) -> dict[str, Any]:
    params = {}
    for k, v in kwargs.items():
        if isinstance(v, (bool, int, float, str)):
            params[k] = str(v)
        elif v is None:
            continue
        else:
            raise TypeError(f"Unknown type f{type(v)}")

    return params


class Infodengue:
    @classmethod
    def get(cls, **kwargs):
        cls._validate_fields(**kwargs)
        params = _params(**kwargs)

        gen_data = get_datastore("datastore", "infodengue", params)

        return pd.DataFrame(list(gen_data))

    @staticmethod
    def _validate_fields(**kwargs) -> None:
        DataFieldValidator(**kwargs)


class Climate:
    @classmethod
    def get(cls, **kwargs):
        cls._validate_fields(**kwargs)
        params = _params(**kwargs)

        gen_data = get_datastore("datastore", "climate", params)

        return pd.DataFrame(list(gen_data))

    @staticmethod
    def _validate_fields(**kwargs) -> None:
        DataFieldValidator(**kwargs)


class DataFieldValidator:
    FIELDS = {
        "per_page": int,
        "disease": str,
        "start": str,
        "end": str,
        "uf": str,
        "geocode": int,
    }
    DISEASES = ["dengue", "zika", "chikungunya"]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if not isinstance(v, self.FIELDS[k]):
                raise TypeError(f"Field '{k}' must have instance of " f"{' or '.join(self.FIELDS[k])}")

            if k == "disease":
                if v == "chik":
                    v = "chikungunya"

                if v not in self.DISEASES:
                    raise ValueError(f"Unkown 'disease'. Options: {self.DISEASES}")

            if (k == "start") | (k == "end"):
                validate_date(v, k)

            if k == "uf":
                if len(v) != 2:
                    raise ValueError("Invalid 'uf' parameter. It should be a two-letter value")

            if k == "geocode":
                if len(str(v)) != 7:
                    raise ValueError("Invalid 'geocode' parameter. It should be a seven numbers code")
