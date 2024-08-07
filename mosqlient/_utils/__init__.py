from typing import Any
from datetime import date


def parse_params(**kwargs) -> dict[str, Any]:
    from mosqlient.registry.models import Base

    params = {}
    for k, v in kwargs.items():
        if isinstance(v, (bool, int, float, str, date, Base)):
            params[k] = str(v)
        elif v is None:
            continue
        else:
            raise TypeError(f"Unknown type f{type(v)}")

    return params
