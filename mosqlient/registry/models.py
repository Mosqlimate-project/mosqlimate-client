import asyncio
from typing import Any

from mosqlient.requests import get_all


def _params(**kwargs) -> dict[str, Any]:
    for k, v in kwargs.items():
        if v is None:
            del k

        if isinstance(v, (bool, int, float)):
            kwargs[k] = str(v)
        else:
            raise TypeError(f"Unkown value type for field {k}")

    return kwargs


class Model:
    def __init__(self):
        ...

    @classmethod
    def get(cls, **kwargs):
        cls._validate_fields(**kwargs)
        params = _params(**kwargs)

        async def fetch_models():
            return await get_all("registry", "models", params)
        return asyncio.run(fetch_models())

    @classmethod
    def post(cls):
        print("post")

    @staticmethod
    def _validate_fields(**kwargs) -> None:
        ModelFieldValidator(**kwargs)


class ModelFieldValidator:
    FIELDS = {
        "id": (int, str),
        "name": str,
        # "author_name": str,
        # "author_username": str,
        # "author_institution": str, move this to mosqlient.Client
        "repository": str,
        "implementation_language": str,
        "disease": str,
        "ADM_level": (str, int),
        "temporal": bool,
        "spatial": bool,
        "categorical": bool,
        "time_resolution": str,
    }
    DISEASES = ["dengue", "zika", "chikungunya"]
    ADM_LEVELS = [1, 2, 3, 4]
    TIME_RESOLITIONS = ["day", "week", "month", "year"]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            if not isinstance(v, self.FIELDS[k]):
                raise TypeError(
                    f"Field '{k}' must have instance of '{self.FIELDS[k]}'"
                )

            if k == "id":
                if v <= 0:
                    raise ValueError("Incorrect value for field 'id'")

            if k == "disease":
                if v == "chik":
                    v = "chikungunya"
                if v not in self.DISEASES:
                    raise ValueError(
                        f"Unkown 'disease'. Options: '{self.DISEASES}'"
                    )

            if k == "ADM_level":
                v = int(v)
                if v not in self.ADM_LEVELS:
                    raise ValueError(
                        f"Unkown 'ADM_level'. Options: '{self.ADM_LEVELS}'"
                    )

            if k == "time_resolution":
                if v not in self.TIME_RESOLITIONS:
                    raise ValueError(
                        "Unkown 'time_resolution'. Options: "
                        f"{self.TIME_RESOLITIONS}"
                    )
