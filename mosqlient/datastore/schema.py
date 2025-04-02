from typing import Optional, Literal

from mosqlient import types


class InfodengueSchema(types.Schema):
    pass


class ClimateSchema(types.Schema):
    pass


class InfodengueGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "datastore"
    endpoint: str = "infodengue"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None
    disease: Optional[types.Disease] = None
    uf: Optional[types.UF] = None
    geocode: Optional[types.Geocode] = None

    def params(self) -> dict:
        p = {
            "start": self.start,
            "end": self.end,
            "disease": self.disease,
            "uf": self.uf,
            "geocode": self.geocode,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}


class ClimateGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "datastore"
    endpoint: str = "climate"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
    start: Optional[types.Date] = None
    end: Optional[types.Date] = None
    geocode: Optional[types.Geocode] = None
    uf: Optional[types.UF] = None

    def params(self) -> dict:
        p = {
            "start": self.start,
            "end": self.end,
            "geocode": self.geocode,
            "uf": self.uf,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}
