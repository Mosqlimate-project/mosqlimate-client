from typing import Optional, Literal

from mosqlient import types


class InfodengueSchema(types.Schema):
    pass


class ClimateSchema(types.Schema):
    pass


class ClimateWeeklySchema(types.Schema):
    pass


class EpiscannerSchema(types.Schema):
    pass


class MosquitoSchema(types.Schema):
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


class ClimateWeeklyGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "datastore"
    endpoint: str = "climate/weekly"
    page: Optional[int] = None
    per_page: Optional[int] = None
    #
    start: str
    end: str
    uf: Optional[types.UF] = None
    geocode: Optional[types.Geocode] = None
    macro_health_code: Optional[types.MacroHealthGeocode] = None

    def params(self) -> dict:
        p = {
            "start": self.start,
            "end": self.end,
            "uf": self.uf,
            "geocode": self.geocode,
            "macro_health_code": self.macro_health_code,
            "page": self.page,
            "per_page": self.per_page,
        }
        return {k: v for k, v in p.items() if v is not None}


class EpiscannerGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "datastore"
    endpoint: str = "episcanner"
    #
    disease: types.Disease
    uf: types.UF
    year: Optional[int] = None

    def params(self) -> dict:
        p = {
            "disease": self.disease,
            "uf": self.uf,
            "year": self.year,
        }
        return {k: v for k, v in p.items() if v is not None}


class MosquitoGETParams(types.Params):
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    app: types.APP = "datastore"
    endpoint: str = "mosquito"
    #
    date_start: Optional[types.Date] = None
    date_end: Optional[types.Date] = None
    state: Optional[types.UF] = None
    municipality: Optional[str] = None
    page_: Optional[int] = None

    def params(self) -> dict:
        p = {
            "date_start": self.date_start,
            "date_end": self.date_end,
            "state": self.state,
            "municipality": self.municipality,
            "page": self.page_,
        }
        return {k: v for k, v in p.items() if v is not None}
