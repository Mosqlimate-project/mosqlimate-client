from datetime import date
from typing import Literal, Optional

from mosqlient.datastore import schema
from mosqlient import types, Mosqlient


class Infodengue(types.Model):
    _schema: schema.InfodengueSchema

    @classmethod
    def get(
        cls,
        api_key: str,
        disease: Literal["dengue", "zika", "chikungunya"],
        start: str | date,
        end: str | date,
        uf: Optional[types.UF] = None,
        geocode: Optional[int] = None,
        page: Optional[int] = None,
    ):
        """
        datastore.schema.InfodengueGETParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.InfodengueGETParams(
            start=start,
            end=end,
            disease=disease,
            uf=uf,
            geocode=geocode,
            page=page,
        )
        return client.get(params=params)


class Climate(types.Model):
    _schema: schema.ClimateSchema

    @classmethod
    def get(
        cls,
        api_key: str,
        start: str | date,
        end: str | date,
        uf: Optional[str] = None,
        geocode: Optional[int] = None,
        page: Optional[int] = None,
    ):
        """
        datastore.schema.ClimateGETParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.ClimateGETParams(
            start=start, end=end, uf=uf, geocode=geocode, page=page
        )
        return client.get(params=params)


class ClimateWeekly(types.Model):
    _schema: schema.ClimateWeeklySchema

    @classmethod
    def get(
        cls,
        api_key: str,
        start: str,
        end: str,
        uf: Optional[str] = None,
        geocode: Optional[int] = None,
        macro_health_code: Optional[int] = None,
        page: Optional[int] = None,
    ):
        """
        datastore.schema.ClimateWeeklyGETParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.ClimateWeeklyGETParams(
            start=start,
            end=end,
            uf=uf,
            geocode=geocode,
            macro_health_code=macro_health_code,
            page=page,
        )
        return client.get(params=params)


class Episcanner(types.Model):
    _schema: schema.EpiscannerSchema

    @classmethod
    def get(
        cls,
        api_key: str,
        disease: str,
        uf: str,
        year: Optional[int] = date.today().year,
    ):
        """
        datastore.schema.EpiscannerGETParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.EpiscannerGETParams(disease=disease, uf=uf, year=year)
        return client.get(params=params)


class Mosquito(types.Model):
    _schema: schema.MosquitoSchema

    @classmethod
    def get(
        cls,
        api_key: str,
        date_start: Optional[types.Date] = None,
        date_end: Optional[types.Date] = None,
        state: Optional[types.UF] = None,
        municipality: Optional[str] = None,
        page: Optional[int] = None,
    ):
        """
        datastore.schema.MosquitoGETParams
        """
        client = Mosqlient(x_uid_key=api_key)
        params = schema.MosquitoGETParams(
            date_start=date_start,
            date_end=date_end,
            state=state,
            municipality=municipality,
            page_=page,
        )
        return client.get(params=params)
