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
