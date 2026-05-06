"""Tests for the datastore schema module."""

import pytest
from datetime import date
from mosqlient.datastore.schema import (
    InfodengueSchema,
    ClimateSchema,
    ClimateWeeklySchema,
    EpiscannerSchema,
    MosquitoSchema,
    InfodengueGETParams,
    ClimateGETParams,
    ClimateWeeklyGETParams,
    EpiscannerGETParams,
    MosquitoGETParams,
)


class TestInfodengueSchema:
    def test_create(self):
        schema = InfodengueSchema()
        assert schema is not None


class TestClimateSchema:
    def test_create(self):
        schema = ClimateSchema()
        assert schema is not None


class TestClimateWeeklySchema:
    def test_create(self):
        schema = ClimateWeeklySchema()
        assert schema is not None


class TestEpiscannerSchema:
    def test_create(self):
        schema = EpiscannerSchema()
        assert schema is not None


class TestMosquitoSchema:
    def test_create(self):
        schema = MosquitoSchema()
        assert schema is not None


class TestInfodengueGETParams:
    def test_default_values(self):
        p = InfodengueGETParams()
        assert p.method == "GET"
        assert p.app == "datastore"
        assert p.endpoint == "infodengue"

    def test_params_with_filters(self):
        p = InfodengueGETParams(
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            disease="dengue",
            uf="SP",
            geocode=3550308,
        )
        params = p.params()
        assert params["disease"] == "dengue"
        assert params["uf"] == "SP"
        assert params["geocode"] == 3550308

    def test_params_filters_none(self):
        p = InfodengueGETParams()
        params = p.params()
        assert params == {}

    def test_params_with_page(self):
        p = InfodengueGETParams(page=1, per_page=10)
        params = p.params()
        assert params["page"] == 1
        assert params["per_page"] == 10


class TestClimateGETParams:
    def test_default_values(self):
        p = ClimateGETParams()
        assert p.method == "GET"
        assert p.app == "datastore"
        assert p.endpoint == "climate"

    def test_params_with_filters(self):
        p = ClimateGETParams(
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            uf="RJ",
            geocode=3304557,
        )
        params = p.params()
        assert params["uf"] == "RJ"
        assert params["geocode"] == 3304557

    def test_params_filters_none(self):
        p = ClimateGETParams()
        params = p.params()
        assert params == {}


class TestClimateWeeklyGETParams:
    def test_default_values(self):
        p = ClimateWeeklyGETParams(start="202401", end="202452")
        assert p.method == "GET"
        assert p.app == "datastore"
        assert p.endpoint == "climate/weekly"

    def test_params_with_filters(self):
        p = ClimateWeeklyGETParams(
            start="202401",
            end="202452",
            uf="SP",
            geocode=3550308,
            macro_health_code=1101,
        )
        params = p.params()
        assert params["start"] == "202401"
        assert params["end"] == "202452"
        assert params["uf"] == "SP"
        assert params["macro_health_code"] == 1101

    def test_params_required_fields(self):
        p = ClimateWeeklyGETParams(start="202401", end="202452")
        params = p.params()
        assert params["start"] == "202401"
        assert params["end"] == "202452"


class TestEpiscannerGETParams:
    def test_default_values(self):
        p = EpiscannerGETParams(disease="dengue", uf="SP")
        assert p.method == "GET"
        assert p.app == "datastore"
        assert p.endpoint == "episcanner"

    def test_params_with_year(self):
        p = EpiscannerGETParams(disease="zika", uf="RJ", year=2023)
        params = p.params()
        assert params["disease"] == "zika"
        assert params["uf"] == "RJ"
        assert params["year"] == 2023

    def test_params_without_year(self):
        p = EpiscannerGETParams(disease="dengue", uf="SP")
        params = p.params()
        assert params["disease"] == "dengue"
        assert "year" not in params


class TestMosquitoGETParams:
    def test_default_values(self):
        p = MosquitoGETParams()
        assert p.method == "GET"
        assert p.app == "datastore"
        assert p.endpoint == "mosquito"

    def test_params_with_filters(self):
        p = MosquitoGETParams(
            date_start=date(2024, 1, 1),
            date_end=date(2024, 12, 31),
            state="MG",
            municipality="Ponta Porã",
            page_=1,
        )
        params = p.params()
        assert params["state"] == "MG"
        assert params["municipality"] == "Ponta Porã"
        assert params["page"] == 1

    def test_params_filters_none(self):
        p = MosquitoGETParams()
        params = p.params()
        assert params == {}
