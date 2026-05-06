"""Tests for the registry schema module."""

import pytest
from datetime import date, timedelta
from pydantic import ValidationError
from mosqlient.registry.schema import (
    Model,
    PredictionDataRow,
    Prediction,
    ModelGETParams,
    PredictionGETParams,
    PredictionPOSTParams,
    PredictionDELETEParams,
    PredictionPublishPATCHParams,
    PredictionDataGETParams,
)


class TestModelSchema:
    def test_create_model(self, sample_model_schema):
        assert sample_model_schema.id == 1
        assert sample_model_schema.repository == "owner/repo"
        assert sample_model_schema.description == "Test model"
        assert sample_model_schema.category == "quantitative"
        assert sample_model_schema.time_resolution == "week"
        assert sample_model_schema.imdc_year is None
        assert sample_model_schema.predictions_count == 5
        assert sample_model_schema.active is True

    def test_model_with_imdc_year(self, sample_model_schema):
        sample_model_schema.imdc_year = 2024
        assert sample_model_schema.imdc_year == 2024

    def test_model_default_description(self):
        m = Model(
            id=1,
            repository="owner/repo",
            category="quantitative",
            time_resolution="week",
            predictions_count=0,
            active=True,
            created_at=date(2023, 1, 1),
            last_update=date(2023, 1, 1),
        )
        assert m.description == ""


class TestPredictionDataRow:
    def test_create_row(self, sample_prediction_data_row):
        row = PredictionDataRow(**sample_prediction_data_row)
        assert row.pred == 50.0
        assert row.lower_95 == 10.0
        assert row.upper_95 == 90.0

    def test_serialize_date(self, sample_prediction_data_row):
        row = PredictionDataRow(**sample_prediction_data_row)
        assert row.serialize_date(row.date, None) == "2023-01-01"

    def test_serialize_date_none(self, sample_prediction_data_row):
        row = PredictionDataRow(**sample_prediction_data_row)
        assert row.serialize_date(None, None) is None

    def test_dict_method(self, sample_prediction_data_row):
        row = PredictionDataRow(**sample_prediction_data_row)
        d = row.dict()
        assert d["date"] == "2023-01-01"
        assert d["pred"] == 50.0

    def test_model_dump(self, sample_prediction_data_row):
        row = PredictionDataRow(**sample_prediction_data_row)
        d = row.model_dump()
        assert "date" in d
        assert "pred" in d

    def test_validate_bounds_full(self, sample_prediction_data_row):
        row = PredictionDataRow(**sample_prediction_data_row)
        assert row.pred == 50.0

    def test_validate_bounds_partial(self):
        row = PredictionDataRow(
            date=date(2023, 1, 1),
            lower_90=20.0,
            pred=50.0,
            upper_90=80.0,
        )
        assert row.pred == 50.0

    def test_validate_bounds_invalid_order(self):
        pass

    def test_validate_bounds_negative(self):
        with pytest.raises(
            ValueError, match="Prediction bounds are not in the correct order"
        ):
            PredictionDataRow(
                date=date(2023, 1, 1),
                lower_90=-10.0,
                pred=50.0,
                upper_90=80.0,
            )

    def test_validate_partial_bounds_invalid(self):
        with pytest.raises(
            ValueError, match="Prediction bounds are not in the correct order"
        ):
            PredictionDataRow(
                date=date(2023, 1, 1),
                lower_90=60.0,
                pred=50.0,
                upper_90=80.0,
            )


class TestPredictionSchema:
    def test_create_prediction(self, sample_prediction_schema):
        assert sample_prediction_schema.id == 1
        assert sample_prediction_schema.disease == "A90"
        assert sample_prediction_schema.published is True
        assert sample_prediction_schema.adm_level == 1
        assert sample_prediction_schema.adm_0 == "BRA"
        assert sample_prediction_schema.adm_1 == 33

    def test_validate_adm_level_0(self, sample_model_schema):
        p = Prediction(
            id=1,
            model=sample_model_schema,
            disease="A90",
            commit="a" * 40,
            description="Test",
            case_definition="probable",
            published=True,
            created_at=date(2023, 1, 1),
            adm_level=0,
            adm_0="BRA",
        )
        assert p.adm_0 == "BRA"

    def test_validate_adm_level_1_missing_adm_1(self, sample_model_schema):
        with pytest.raises(ValueError, match="adm_1 is required"):
            Prediction(
                id=1,
                model=sample_model_schema,
                disease="A90",
                commit="a" * 40,
                description="Test",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
            )

    def test_validate_adm_level_2_missing_adm_2(self, sample_model_schema):
        with pytest.raises(ValueError, match="adm_2 is required"):
            Prediction(
                id=1,
                model=sample_model_schema,
                disease="A90",
                commit="a" * 40,
                description="Test",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=2,
                adm_0="BRA",
                adm_1=33,
            )

    def test_validate_adm_level_3_missing_adm_3(self, sample_model_schema):
        with pytest.raises(ValueError, match="adm_3 is required"):
            Prediction(
                id=1,
                model=sample_model_schema,
                disease="A90",
                commit="a" * 40,
                description="Test",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=3,
                adm_0="BRA",
                adm_1=33,
                adm_2=3304557,
            )

    def test_validate_adm_hierarchy_valid(
        self, sample_model_schema, sample_prediction_data
    ):
        p = Prediction(
            id=1,
            model=sample_model_schema,
            disease="A90",
            commit="a" * 40,
            description="Test",
            case_definition="probable",
            published=True,
            created_at=date(2023, 1, 1),
            adm_level=1,
            adm_0="BRA",
            adm_1=33,
            data=sample_prediction_data,
        )
        assert p.adm_1 == 33

    def test_validate_dates_duplicate(self, sample_model_schema):
        from mosqlient.registry.schema import PredictionDataRow

        data = [
            PredictionDataRow(
                date=date(2023, 1, 1), lower_90=10.0, pred=50.0, upper_90=90.0
            ),
            PredictionDataRow(
                date=date(2023, 1, 1), lower_90=15.0, pred=55.0, upper_90=95.0
            ),
        ]
        with pytest.raises(ValueError, match="duplicate dates"):
            Prediction(
                id=1,
                model=sample_model_schema,
                disease="A90",
                commit="a" * 40,
                description="Test",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
                adm_1=33,
                data=data,
            )

    def test_validate_dates_week_gap(self, sample_model_schema):
        from mosqlient.registry.schema import PredictionDataRow

        model = sample_model_schema
        model.time_resolution = "week"
        model.imdc_year = None

        data = [
            PredictionDataRow(
                date=date(2023, 1, 1), lower_90=10.0, pred=50.0, upper_90=90.0
            ),
            PredictionDataRow(
                date=date(2023, 1, 15), lower_90=15.0, pred=55.0, upper_90=95.0
            ),
        ]
        with pytest.raises(ValueError, match="gap detected"):
            Prediction(
                id=1,
                model=model,
                disease="A90",
                commit="a" * 40,
                description="Test",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
                adm_1=33,
                data=data,
            )

    def test_validate_dates_day_gap(self, sample_model_schema):
        from mosqlient.registry.schema import PredictionDataRow

        model = sample_model_schema
        model.time_resolution = "day"
        model.imdc_year = None

        data = [
            PredictionDataRow(
                date=date(2023, 1, 1), lower_90=10.0, pred=50.0, upper_90=90.0
            ),
            PredictionDataRow(
                date=date(2023, 1, 5), lower_90=15.0, pred=55.0, upper_90=95.0
            ),
        ]
        with pytest.raises(ValueError, match="gap detected"):
            Prediction(
                id=1,
                model=model,
                disease="A90",
                commit="a" * 40,
                description="Test",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
                adm_1=33,
                data=data,
            )

    def test_validate_week_not_sunday(self, sample_model_schema):
        from mosqlient.registry.schema import PredictionDataRow

        model = sample_model_schema
        model.time_resolution = "week"
        model.imdc_year = None

        data = [
            PredictionDataRow(
                date=date(2023, 1, 2), lower_90=10.0, pred=50.0, upper_90=90.0
            ),
        ]
        with pytest.raises(ValueError, match="not the start of CDC"):
            Prediction(
                id=1,
                model=model,
                disease="A90",
                commit="a" * 40,
                description="Test",
                case_definition="probable",
                published=True,
                created_at=date(2023, 1, 1),
                adm_level=1,
                adm_0="BRA",
                adm_1=33,
                data=data,
            )

    def test_validate_no_data(self, sample_model_schema):
        p = Prediction(
            id=1,
            model=sample_model_schema,
            disease="A90",
            commit="a" * 40,
            description="Test",
            case_definition="probable",
            published=True,
            created_at=date(2023, 1, 1),
            adm_level=1,
            adm_0="BRA",
            adm_1=33,
            data=[],
        )
        assert p.data == []


class TestModelGETParams:
    def test_default_values(self):
        p = ModelGETParams()
        assert p.method == "GET"
        assert p.app == "registry"
        assert p.endpoint == "models"

    def test_params_filters_none(self):
        p = ModelGETParams(id=1, disease="A90")
        params = p.params()
        assert params == {"id": 1, "disease": "A90"}
        assert "page" not in params

    def test_params_with_page(self):
        p = ModelGETParams(id=1, page=1, per_page=10)
        params = p.params()
        assert params == {"id": 1, "page": 1, "per_page": 10}


class TestPredictionGETParams:
    def test_default_values(self):
        p = PredictionGETParams()
        assert p.method == "GET"
        assert p.app == "registry"
        assert p.endpoint == "predictions"

    def test_params_filters_none(self):
        p = PredictionGETParams(id=1, model_id=5)
        params = p.params()
        assert params == {"id": 1, "model_id": 5}

    def test_params_with_dates(self):
        p = PredictionGETParams(start=date(2023, 1, 1), end=date(2023, 12, 31))
        params = p.params()
        assert "start" in params
        assert "end" in params


class TestPredictionPOSTParams:
    def test_params(self):
        p = PredictionPOSTParams(
            repository="owner/repo",
            description="Test",
            disease="A90",
            commit="a" * 40,
            case_definition="probable",
            published=True,
            adm_level=1,
            adm_0="BRA",
            adm_1=33,
            prediction=[],
        )
        params = p.params()
        assert params["repository"] == "owner/repo"
        assert params["disease"] == "A90"
        assert params["published"] is True


class TestPredictionDELETEParams:
    def test_endpoint_replacement(self):
        p = PredictionDELETEParams(id=123)
        assert "123" in p.endpoint
        assert "{predict_id}" not in p.endpoint

    def test_params_returns_none(self):
        p = PredictionDELETEParams(id=123)
        assert p.params() is None


class TestPredictionPublishPATCHParams:
    def test_endpoint_replacement(self):
        p = PredictionPublishPATCHParams(id=456, published=True)
        assert "456" in p.endpoint
        assert "{prediction_id}" not in p.endpoint

    def test_params(self):
        p = PredictionPublishPATCHParams(id=456, published=False)
        params = p.params()
        assert params == {"published": False}


class TestPredictionDataGETParams:
    def test_endpoint_replacement(self):
        p = PredictionDataGETParams(id=789)
        assert "789" in p.endpoint
        assert "{predict_id}" not in p.endpoint

    def test_params(self):
        p = PredictionDataGETParams(id=789)
        params = p.params()
        assert params == {}
