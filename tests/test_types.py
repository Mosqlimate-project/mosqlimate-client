"""Tests for the types module."""

import pytest
from datetime import date
from mosqlient import types
from mosqlient import validations


class TestSchema:
    def test_schema_can_be_instantiated(self):
        schema = types.Schema()
        assert schema is not None


class TestParams:
    def test_params_is_abstract(self):
        with pytest.raises(TypeError):
            types.Params(method="GET", app="registry", endpoint="models")

    def test_params_abstract_method(self):
        class ConcreteParams(types.Params):
            method: str = "GET"  # type: ignore[assignment]
            app: str = "registry"
            endpoint: str = "models"

            def params(self):
                return {}

        p = ConcreteParams()
        assert p.params() == {}


class TestModel:
    def test_model_can_be_instantiated(self):
        from mosqlient.types import Model

        class DummySchema:
            def json(self):
                return "{}"

            def dict(self):
                return {}

        class ConcreteModel(Model):
            _schema: DummySchema  # type: ignore[assignment]

        m = ConcreteModel()
        m._schema = DummySchema()
        assert m is not None

    def test_model_str(self):
        class ConcreteModel(types.Model):
            _schema: object  # type: ignore[assignment]

        class DummySchema:
            def json(self):
                return '{"test": "data"}'

            def dict(self):
                return {"test": "data"}

        m = ConcreteModel()
        m._schema = DummySchema()
        assert m.json() == '{"test": "data"}'
        assert m.dict() == {"test": "data"}

    def test_model_json(self):
        class ConcreteModel(types.Model):
            _schema: object  # type: ignore[assignment]

        class DummySchema:
            def json(self):
                return '{"key": "value"}'

        m = ConcreteModel()
        m._schema = DummySchema()
        assert m.json() == '{"key": "value"}'

    def test_model_dict(self):
        class ConcreteModel(types.Model):
            _schema: object  # type: ignore[assignment]

        class DummySchema:
            def dict(self):
                return {"key": "value"}

        m = ConcreteModel()
        m._schema = DummySchema()
        assert m.dict() == {"key": "value"}


class TestAnnotatedTypes:
    def test_annotated_app_validation(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            app: types.APP

        model = TestModel(app="registry")
        assert model.app == "registry"

        with pytest.raises(Exception):
            TestModel(app="invalid")

    def test_annotated_id_validation(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            id: types.ID

        model = TestModel(id=1)
        assert model.id == 1

        with pytest.raises(Exception):
            TestModel(id=0)

    def test_annotated_disease_validation(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            disease: types.Disease

        model = TestModel(disease="A90")
        assert model.disease == "A90"

        with pytest.raises(Exception):
            TestModel(disease="INVALID")

    def test_annotated_commit_validation(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            commit: types.Commit

        model = TestModel(commit="a" * 40)
        assert model.commit == "a" * 40

        with pytest.raises(Exception):
            TestModel(commit="short")

    def test_annotated_uf_validation(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            uf: types.UF

        model = TestModel(uf="SP")
        assert model.uf == "SP"

        with pytest.raises(Exception):
            TestModel(uf="XX")

    def test_annotated_geocode_validation(self):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            geocode: types.Geocode

        model = TestModel(geocode=3304557)
        assert model.geocode == 3304557

        with pytest.raises(Exception):
            TestModel(geocode=123)
