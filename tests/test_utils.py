"""Tests for the _utils module."""

import pytest
from datetime import date
from mosqlient._utils import parse_params


class TestParseParams:
    def test_string_param(self):
        result = parse_params(name="test")
        assert result == {"name": "test"}

    def test_int_param(self):
        result = parse_params(count=42)
        assert result == {"count": "42"}

    def test_float_param(self):
        result = parse_params(value=3.14)
        assert result == {"value": "3.14"}

    def test_bool_param(self):
        result = parse_params(active=True)
        assert result == {"active": "True"}

        result = parse_params(active=False)
        assert result == {"active": "False"}

    def test_date_param(self):
        d = date(2023, 1, 15)
        result = parse_params(start_date=d)
        assert result == {"start_date": "2023-01-15"}

    def test_none_param_skipped(self):
        result = parse_params(name="test", empty=None)
        assert result == {"name": "test"}
        assert "empty" not in result

    def test_multiple_params(self):
        result = parse_params(name="test", count=10, active=True)
        assert result == {"name": "test", "count": "10", "active": "True"}

    def test_model_param(self):
        from mosqlient.types import Model

        class DummySchema:
            def json(self):
                return '{"test": "model"}'

            def dict(self):
                return {"test": "model"}

        class DummyModel(Model):
            _schema: DummySchema  # type: ignore[assignment]

        model = DummyModel()
        model._schema = DummySchema()
        result = parse_params(model=model)
        assert "model" in result
        assert isinstance(result["model"], str)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Unknown type"):
            parse_params(data=[1, 2, 3])

    def test_unknown_type_dict_raises(self):
        with pytest.raises(TypeError, match="Unknown type"):
            parse_params(data={"key": "value"})

    def test_empty_kwargs(self):
        result = parse_params()
        assert result == {}
