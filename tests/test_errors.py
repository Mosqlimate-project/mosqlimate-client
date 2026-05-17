"""Tests for the errors module."""

import pytest
from mosqlient import errors


class TestValidationErrorBase:
    def test_default_message(self):
        err = errors.ValidationErrorBase()
        assert str(err) == "A validation error occurred"

    def test_custom_message(self):
        err = errors.ValidationErrorBase("custom error")
        assert str(err) == "custom error"


class TestFieldTypeError:
    def test_str_representation(self):
        err = errors.FieldTypeError("field_name", "str")
        assert "field_name" in str(err)
        assert "Incorrect type" in str(err)

    def test_multiple_types(self):
        err = errors.FieldTypeError("age", ["int", "float"])
        assert "age" in str(err)
        assert "int" in str(err)
        assert "float" in str(err)

    def test_field_attribute(self):
        err = errors.FieldTypeError("test_field", "str")
        assert err.field == "test_field"


class TestParameterError:
    def test_with_options(self):
        err = errors.ParameterError(
            "Unknown app", options=["registry", "datastore"]
        )
        msg = str(err)
        assert "Unknown app" in msg
        assert "OPTIONS" in msg
        assert "registry" in msg
        assert "datastore" in msg

    def test_without_options(self):
        err = errors.ParameterError("Error")
        msg = str(err)
        assert "Error" in msg


class TestClientError:
    def test_default_message(self):
        err = errors.ClientError()
        assert str(err) == "Client error"

    def test_custom_message(self):
        err = errors.ClientError("custom client error")
        assert str(err) == "custom client error"


class TestModelPostError:
    def test_default_message(self):
        err = errors.ModelPostError()
        assert "registry.Model POST request error" in str(err)

    def test_custom_message(self):
        err = errors.ModelPostError("custom error")
        assert str(err) == "custom error"


class TestPredictionPostError:
    def test_default_message(self):
        err = errors.PredictionPostError()
        assert "registry.Prediction POST request error" in str(err)

    def test_custom_message(self):
        err = errors.PredictionPostError("custom error")
        assert str(err) == "custom error"
