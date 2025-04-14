from typing import Any, Iterable, Union
from tabulate import tabulate


class ValidationErrorBase(Exception):
    """Base exception class for validation errors"""

    def __init__(self, message=None):
        super().__init__(message or "A validation error occurred")


class FieldTypeError(ValidationErrorBase):
    """Exception raised for errors in a specific field."""

    def __init__(self, field: str, _type: Union[Iterable, Any]):
        self.field = field
        super().__init__(
            f"Incorrect type for field: '{field}'. Expected type(s): '{_type}'"
        )

    def __str__(self):
        return f"{self.field}: {self.args[0]}"


class ParameterError(ValidationErrorBase):
    def __init__(self, message: str = "", options: list[str] = []):
        table = tabulate([[option] for option in options], headers=["OPTIONS"])
        super().__init__(f"{message}\n\n{table}")


class ClientError(Exception):
    def __init__(self, message=None):
        super().__init__(message or "Client error")


class ModelPostError(Exception):
    def __init__(self, message=None):
        super().__init__(message or "registry.Model POST request error")


class PredictionPostError(Exception):
    def __init__(self, message=None):
        super().__init__(message or "registry.Prediction POST request error")
