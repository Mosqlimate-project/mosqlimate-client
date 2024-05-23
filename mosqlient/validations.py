from typing import Union

from mosqlient.errors import FieldTypeError


class FieldValidator():
    def __init__(self, **kwargs):
        if "id" in kwargs:
            self.validate_id(kwargs["id"])

    def validate_id(self, ID: Union[str, int]) -> None:
        if not isinstance(ID, (int, str)):
            raise FieldTypeError("id", (int, str))

        if ID <= 0:
            raise ValueError("Incorrect value for field 'id'")
