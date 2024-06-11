import uuid
from typing import Literal, Optional
from typing_extensions import Annotated

from pydantic.functional_validators import AfterValidator

from mosqlient.requests import get


class PlatformClient:
    def __init__(
        self,
        x_uid_key: str,
        env: Optional[Literal["dev", "prod"]] = "prod"
    ):
        self.username, self.uid_key = x_uid_key.split(":")
        self.env = env
        self._check_username()
        self._check_uuid4()

    def __str__(self):
        return self.username

    @property
    def X_UID_KEY(self):
        return f"{self.username}:{self.uid_key}"

    def _check_username(self):
        author = get(
            "registry",
            "authors",
            {"username": self.username},
            pagination=False,
            env=self.env
        )

        if author.status_code != 200:
            raise ValueError(
                f"Could not get user '{self.username}' info. ",
                f"Status code: {author.status_code}"
            )

    def _check_uuid4(self):
        try:
            uuid.UUID(self.uid_key, version=4)
        except ValueError:
            raise ValueError("uid_key is not a valid key")


# Avoiding circular imports
def validate_client(c: PlatformClient) -> PlatformClient:
    return c


Client = Annotated[PlatformClient, AfterValidator(validate_client)]
