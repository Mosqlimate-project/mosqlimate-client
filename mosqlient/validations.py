from mosqlient.client import PlatformClient

from mosqlient.config import DJANGO_APPS


def validate_django_app(app: str) -> str:
    assert app in DJANGO_APPS, (
        f"Unknown Mosqlimate app '{app}'. Options: {DJANGO_APPS}"
    )
    return app


def validate_client(c: PlatformClient) -> PlatformClient:
    return c


def validate_id(ID: int) -> int:
    assert ID > 0, f"Incorrect ID {ID}"
    return ID


def validate_name(name: str) -> str:
    assert len(name) <= 100, "name too long"
    assert len(name) > 0, "empty name"
    return name


def validate_description(description: str) -> str:
    assert len(description) <= 500, "description too long"
    assert len(description) > 0, "empty description"
    return description


def validate_author_name(author_name: str) -> str:
    assert len(author_name) <= 100, "author_name too long"
    assert len(author_name) > 0, "empty author_name"
    return author_name


def validate_author_username(author_username: str) -> str:
    assert len(author_username) < 40, "author_username too long"
    assert len(author_username) > 0, "empty author_username"
    return author_username


def validate_author_institution(author_institution: str) -> str:
    assert len(author_institution) <= 100, "author_institution too long"
    assert len(author_institution) > 0, "empty author_institution"
    return author_institution


def validate_repository(repository: str) -> str:
    assert len(repository) <= 100, "repository too long"
    assert len(repository) > 0, "empty repository"
    return repository


def validate_implementation_language(implementation_language: str) -> str:
    languages = [
        "Zig",
        "Rust",
        "Ruby",
        "R",
        "Lua",
        "Kotlin",
        "Java",
        "JavaScript",
        "Haskell",
        "Go",
        "Erlang",
        ".NET",
        "C",
        "C#",
        "CoffeeScript",
        "C++",
        "Python",
    ]
    assert implementation_language in languages, (
        f"Unknown implementation_language {implementation_language}"
    )
    return implementation_language
