__all__ = [
    "get_all_authors",
    "get_authors",
    "get_author_by_username",
    "get_authors_by_name",
    "get_authors_by_institution",
]

from typing import Optional, List

from .models import Author


def get_all_authors(api_key: str) -> List[Author]:
    return Author.get(api_key=api_key)


def get_authors(
    api_key: str,
    name: Optional[str] = None,
    institution: Optional[str] = None,
    username: Optional[str] = None,
) -> list[Author]:
    return Author.get(
        api_key=api_key, name=name, institution=institution, username=username
    )


def get_author_by_username(api_key: str, username: str) -> Author | None:
    author = Author.get(api_key=api_key, username=username)
    return author[0] if len(author) == 1 else None


def get_authors_by_name(api_key: str, name: str) -> List[Author]:
    return Author.get(api_key=api_key, name=name)


def get_authors_by_institution(api_key: str, institution: str) -> List[Author]:
    return Author.get(api_key=api_key, institution=institution)
