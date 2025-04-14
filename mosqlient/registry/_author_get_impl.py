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
    """
    Function that returns the list of all authors registered in the plaform

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.

    Returns
    -------
    List of Authors
    """
    return Author.get(api_key=api_key)


def get_authors(
    api_key: str,
    name: Optional[str] = None,
    institution: Optional[str] = None,
    username: Optional[str] = None,
) -> list[Author]:
    """
    Function that returns the list of authors registered in the plaform filtered
    by one of the parameters in the function

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        name : str, optional
            Author name.
        institution : str, optional
            Author institution.
        username : str, optional
            Author username.

    Returns
    -------
    List of Authors that match the parameters provided
    """

    return Author.get(
        api_key=api_key, name=name, institution=institution, username=username
    )


def get_author_by_username(api_key: str, username: str) -> Author | None:
    author = Author.get(api_key=api_key, username=username)
    """
    Function that return the author based on the username

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        username : str
            Author username.

    Returns
    -------
    Author class
    """
    return author[0] if len(author) == 1 else None


def get_authors_by_name(api_key: str, name: str) -> List[Author]:
    """
    Function that return the authors based on the name

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        name : str
            Author name.

    Returns
    -------
    List of authors
    """
    return Author.get(api_key=api_key, name=name)


def get_authors_by_institution(api_key: str, institution: str) -> List[Author]:
    """
    Function that return the authors based on the institution

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        institution : str
            Author institution.

    Returns
    -------
    List of authors
    """
    return Author.get(api_key=api_key, institution=institution)
