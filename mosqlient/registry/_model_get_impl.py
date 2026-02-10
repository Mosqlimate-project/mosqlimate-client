__all__ = [
    "get_all_models",
    "get_models",
    "get_model_by_id",
    "get_models_by_repository_owner",
    "get_models_by_repository_organization",
    "get_models_by_repository_name",
    "get_models_by_disease",
    "get_models_by_category",
    "get_models_by_adm_level",
    "get_models_by_time_resolution",
    "get_models_by_sprint",
]

from typing import Optional, List, Literal

from .models import Model


def get_all_models(api_key: str) -> List[Model]:
    """
    Returns a list of all models that are registered and available on the platform.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.

    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key)


def get_models(
    api_key: str,
    id: Optional[int] = None,
    repository_owner: Optional[str] = None,
    repository_organization: Optional[str] = None,
    repository_name: Optional[str] = None,
    disease: Optional[Literal["A90", "A92.0", "A92.5"]] = None,
    category: Optional[
        Literal[
            "quantitative",
            "categorical",
            "spatial_quantitative",
            "spatial_categorical",
            "spatio_temporal_quantitative",
            "spatio_temporal_categorical",
        ]
    ] = None,
    adm_level: Optional[Literal[0, 1, 2, 3]] = None,
    time_resolution: Optional[Literal["day", "week", "month", "year"]] = None,
    sprint: Optional[int] = None,
) -> List[Model]:
    """
    Returns a list of all models registered on the platform that match the
    specified filter parameters.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        id: int, optional
            Model id.
        repository_owner: str, optional
            Username of the repository owner.
        repository_organization: str, optional
            Name of the repository organization.
        repository_name: str, optional
            Name of the repository.
        disease: str, optional
            Disease code. Options: 'A90' (Dengue), 'A92.0' (Chikungunya), 'A92.5' (Zika).
        category: str, optional
            Model category. Options: 'quantitative', 'categorical', 'spatial_quantitative', etc.
        adm_level: int, optional
            ADM level. Options: 0 (National), 1 (State), 2 (Municipality), 3 (Sub-Municipality).
        time_resolution: str, optional
            Time resolution. Options: 'day', 'week', 'month', 'year'.
        sprint: int, optional
            The year of the sprint the model belongs to (e.g., 2024).

    Returns
    -------
    List of Models
    """

    return Model.get(
        api_key=api_key,
        id=id,
        repository_owner=repository_owner,
        repository_organization=repository_organization,
        repository_name=repository_name,
        disease=disease,
        category=category,
        adm_level=adm_level,
        time_resolution=time_resolution,
        sprint=sprint,
    )


def get_model_by_id(api_key: str, id: int) -> Model | None:
    """
    Returns a model based on the id.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        id: int
            Model id.
    Returns
    -------
    Model
    """
    res = Model.get(api_key=api_key, id=id)
    return res[0] if len(res) == 1 else None


def get_models_by_repository_owner(
    api_key: str,
    repository_owner: str,
) -> List[Model]:
    """
    Returns a list of models based on the repository owner (username).

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        repository_owner: str
            Username of the owner.
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, repository_owner=repository_owner)


def get_models_by_repository_organization(
    api_key: str,
    repository_organization: str,
) -> List[Model]:
    """
    Returns a list of models based on the repository organization.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        repository_organization: str
            Name of the organization.
    Returns
    -------
    List of Models
    """
    return Model.get(
        api_key=api_key, repository_organization=repository_organization
    )


def get_models_by_repository_name(
    api_key: str, repository_name: str
) -> List[Model]:
    """
    Returns a list of models based on the repository name.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        repository_name: str
            Name of the repository.
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, repository_name=repository_name)


def get_models_by_disease(api_key: str, disease: str) -> List[Model]:
    """
    Returns a list of models based on the disease code.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        disease: str
            Disease code (e.g., 'A90', 'A92.0').
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, disease=disease)


def get_models_by_category(api_key: str, category: str) -> List[Model]:
    """
    Returns a list of models based on the category.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        category: str
            Model category (e.g., 'spatial_quantitative').
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, category=category)


def get_models_by_adm_level(api_key: str, adm_level: int) -> List[Model]:
    """
    Returns a list of models based on the ADM level.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        adm_level: int
            ADM level (0, 1, 2, or 3).
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, adm_level=adm_level)


def get_models_by_time_resolution(
    api_key: str,
    time_resolution: str,
) -> List[Model]:
    """
    Returns a list of models based on the time resolution.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        time_resolution: str
            Time resolution ('day', 'week', 'month', 'year').
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, time_resolution=time_resolution)


def get_models_by_sprint(api_key: str, sprint: int) -> List[Model]:
    """
    Returns a list of models based on the sprint year.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        sprint: int
            Year of the sprint.
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, sprint=sprint)
