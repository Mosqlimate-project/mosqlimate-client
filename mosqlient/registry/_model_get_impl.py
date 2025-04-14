__all__ = [
    "get_all_models",
    "get_models",
    "get_model_by_id",
    "get_models_by_author_name",
    "get_models_by_author_username",
    "get_models_by_author_institution",
    "get_models_by_repository",
    "get_models_by_implementation_language",
    "get_models_by_disease",
    "get_models_by_adm_level",
    "get_models_by_time_resolution",
]

from typing import Optional, List

from .models import Model


def get_all_models(api_key: str) -> list[dict]:
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
    name: Optional[str] = None,
    author_name: Optional[str] = None,
    author_username: Optional[str] = None,
    author_institution: Optional[str] = None,
    repository: Optional[str] = None,
    implementation_language: Optional[str] = None,
    disease: Optional[str] = None,
    ADM_level: Optional[int] = None,
    temporal: Optional[bool] = None,
    spatial: Optional[bool] = None,
    categorical: Optional[bool] = None,
    time_resolution: Optional[str] = None,
    sprint: Optional[bool] = None,
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
        name: str, optional
            Model name.
        author_name: str, optional
            Author name.
        author_username: str, optional
            Author username.
        author_institution:str, optional
            Author institution
        repository: str, optional
            Name of the Github repository where the model's source code is stored.
        implementation_language: str, optional
            Name of the implementation language of the model.
        disease: str, optional
            Disease name. Options are: 'dengue', 'chikungunya' and 'zika'
        ADM_level: int, optional
            ADM level of the model. Options:
            0, 1, 2, 3 (National, State, Municipality, Sub Municipality)
        temporal: bool, optional
            Indicates whether the model is temporal.
        spatial: bool, optional
            Indicates whether the model is spatial.
        categorical: bool, optional
            Indicates whether the model is categorical.
        time_resolution: str, optional
            Time resolution of the model. Options are: 'day', 'week', 'month' or 'year'
        sprint: bool, optional
            Indicates whether the model belong to the sprint.

    Returns
    -------
    List of Models
    """

    return Model.get(
        api_key=api_key,
        id=id,
        name=name,
        author_name=author_name,
        author_username=author_username,
        author_institution=author_institution,
        repository=repository,
        implementation_language=implementation_language,
        disease=disease,
        ADM_level=ADM_level,
        temporal=temporal,
        spatial=spatial,
        categorical=categorical,
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


def get_models_by_author_name(api_key: str, author_name: str) -> List[Model]:
    """
    Returns a list of models based on the author name.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        author_name: str, optional
            Author name.
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, author_name=author_name)


def get_models_by_author_username(
    api_key: str,
    author_username: str,
) -> List[Model]:
    """
    Returns a list of models based on the author username.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        author_username: str, optional
            Author username.
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, author_username=author_username)


def get_models_by_author_institution(
    api_key: str,
    author_institution: str,
) -> List[Model]:
    """
    Returns a list of models based on the author institution.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        author_institution: str, optional
            Author institution.
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, author_institution=author_institution)


def get_models_by_repository(api_key: str, repository: str) -> List[Model]:
    """
    Returns a list of models based on the author institution.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        author_institution: str, optional
            Author institution.
    Returns
    -------
    List of Models
    """

    return Model.get(api_key=api_key, repository=repository)


def get_models_by_implementation_language(
    api_key: str,
    implementation_language: str,
) -> List[Model]:
    """
    Returns a list of models based on the author institution.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        implementation_language: str, optional
            Name of the implementation language of the model.
    Returns
    -------
    List of Models
    """
    return Model.get(
        api_key=api_key, implementation_language=implementation_language
    )


def get_models_by_disease(api_key: str, disease: str) -> List[Model]:
    """
    Returns a list of models based on the disease.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        disease: str, optional
            Disease. Options are: 'dengue', 'chikungunya' and 'zika'.
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, disease=disease)


def get_models_by_adm_level(api_key: str, adm_level: int) -> List[Model]:
    """
    Returns a list of models based on the ADM level.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        ADM_level: int, optional
            ADM level of the model. Options:
            0, 1, 2, 3 (National, State, Municipality, Sub Municipality).
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, adm_level=adm_level)


def get_models_by_time_resolution(
    api_key: str,
    time_resolution: int,
) -> List[Model]:
    """
    Returns a list of models based on the time resolution.

    Parameters
    ----------
        api_key : str
            API key used to authenticate with the Mosqlimate service.
        time_resolution: str, optional
            Time resolution of the model. Options are: 'day', 'week', 'month' or 'year'
    Returns
    -------
    List of Models
    """
    return Model.get(api_key=api_key, time_resolution=time_resolution)
