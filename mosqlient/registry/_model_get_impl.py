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
    "get_models_by_tag_ids",
]

from typing import Optional, List

from .models import Model


def get_all_models(api_key: str) -> list[dict]:
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
    tags: Optional[List[int]] = None,
    sprint: Optional[bool] = None,
) -> List[Model]:
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
        tags=tags,
        sprint=sprint,
    )


def get_model_by_id(api_key: str, id: int) -> Model | None:
    res = Model.get(api_key=api_key, id=id)
    return res[0] if len(res) == 1 else None


def get_models_by_author_name(api_key: str, author_name: str) -> List[Model]:
    return Model.get(api_key=api_key, author_name=author_name)


def get_models_by_author_username(
    api_key: str,
    author_username: str,
) -> List[Model]:
    return Model.get(api_key=api_key, author_username=author_username)


def get_models_by_author_institution(
    api_key: str,
    author_institution: str,
) -> List[Model]:
    return Model.get(api_key=api_key, author_institution=author_institution)


def get_models_by_repository(api_key: str, repository: str) -> List[Model]:
    return Model.get(api_key=api_key, repository=repository)


def get_models_by_implementation_language(
    api_key: str,
    implementation_language: str,
) -> List[Model]:
    return Model.get(
        api_key=api_key, implementation_language=implementation_language
    )


def get_models_by_disease(api_key: str, disease: str) -> List[Model]:
    return Model.get(api_key=api_key, disease=disease)


def get_models_by_adm_level(api_key: str, adm_level: int) -> List[Model]:
    return Model.get(api_key=api_key, adm_level=adm_level)


def get_models_by_time_resolution(
    api_key: str,
    time_resolution: int,
) -> List[Model]:
    return Model.get(api_key=api_key, time_resolution=time_resolution)


def get_models_by_tag_ids(api_key: str, tags_ids: list[int]) -> List[Model]:
    return Model.get(api_key=api_key, tags=tags_ids)
