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
    "get_models_by_tag_ids"
]

from typing import Optional, List, Union

from .models import Model


def get_all_models() -> list[dict]:
    return Model.get()


def get_models(
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
) -> Union[dict, list[dict]]:
    params = {
        "id": id,
        "name": name,
        "author_name": author_name,
        "author_username": author_username,
        "author_institution": author_institution,
        "repository": repository,
        "implementation_language": implementation_language,
        "disease": disease,
        "ADM_level": ADM_level,
        "temporal": temporal,
        "spatial": spatial,
        "categorical": categorical,
        "time_resolution": time_resolution,
        "tags": tags,
    }
    res = Model.get(**params)
    return res[0] if len(res) == 1 else res


def get_model_by_id(id: int) -> Union[dict, list]:
    res = Model.get(id=id)
    return res[0] if len(res) == 1 else res


def get_models_by_author_name(author_name: str) -> Union[dict, list[dict]]:
    res = Model.get(author_name=author_name)
    return res[0] if len(res) == 1 else res


def get_models_by_author_username(
    author_username: str
) -> Union[dict, list[dict]]:
    res = Model.get(author_username=author_username)
    return res[0] if len(res) == 1 else res


def get_models_by_author_institution(
    author_institution: str
) -> Union[dict, list[dict]]:
    res = Model.get(author_institution=author_institution)
    return res[0] if len(res) == 1 else res


def get_models_by_repository(repository: str) -> Union[dict, list[dict]]:
    res = Model.get(repository=repository)
    return res[0] if len(res) == 1 else res


def get_models_by_implementation_language(
    implementation_language: str
) -> Union[dict, list[dict]]:
    res = Model.get(implementation_language=implementation_language)
    return res[0] if len(res) == 1 else res


def get_models_by_disease(disease: str) -> Union[dict, list[dict]]:
    res = Model.get(disease=disease)
    return res[0] if len(res) == 1 else res


def get_models_by_adm_level(adm_level: int) -> Union[dict, list[dict]]:
    res = Model.get(adm_level=adm_level)
    return res[0] if len(res) == 1 else res


def get_models_by_time_resolution(
    time_resolution: int
) -> Union[dict, list[dict]]:
    res = Model.get(time_resolution=time_resolution)
    return res[0] if len(res) == 1 else res


def get_models_by_tag_ids(tags_ids: list[int]) -> Union[dict, list[dict]]:
    res = Model.get(tags=tags_ids)
    return res[0] if len(res) == 1 else res
