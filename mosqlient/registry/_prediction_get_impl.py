__all__ = [
    "get_predictions",
    "get_prediction_by_id",
    "get_predictions_by_model_id",
    "get_predictions_by_model_name",
    "get_predictions_by_adm_level",
    "get_predictions_by_time_resolution",
    "get_predictions_by_disease",
    "get_predictions_by_author_name",
    "get_predictions_by_author_username",
    "get_predictions_by_author_institution",
    "get_predictions_by_repository",
    "get_predictions_by_implementation_language",
    "get_predictions_by_predict_date",
    "get_predictions_between",
]

from datetime import date

from typing import Optional, List

from .models import Prediction


def get_predictions(
    api_key: str,
    id: Optional[int] = None,
    model_id: Optional[int] = None,
    model_name: Optional[str] = None,
    model_ADM_level: Optional[int] = None,
    model_time_resolution: Optional[str] = None,
    model_disease: Optional[str] = None,
    author_name: Optional[str] = None,
    author_username: Optional[str] = None,
    author_institution: Optional[str] = None,
    repository: Optional[str] = None,
    implementation_language: Optional[str] = None,
    temporal: Optional[bool] = None,
    spatial: Optional[bool] = None,
    categorical: Optional[bool] = None,
    commit: Optional[str] = None,
    predict_date: Optional[date] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    adm_1_geocode: Optional[int] = None,
    adm_2_geocode: Optional[int] = None,
    sprint: Optional[bool] = None,
) -> list[dict] | dict:
    params = {
        "id": id,
        "model_id": model_id,
        "model_name": model_name,
        "model_ADM_level": model_ADM_level,
        "model_time_resolution": model_time_resolution,
        "model_disease": model_disease,
        "author_name": author_name,
        "author_username": author_username,
        "author_institution": author_institution,
        "repository": repository,
        "implementation_language": implementation_language,
        "temporal": temporal,
        "spatial": spatial,
        "categorical": categorical,
        "commit": commit,
        "predict_date": (
            str(predict_date) if predict_date is not None else None
        ),
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "adm_1_geocode": adm_1_geocode,
        "adm_2_geocode": adm_2_geocode,
        "sprint": sprint,
    }
    return Prediction.get(api_key=api_key, **params)


def get_prediction_by_id(api_key: str, id: int) -> Prediction | None:
    res = Prediction.get(api_key=api_key, id=id)
    return res[0] if len(res) == 1 else None


def get_predictions_by_model_id(
    api_key: str, model_id: int
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, model_id=model_id)


def get_predictions_by_model_name(
    api_key: str, model_name: str
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, model_name=model_name)


def get_predictions_by_adm_level(
    api_key: str, adm_level: int
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, adm_level=adm_level)


def get_predictions_by_time_resolution(
    api_key: str,
    time_resolution: str,
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, time_resolution=time_resolution)


def get_predictions_by_disease(api_key: str, disease: str) -> List[Prediction]:
    return Prediction.get(api_key=api_key, model_disease=disease)


def get_predictions_by_author_name(
    api_key: str, author_name: str
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, author_name=author_name)


def get_predictions_by_author_username(
    api_key: str,
    author_username: str,
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, author_username=author_username)


def get_predictions_by_author_institution(
    api_key: str,
    author_institution: str,
) -> List[Prediction]:
    return Prediction.get(
        api_key=api_key, author_institution=author_institution
    )


def get_predictions_by_repository(
    api_key: str, repository: str
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, repository=repository)


def get_predictions_by_implementation_language(
    api_key: str,
    implementation_language: str,
) -> List[Prediction]:
    return Prediction.get(
        api_key=api_key, implementation_language=implementation_language
    )


def get_predictions_by_predict_date(
    api_key: str, predict_date: date
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, predict_date=str(predict_date))


def get_predictions_between(
    api_key: str, start: date, end: date
) -> List[Prediction]:
    return Prediction.get(api_key=api_key, start=str(start), end=str(end))
