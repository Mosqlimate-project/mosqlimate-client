__all__ = [
    "get_all_predictions",
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
    "get_predictions_between"
]

from datetime import date

from typing import Optional, Union, List

from .models import Prediction


def get_all_predictions() -> list[dict]:
    return Prediction.get()


def get_predictions(
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
        "predict_date": str(predict_date) if predict_date is not None else None,
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
    }

    res = Prediction.get(**params)
    return res[0] if len(res) == 1 else res


def get_prediction_by_id(id: int) -> Prediction:
    res = Prediction.get(id=id)
    return res[0] if len(res) == 1 else res


def get_predictions_by_model_id(model_id: int) -> List[Prediction]:
    return Prediction.get(model_id=model_id)


def get_predictions_by_model_name(model_name: str) -> List[Prediction]:
    return Prediction.get(model_name=model_name)


def get_predictions_by_adm_level(adm_level: int) -> List[Prediction]:
    return Prediction.get(adm_level=adm_level)


def get_predictions_by_time_resolution(
    time_resolution: str
) -> List[Prediction]:
    return Prediction.get(time_resolution=time_resolution)


def get_predictions_by_disease(disease: str) -> List[Prediction]:
    return Prediction.get(disease=disease)


def get_predictions_by_author_name(author_name: str) -> List[Prediction]:
    return Prediction.get(author_name=author_name)


def get_predictions_by_author_username(
    author_username: str
) -> List[Prediction]:
    return Prediction.get(author_username=author_username)


def get_predictions_by_author_institution(
    author_institution: str
) -> List[Prediction]:
    return Prediction.get(author_institution=author_institution)


def get_predictions_by_repository(
    repository: str
) -> List[Prediction]:
    return Prediction.get(repository=repository)


def get_predictions_by_implementation_language(
    implementation_language: str
) -> List[Prediction]:
    return Prediction.get(implementation_language=implementation_language)


def get_predictions_by_predict_date(predict_date: date) -> List[Prediction]:
    return Prediction.get(predict_date=str(predict_date))


def get_predictions_between(start: date, end: date) -> List[Prediction]:
    return Prediction.get(start=str(start), end=str(end))
