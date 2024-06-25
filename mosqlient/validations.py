import json
from typing import Literal
import datetime as dt
from string import ascii_lowercase, digits

import pandas as pd

from pydantic import ValidationError

from mosqlient._config import *  # noqa
from mosqlient._utils.brasil import UF_CODES


def validate_django_app(app: str) -> str:
    assert app in DJANGO_APPS, (
        f"Unknown Mosqlimate app '{app}'. Options: {DJANGO_APPS}"
    )
    return app


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
        "zig",
        "rust",
        "ruby",
        "r",
        "lua",
        "kotlin",
        "java",
        "javascript",
        "haskell",
        "go",
        "erlang",
        ".net",
        "c",
        "c#",
        "coffeescript",
        "c++",
        "python",
    ]
    assert implementation_language.lower() in languages, (
        f"Unknown implementation_language {implementation_language}"
    )
    return implementation_language


def validate_disease(disease: Literal["dengue", "zika", "chikungunya"]) -> str:
    assert disease.lower() in DISEASES, (
        f"Unknown disease '{disease}'. Options: {DISEASES}"
    )
    return disease


def validate_adm_level(adm_level: int) -> int:
    assert adm_level in ADM_LEVELS, (
        f"Unknown adm_level {adm_level}. Options {ADM_LEVELS}"
    )
    return adm_level


def validate_time_resolution(time_resolution: str) -> str:
    assert time_resolution in TIME_RESOLUTIONS, (
        f"Unkown time_resolution {time_resolution}. ",
        f"Options: {TIME_RESOLUTIONS}"
    )
    return time_resolution


def validate_temporal(temporal: bool) -> bool:
    return temporal


def validate_spatial(spatial: bool) -> bool:
    return spatial


def validate_categorical(categorical: bool) -> bool:
    return categorical


def validate_commit(commit: str) -> str:
    err = "Invalid GitHub commit hash"
    assert len(commit) == 40, err
    assert all(c in ascii_lowercase + digits for c in commit), err
    return commit


def validate_date(date: str) -> str:
    error = "Incorrect date format. Format: YYYY-MM-DD \n%s"
    try:
        dt_date = dt.date.fromisoformat(date)
    except Exception as err:
        raise ValidationError(error % err)

    assert dt_date >= dt.date(2010, 1, 1), "date is too old"
    assert dt_date <= dt.datetime.now().date(), "date is in the future"

    return str(date)


def validate_prediction_data(data: str) -> str:
    if not isinstance(data, (str)):
        raise TypeError("`data` must be a str or a list of dicts")

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.decoder.JSONDecodeError:
            raise ValueError("`data` object must be JSON serializable")

    data_df = pd.DataFrame(data)

    assert set(data_df.columns) == set(PREDICTION_DATA_COLUMNS), (
        f"Incorrect data columns. Expecting: {PREDICTION_DATA_COLUMNS}"
    )
    # TODO: Include more checks
    return json.dumps(data)


def validate_tags(tags: list[int]) -> list[int]:
    # TODO:
    return tags

# fmt: off
def validate_uf(
    uf: Literal[
        "AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG",
        "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP",
        "SE", "TO", "DF"
]) -> str:
    # fmt: on
    assert uf.upper() in UF_CODES, f"Unknown UF {uf}. Example: 'SP'"
    return uf.upper()


def validate_geocode(geocode: int) -> int:
    err = "Invalid municipality geocode %s. Example: 3304557"
    assert len(str(geocode)) == 7, err % geocode
    assert int(str(geocode)[:2]) in UF_CODES.values(), err % geocode
    return geocode
