"""Tests for the validations module."""

import pytest
import datetime as dt
from pydantic import ValidationError
from mosqlient import validations
from mosqlient._config import (
    DJANGO_APPS,
    DISEASES,
    ADM_LEVELS,
    TIME_RESOLUTIONS,
)


class TestValidateDjangoApp:
    def test_valid_app_registry(self):
        assert validations.validate_django_app("registry") == "registry"

    def test_valid_app_datastore(self):
        assert validations.validate_django_app("datastore") == "datastore"

    def test_valid_app_vis(self):
        assert validations.validate_django_app("vis") == "vis"

    def test_invalid_app(self):
        with pytest.raises(AssertionError, match="Unknown Mosqlimate app"):
            validations.validate_django_app("invalid")


class TestValidateId:
    def test_positive_id(self):
        assert validations.validate_id(1) == 1
        assert validations.validate_id(100) == 100

    def test_zero_id(self):
        with pytest.raises(AssertionError, match="Incorrect ID"):
            validations.validate_id(0)

    def test_negative_id(self):
        with pytest.raises(AssertionError, match="Incorrect ID"):
            validations.validate_id(-1)


class TestValidateName:
    def test_valid_name(self):
        assert validations.validate_name("test") == "test"

    def test_max_length_name(self):
        name = "a" * 100
        assert validations.validate_name(name) == name

    def test_empty_name(self):
        with pytest.raises(AssertionError, match="empty name"):
            validations.validate_name("")

    def test_too_long_name(self):
        with pytest.raises(AssertionError, match="name too long"):
            validations.validate_name("a" * 101)


class TestValidateDescription:
    def test_valid_description(self):
        assert validations.validate_description("test") == "test"

    def test_max_length_description(self):
        desc = "a" * 500
        assert validations.validate_description(desc) == desc

    def test_empty_description(self):
        with pytest.raises(AssertionError, match="empty description"):
            validations.validate_description("")

    def test_too_long_description(self):
        with pytest.raises(AssertionError, match="description too long"):
            validations.validate_description("a" * 501)


class TestValidateAuthorName:
    def test_valid_author_name(self):
        assert validations.validate_author_name("John") == "John"

    def test_max_length_author_name(self):
        name = "a" * 100
        assert validations.validate_author_name(name) == name

    def test_none_author_name(self):
        assert validations.validate_author_name(None) is None

    def test_empty_author_name(self):
        with pytest.raises(AssertionError, match="empty author_name"):
            validations.validate_author_name("")

    def test_too_long_author_name(self):
        with pytest.raises(AssertionError, match="author_name too long"):
            validations.validate_author_name("a" * 101)


class TestValidateAuthorUsername:
    def test_valid_username(self):
        assert validations.validate_author_username("john_doe") == "john_doe"

    def test_max_length_username(self):
        username = "a" * 39
        assert validations.validate_author_username(username) == username

    def test_empty_username(self):
        with pytest.raises(AssertionError, match="empty author_username"):
            validations.validate_author_username("")

    def test_too_long_username(self):
        with pytest.raises(AssertionError, match="author_username too long"):
            validations.validate_author_username("a" * 40)


class TestValidateAuthorInstitution:
    def test_valid_institution(self):
        assert (
            validations.validate_author_institution("University")
            == "University"
        )

    def test_max_length_institution(self):
        inst = "a" * 100
        assert validations.validate_author_institution(inst) == inst

    def test_none_institution(self):
        assert validations.validate_author_institution(None) is None

    def test_empty_string_institution(self):
        assert validations.validate_author_institution("") is None

    def test_whitespace_institution(self):
        result = validations.validate_author_institution(" ")
        assert result == " "

    def test_too_long_institution(self):
        with pytest.raises(
            AssertionError, match="author_institution too long"
        ):
            validations.validate_author_institution("a" * 101)


class TestValidateRepository:
    def test_valid_repository(self):
        assert validations.validate_repository("owner/repo") == "owner/repo"

    def test_invalid_repository_no_slash(self):
        with pytest.raises(AssertionError, match="malformed repository"):
            validations.validate_repository("owner_repo")


class TestValidateImplementationLanguage:
    @pytest.mark.parametrize(
        "lang",
        [
            "python",
            "Python",
            "PYTHON",
            "r",
            "R",
            "c++",
            "c#",
            "java",
            "javascript",
            "go",
            "rust",
            "zig",
            "ruby",
            "lua",
            "kotlin",
            "haskell",
            "erlang",
            ".net",
            "c",
            "coffeescript",
        ],
    )
    def test_valid_languages(self, lang):
        result = validations.validate_implementation_language(lang)
        assert result == lang

    def test_invalid_language(self):
        with pytest.raises(
            AssertionError, match="Unknown implementation_language"
        ):
            validations.validate_implementation_language("cobol")


class TestValidateDisease:
    @pytest.mark.parametrize("disease", DISEASES)
    def test_valid_diseases(self, disease):
        assert validations.validate_disease(disease) == disease

    def test_invalid_disease(self):
        with pytest.raises(AssertionError, match="Unknown disease"):
            validations.validate_disease("A99")


class TestValidateCategory:
    pass


class TestValidateAdmLevel:
    @pytest.mark.parametrize("level", ADM_LEVELS)
    def test_valid_adm_levels(self, level):
        assert validations.validate_adm_level(level) == level

    def test_invalid_adm_level(self):
        with pytest.raises(AssertionError, match="Unknown adm_level"):
            validations.validate_adm_level(5)


class TestValidateTimeResolution:
    @pytest.mark.parametrize("resolution", TIME_RESOLUTIONS)
    def test_valid_time_resolutions(self, resolution):
        assert validations.validate_time_resolution(resolution) == resolution

    def test_invalid_time_resolution(self):
        with pytest.raises(AssertionError, match="Unkown time_resolution"):
            validations.validate_time_resolution("hour")


class TestValidateTemporal:
    def test_temporal_true(self):
        assert validations.validate_temporal(True) is True

    def test_temporal_false(self):
        assert validations.validate_temporal(False) is False


class TestValidateSpatial:
    def test_spatial_true(self):
        assert validations.validate_spatial(True) is True

    def test_spatial_false(self):
        assert validations.validate_spatial(False) is False


class TestValidateCategorical:
    def test_categorical_true(self):
        assert validations.validate_categorical(True) is True

    def test_categorical_false(self):
        assert validations.validate_categorical(False) is False


class TestValidateCommit:
    def test_valid_commit(self):
        commit = "a" * 40
        assert validations.validate_commit(commit) == commit

    def test_valid_commit_mixed(self):
        commit = "abc123def456" * 3 + "abc1"
        assert len(commit) == 40
        assert validations.validate_commit(commit) == commit

    def test_invalid_characters(self):
        commit = "G" + "a" * 39
        assert len(commit) == 40
        with pytest.raises(AssertionError, match="Invalid GitHub commit hash"):
            validations.validate_commit(commit)


class TestValidateDate:
    def test_valid_date_string(self):
        result = validations.validate_date("2023-01-01")
        assert result == "2023-01-01"

    def test_valid_date_object(self):
        d = dt.date(2023, 6, 15)
        result = validations.validate_date(d)
        assert result == "2023-06-15"

    def test_date_too_old(self):
        with pytest.raises(
            (ValidationError, AssertionError), match="date is too old"
        ):
            validations.validate_date("2000-01-01")

    def test_date_in_future(self):
        future = dt.date.today() + dt.timedelta(days=365)
        with pytest.raises(
            (ValidationError, AssertionError), match="date is in the future"
        ):
            validations.validate_date(future.isoformat())

    def test_invalid_format(self):
        with pytest.raises(Exception):
            validations.validate_date("01-01-2023")


class TestValidatePredictionData:
    def test_valid_prediction_data(self):
        data = [
            {
                "date": "2023-01-01",
                "lower_50": 40.0,
                "lower_80": 30.0,
                "lower_90": 20.0,
                "lower_95": 10.0,
                "pred": 50.0,
                "upper_50": 60.0,
                "upper_80": 70.0,
                "upper_90": 80.0,
                "upper_95": 90.0,
            }
        ]
        result = validations.validate_prediction_data(data)
        assert result == data

    def test_invalid_type_not_list(self):
        with pytest.raises(AssertionError, match="invalid `data` type"):
            validations.validate_prediction_data({"key": "value"})

    def test_invalid_item_not_dict(self):
        with pytest.raises(AssertionError, match="invalid `data` type"):
            validations.validate_prediction_data([["list", "item"]])

    def test_empty_list(self):
        with pytest.raises(ValueError, match="empty prediction"):
            validations.validate_prediction_data([])

    def test_missing_columns(self):
        data = [{"date": "2023-01-01", "pred": 50.0}]
        with pytest.raises(AssertionError, match="expected fields"):
            validations.validate_prediction_data(data)


class TestValidateTags:
    def test_returns_tags(self):
        tags = [1, 2, 3]
        assert validations.validate_tags(tags) == tags


class TestValidateUf:
    @pytest.mark.parametrize(
        "uf",
        [
            "AC",
            "AL",
            "AP",
            "AM",
            "BA",
            "CE",
            "ES",
            "GO",
            "MA",
            "MT",
            "MS",
            "MG",
            "PA",
            "PB",
            "PR",
            "PE",
            "PI",
            "RJ",
            "RN",
            "RS",
            "RO",
            "RR",
            "SC",
            "SP",
            "SE",
            "TO",
            "DF",
        ],
    )
    def test_valid_ufs(self, uf):
        result = validations.validate_uf(uf)
        assert result == uf

    def test_lowercase_uf(self):
        result = validations.validate_uf("sp")
        assert result == "SP"

    def test_invalid_uf(self):
        with pytest.raises(AssertionError, match="Unknown UF"):
            validations.validate_uf("XX")


class TestValidateGeocode:
    def test_valid_geocode(self):
        assert validations.validate_geocode(3304557) == 3304557

    def test_invalid_length(self):
        with pytest.raises(
            AssertionError, match="Invalid municipality geocode"
        ):
            validations.validate_geocode(123456)

    def test_invalid_state_code(self):
        with pytest.raises(
            AssertionError, match="Invalid municipality geocode"
        ):
            validations.validate_geocode(9900001)


class TestValidateMacroHealthGeocode:
    def test_valid_geocode(self):
        assert validations.validate_macro_health_geocode(1101) == 1101

    def test_invalid_length(self):
        with pytest.raises(
            AssertionError, match="Invalid macro health geocode"
        ):
            validations.validate_macro_health_geocode(12345)
