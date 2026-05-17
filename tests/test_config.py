"""Tests for the _config module."""

from mosqlient._config import globals


class TestGlobals:
    def test_django_apps(self):
        assert globals.DJANGO_APPS == ["registry", "datastore", "vis"]

    def test_diseases(self):
        assert globals.DISEASES == ["A90", "A92.0", "A92.5"]

    def test_adm_levels(self):
        assert globals.ADM_LEVELS == [0, 1, 2, 3]

    def test_time_resolutions(self):
        assert globals.TIME_RESOLUTIONS == ["day", "week", "month", "year"]

    def test_prediction_data_columns(self):
        expected = [
            "date",
            "lower_50",
            "lower_80",
            "lower_90",
            "lower_95",
            "pred",
            "upper_50",
            "upper_80",
            "upper_90",
            "upper_95",
        ]
        assert globals.PREDICTION_DATA_COLUMNS == expected
