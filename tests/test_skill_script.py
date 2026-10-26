"""Tests for the ClawHub mosqlimate helper script."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "mosqlimate"
    / "scripts"
    / "mosqlimate.py"
)


def _load_skill_module():
    spec = importlib.util.spec_from_file_location(
        "mosqlimate_skill", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def skill():
    return _load_skill_module()


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient="records"):
        assert orient == "records"
        return self.rows


class FakeMosqlient:
    def __init__(self):
        self.calls = []

    def get_infodengue(self, **kwargs):
        self.calls.append(("infodengue", kwargs))
        return FakeFrame([{"casos": 1}])

    def get_predictions(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(("predictions", kwargs))
        return [
            SimpleNamespace(id=1, adm_1=33, adm_2=None),
            SimpleNamespace(id=2, adm_1=35, adm_2=None),
            SimpleNamespace(id=3, adm_1=33, adm_2=3304557),
        ]

    def validate_prediction(self, **kwargs):
        self.calls.append(("validate", kwargs))

    def upload_prediction(self, **kwargs):
        self.calls.append(("upload", kwargs))

        class Model:
            repository = "owner/repo"

        class Result:
            id = 7
            model = Model()
            published = False

        return Result()


def _prediction_row():
    return {
        "date": "2024-01-07",
        "lower_95": 1,
        "lower_90": 2,
        "lower_80": 3,
        "lower_50": 4,
        "pred": 5,
        "upper_50": 6,
        "upper_80": 7,
        "upper_90": 8,
        "upper_95": 9,
    }


def _write_prediction(tmp_path, payload):
    path = tmp_path / "prediction.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _upload_args(path, extra=None):
    args = [
        "upload-prediction",
        "--prediction",
        str(path),
        "--repository",
        "owner/repo",
        "--disease",
        "A90",
        "--description",
        "test forecast",
        "--commit",
        "a" * 40,
        "--adm-level",
        "2",
        "--adm-1",
        "33",
        "--adm-2",
        "3304557",
        "--api-key",
        "user:00000000-0000-4000-8000-000000000000",
    ]
    return args + (extra or [])


def test_infodengue_writes_json(skill, monkeypatch, capsys):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    monkeypatch.setenv("MOSQLIMATE_API_KEY", "user:key")

    code = skill.main(
        [
            "infodengue",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--geocode",
            "3304557",
        ]
    )

    assert code == 0
    assert fake.calls[0][1]["geocode"] == 3304557
    assert json.loads(capsys.readouterr().out) == [{"casos": 1}]


def test_infodengue_requires_location(skill):
    with pytest.raises(SystemExit):
        skill.main(
            [
                "infodengue",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31",
            ]
        )


def test_load_prediction_json(skill, tmp_path):
    path = _write_prediction(tmp_path, [_prediction_row()])

    assert skill._load_prediction(str(path)) == [_prediction_row()]


def test_load_prediction_csv(skill, tmp_path):
    path = tmp_path / "prediction.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=skill.PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerow(_prediction_row())

    rows = skill._load_prediction(str(path))

    assert rows[0]["date"] == "2024-01-07"
    assert rows[0]["pred"] == "5"


def test_load_prediction_rejects_missing_column(skill, tmp_path):
    row = _prediction_row()
    row.pop("pred")
    path = _write_prediction(tmp_path, [row])

    with pytest.raises(ValueError, match="missing columns"):
        skill._load_prediction(str(path))


def test_load_prediction_rejects_empty_value(skill, tmp_path):
    row = _prediction_row()
    row["pred"] = None
    path = _write_prediction(tmp_path, [row])

    with pytest.raises(ValueError, match="empty value"):
        skill._load_prediction(str(path))


def test_upload_requires_yes(skill, monkeypatch, tmp_path):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    path = _write_prediction(tmp_path, [_prediction_row()])

    with pytest.raises(SystemExit, match="--yes"):
        skill.main(_upload_args(path))

    assert fake.calls == []


def test_upload_dry_run_does_not_upload(skill, monkeypatch, tmp_path, capsys):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    path = _write_prediction(tmp_path, [_prediction_row()])

    code = skill.main(_upload_args(path, ["--dry-run"]))

    assert code == 0
    assert [name for name, _ in fake.calls] == ["validate"]
    assert "nothing was uploaded" in capsys.readouterr().out


def test_upload_defaults_to_unpublished(skill, monkeypatch, tmp_path):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    path = _write_prediction(tmp_path, [_prediction_row()])

    code = skill.main(_upload_args(path, ["--yes"]))

    assert code == 0
    assert [name for name, _ in fake.calls] == ["validate", "upload"]
    assert fake.calls[1][1]["published"] is False


def test_upload_can_publish_when_explicit(skill, monkeypatch, tmp_path):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    path = _write_prediction(tmp_path, [_prediction_row()])

    code = skill.main(_upload_args(path, ["--yes", "--published"]))

    assert code == 0
    assert fake.calls[1][1]["published"] is True


def test_infodengue_accepts_disease_code_and_adm_1(skill, monkeypatch, capsys):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    monkeypatch.setenv("MOSQLIMATE_API_KEY", "user:key")

    code = skill.main(
        [
            "infodengue",
            "--disease",
            "A90",
            "--adm-1",
            "33",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert code == 0
    kwargs = fake.calls[0][1]
    assert kwargs["disease"] == "dengue"
    assert kwargs["uf"] == "RJ"
    capsys.readouterr()


def test_predictions_filters_by_adm(skill, monkeypatch, capsys):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    monkeypatch.setattr(
        skill,
        "_prediction_record",
        lambda prediction, include_data=False: {"id": prediction.id},
    )
    monkeypatch.setenv("MOSQLIMATE_API_KEY", "user:key")

    code = skill.main(
        [
            "predictions",
            "--adm-1",
            "RJ",
            "--adm-2",
            "3304557",
        ]
    )

    assert code == 0
    kwargs = fake.calls[0][1]
    assert "adm_1" not in kwargs
    assert "adm_2" not in kwargs
    assert json.loads(capsys.readouterr().out) == [{"id": 3}]


def test_upload_accepts_uf_alias(skill, monkeypatch, tmp_path):
    fake = FakeMosqlient()
    monkeypatch.setattr(skill, "_load_mosqlient", lambda _url: fake)
    path = _write_prediction(tmp_path, [_prediction_row()])

    code = skill.main(_upload_args(path, ["--yes", "--uf", "RJ"]))

    assert code == 0
    kwargs = fake.calls[1][1]
    assert kwargs["adm_1"] == 33
    assert kwargs["disease"] == "A90"


def test_normalizers_accept_aliases(skill):
    assert skill._normalize_disease_code("dengue") == "A90"
    assert skill._normalize_disease_code("A92.0") == "A92.0"
    assert skill._normalize_disease_name("zika") == "zika"
    assert skill._normalize_uf("33") == "RJ"
    assert skill._normalize_uf("rj") == "RJ"
    assert skill._normalize_adm_1("RJ") == 33
