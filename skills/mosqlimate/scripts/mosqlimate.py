#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mosqlient>=2.6,<3",
# ]
# ///
"""Query Mosqlimate data and manage predictions from the command line.

This script is a thin wrapper around the public ``mosqlient`` API. It is
intended to be executed with ``uv run`` so its dependencies stay isolated.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

DISEASES = ("dengue", "zika", "chikungunya")
ICD10 = ("A90", "A92.0", "A92.5")
ADM_LEVELS = (0, 1, 2, 3)
TIME_RESOLUTIONS = ("day", "week", "month", "year")
CATEGORIES = (
    "quantitative",
    "categorical",
    "spatial_quantitative",
    "spatial_categorical",
    "spatio_temporal_quantitative",
    "spatio_temporal_categorical",
)
PREDICTION_COLUMNS = (
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
)
DISEASE_CODES_BY_NAME = {
    "dengue": "A90",
    "zika": "A92.5",
    "chikungunya": "A92.0",
    "chik": "A92.0",
}
DISEASE_NAMES_BY_CODE = {
    "A90": "dengue",
    "A92.0": "chikungunya",
    "A92.5": "zika",
}
DISEASE_CHOICES = tuple(DISEASE_CODES_BY_NAME) + ICD10
UF_CODES = {
    "AC": 12,
    "AL": 27,
    "AM": 13,
    "AP": 16,
    "BA": 29,
    "CE": 23,
    "DF": 53,
    "ES": 32,
    "GO": 52,
    "MA": 21,
    "MG": 31,
    "MS": 50,
    "MT": 51,
    "PA": 15,
    "PB": 25,
    "PE": 26,
    "PI": 22,
    "PR": 41,
    "RJ": 33,
    "RN": 24,
    "RO": 11,
    "RR": 14,
    "RS": 43,
    "SC": 42,
    "SE": 28,
    "SP": 35,
    "TO": 17,
}
CODE_UFS = {code: uf for uf, code in UF_CODES.items()}


def _normalize_disease_code(value: str) -> str:
    key = value.strip().lower()
    if key in DISEASE_CODES_BY_NAME:
        return DISEASE_CODES_BY_NAME[key]
    if key.upper() in DISEASE_NAMES_BY_CODE:
        return key.upper()
    raise SystemExit(
        f"Unknown disease {value}. Use a name (dengue, zika, chikungunya) "
        "or an ICD-10 code (A90, A92.0, A92.5)."
    )


def _normalize_disease_name(value: str) -> str:
    return DISEASE_NAMES_BY_CODE[_normalize_disease_code(value)]


def _normalize_uf(value: str | int) -> str:
    if isinstance(value, int):
        if value in CODE_UFS:
            return CODE_UFS[value]
        raise SystemExit(f"Unknown state code {value}. Example: 33 (RJ).")

    text = value.strip()
    if text.isdigit():
        return _normalize_uf(int(text))
    uf = text.upper()
    if uf in UF_CODES:
        return uf
    raise SystemExit(f"Unknown state {value}. Example: RJ or 33.")


def _normalize_adm_1(value: str | int) -> int:
    return UF_CODES[_normalize_uf(value)]


def _normalize_adm_2(value: int) -> int:
    if len(str(value)) != 7:
        raise SystemExit(
            f"Invalid municipality geocode {value}. Example: 3304557."
        )
    return value


def _load_mosqlient(api_url: str | None):
    """Import mosqlient after applying an optional API URL override."""
    if api_url:
        os.environ["MOSQLIENT_API_URL"] = api_url
    import mosqlient

    return mosqlient


def _get_api_key(
    args: argparse.Namespace, required: bool = False
) -> str | None:
    key = args.api_key or os.environ.get("MOSQLIMATE_API_KEY")
    if not key and required:
        raise SystemExit(
            "An API key is required for this command. Set "
            "MOSQLIMATE_API_KEY or pass --api-key."
        )
    if not key:
        print(
            "warning: no API key found; mosqlient may create a limited "
            "demo user.",
            file=sys.stderr,
        )
    return key


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key",
        default=None,
        help="Mosqlimate API key (username:uuid). Defaults to "
        "MOSQLIMATE_API_KEY.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Override the API base URL (also MOSQLIENT_API_URL).",
    )
    parser.add_argument(
        "--output",
        choices=("csv", "json"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write output to this file instead of stdout.",
    )


def _add_location_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--uf", default=None, help="Brazilian state abbreviation (e.g. RJ)."
    )
    parser.add_argument(
        "--adm-1",
        default=None,
        help="State abbreviation or IBGE state code (alias for --uf).",
    )
    parser.add_argument(
        "--geocode",
        type=int,
        default=None,
        help="IBGE municipality geocode (7 digits).",
    )


def _add_prediction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prediction",
        required=True,
        help="JSON or CSV file with prediction rows.",
    )
    parser.add_argument(
        "--repository",
        required=True,
        help="Model repository in owner/name format.",
    )
    parser.add_argument(
        "--disease",
        required=True,
        choices=DISEASE_CHOICES,
        help="Disease name or ICD-10 code.",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="Text description of the prediction run.",
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="40-character lowercase git commit hash.",
    )
    parser.add_argument(
        "--adm-level",
        type=int,
        required=True,
        choices=ADM_LEVELS,
        help="Administrative level (0-3).",
    )
    parser.add_argument(
        "--case-definition",
        default="probable",
        choices=("probable", "reported"),
        help="Case definition (default: probable).",
    )
    parser.add_argument(
        "--adm-0",
        default="BRA",
        help="Country code (default: BRA).",
    )
    parser.add_argument(
        "--adm-1",
        default=None,
        help="State abbreviation or IBGE state code (required at level 1+).",
    )
    parser.add_argument(
        "--uf",
        default=None,
        help="Alias for --adm-1.",
    )
    parser.add_argument("--adm-2", type=int, default=None)
    parser.add_argument("--adm-3", type=int, default=None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosqlimate",
        description="Query Mosqlimate data and manage predictions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infodengue = subparsers.add_parser(
        "infodengue", help="Fetch InfoDengue cases."
    )
    _add_common_args(infodengue)
    _add_location_args(infodengue)
    infodengue.add_argument(
        "--disease", choices=DISEASE_CHOICES, default="dengue"
    )
    infodengue.add_argument("--start", required=True)
    infodengue.add_argument("--end", required=True)
    infodengue.set_defaults(func=_cmd_infodengue)

    climate = subparsers.add_parser("climate", help="Fetch climate data.")
    _add_common_args(climate)
    _add_location_args(climate)
    climate.add_argument("--start", required=True)
    climate.add_argument("--end", required=True)
    climate.add_argument(
        "--weekly",
        action="store_true",
        help="Use weekly aggregation (start/end in YYYYWW format).",
    )
    climate.add_argument(
        "--macro-health-code",
        type=int,
        default=None,
        help="Macro health geocode (weekly only).",
    )
    climate.set_defaults(func=_cmd_climate)

    episcanner = subparsers.add_parser(
        "episcanner", help="Fetch Episcanner data."
    )
    _add_common_args(episcanner)
    episcanner.add_argument("--uf", default=None)
    episcanner.add_argument("--adm-1", default=None)
    episcanner.add_argument(
        "--disease", choices=DISEASE_CHOICES, default="dengue"
    )
    episcanner.add_argument("--year", type=int, default=None)
    episcanner.set_defaults(func=_cmd_episcanner)

    mosquito = subparsers.add_parser(
        "mosquito", help="Fetch mosquito (ContaOvos) data."
    )
    _add_common_args(mosquito)
    mosquito.add_argument("--date-start", default=None)
    mosquito.add_argument("--date-end", default=None)
    mosquito.add_argument("--state", default=None)
    mosquito.add_argument("--uf", default=None)
    mosquito.add_argument("--adm-1", default=None)
    mosquito.add_argument("--municipality", default=None)
    mosquito.add_argument("--page", type=int, default=None)
    mosquito.set_defaults(func=_cmd_mosquito)

    vegetation = subparsers.add_parser(
        "vegetation", help="Fetch vegetation index data."
    )
    _add_common_args(vegetation)
    _add_location_args(vegetation)
    vegetation.add_argument("--start", required=True)
    vegetation.add_argument("--end", required=True)
    vegetation.add_argument("--collection", default=None)
    vegetation.add_argument("--attribute", default=None)
    vegetation.set_defaults(func=_cmd_vegetation)

    models = subparsers.add_parser(
        "models", help="List or search registered models."
    )
    _add_common_args(models)
    models.add_argument("--id", type=int, default=None)
    models.add_argument("--repository-owner", default=None)
    models.add_argument("--repository-organization", default=None)
    models.add_argument("--repository-name", default=None)
    models.add_argument("--disease", choices=DISEASE_CHOICES, default=None)
    models.add_argument("--category", choices=CATEGORIES, default=None)
    models.add_argument(
        "--adm-level", type=int, choices=ADM_LEVELS, default=None
    )
    models.add_argument(
        "--time-resolution",
        choices=TIME_RESOLUTIONS,
        default=None,
    )
    models.add_argument("--imdc-year", type=int, default=None)
    models.set_defaults(func=_cmd_models)

    predictions = subparsers.add_parser(
        "predictions", help="List or search predictions."
    )
    _add_common_args(predictions)
    predictions.add_argument("--id", type=int, default=None)
    predictions.add_argument("--model-id", type=int, default=None)
    predictions.add_argument("--model-name", default=None)
    predictions.add_argument("--model-owner", default=None)
    predictions.add_argument("--model-organization", default=None)
    predictions.add_argument(
        "--adm-level", type=int, choices=ADM_LEVELS, default=None
    )
    predictions.add_argument(
        "--model-time-resolution",
        choices=TIME_RESOLUTIONS,
        default=None,
    )
    predictions.add_argument(
        "--disease", choices=DISEASE_CHOICES, default=None
    )
    predictions.add_argument(
        "--model-category", choices=CATEGORIES, default=None
    )
    predictions.add_argument("--imdc-year", type=int, default=None)
    predictions.add_argument("--start", default=None)
    predictions.add_argument("--end", default=None)
    predictions.add_argument(
        "--adm-1",
        default=None,
        help="Filter by state abbreviation or IBGE state code.",
    )
    predictions.add_argument(
        "--adm-2",
        type=int,
        default=None,
        help="Filter by municipality IBGE geocode (7 digits).",
    )
    predictions.add_argument(
        "--include-data",
        action="store_true",
        help="Fetch and include prediction data rows.",
    )
    predictions.set_defaults(func=_cmd_predictions)

    validate = subparsers.add_parser(
        "validate-prediction", help="Validate a prediction without uploading."
    )
    _add_common_args(validate)
    _add_prediction_args(validate)
    validate.set_defaults(func=_cmd_validate_prediction)

    upload = subparsers.add_parser(
        "upload-prediction", help="Validate and upload a prediction."
    )
    _add_common_args(upload)
    _add_prediction_args(upload)
    upload.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation for the write operation.",
    )
    upload.add_argument(
        "--published",
        action="store_true",
        help="Publish the prediction (default: upload as unpublished).",
    )
    upload.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; do not upload.",
    )
    upload.set_defaults(func=_cmd_upload_prediction)

    return parser


def _resolve_uf(args: argparse.Namespace) -> str | None:
    value = args.uf if args.uf is not None else args.adm_1
    return _normalize_uf(value) if value is not None else None


def _require_location(args: argparse.Namespace) -> None:
    if not (args.uf or args.adm_1 or args.geocode):
        raise SystemExit("Provide --uf, --adm-1, or --geocode.")


def _require_state(args: argparse.Namespace) -> None:
    if not (args.uf or args.adm_1):
        raise SystemExit("Provide --uf or --adm-1.")


def _records(data: Any) -> list[dict[str, Any]]:
    if hasattr(data, "to_dict"):
        return data.to_dict(orient="records")
    return list(data)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value


def _to_csv(records: Sequence[dict[str, Any]]) -> str:
    if not records:
        return ""
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, extrasaction="ignore"
    )
    writer.writeheader()
    for record in records:
        writer.writerow(
            {key: _csv_value(value) for key, value in record.items()}
        )
    return buffer.getvalue()


def _write_records(
    records: Iterable[dict[str, Any]],
    output: str,
    out: str | None,
) -> None:
    rows = list(records)
    if output == "json":
        text = json.dumps(rows, indent=2, default=str)
    else:
        text = _to_csv(rows)
    if not text.endswith("\n"):
        text += "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
        print(f"{len(rows)} rows", file=sys.stderr)


def _model_record(model: Any) -> dict[str, Any]:
    return {
        "id": model.id,
        "repository": model.repository,
        "description": model.description,
        "category": model.category,
        "time_resolution": model.time_resolution,
        "imdc_year": model.imdc_year,
        "predictions_count": model.predictions_count,
        "active": model.active,
        "created_at": model.created_at,
        "last_update": model.last_update,
    }


def _prediction_record(
    prediction: Any, include_data: bool = False
) -> dict[str, Any]:
    record = {
        "id": prediction.id,
        "model_id": prediction.model.id,
        "repository": prediction.model.repository,
        "disease": prediction.disease,
        "description": prediction.description,
        "commit": prediction.commit,
        "case_definition": prediction.case_definition,
        "published": prediction.published,
        "start": prediction.start,
        "end": prediction.end,
        "adm_level": prediction.adm_level,
        "adm_0": prediction.adm_0,
        "adm_1": prediction.adm_1,
        "adm_2": prediction.adm_2,
        "adm_3": prediction.adm_3,
        "scores": dict(prediction.scores or {}),
    }
    if include_data:
        record["data"] = [
            row.model_dump(mode="json") for row in prediction.data
        ]
    return record


def _load_prediction(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Prediction file not found: {path}")
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get("prediction", payload.get("data"))
        if not isinstance(payload, list):
            raise ValueError("JSON must contain a list of prediction rows.")
        rows = payload
    elif suffix == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Prediction file must be .json or .csv.")

    if not rows:
        raise ValueError("Prediction file is empty.")

    expected = set(PREDICTION_COLUMNS)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not an object.")
        missing = expected.difference(row)
        if missing:
            raise ValueError(
                f"Row {index} is missing columns: {sorted(missing)}"
            )
        extra = set(row).difference(expected)
        if extra:
            raise ValueError(
                f"Row {index} has unknown columns: {sorted(extra)}"
            )
        for column in PREDICTION_COLUMNS:
            if row[column] in (None, ""):
                raise ValueError(
                    f"Row {index} has an empty value for {column}."
                )
    return rows


def _prediction_kwargs(
    args: argparse.Namespace, published: bool = False
) -> dict[str, Any]:
    adm_1 = args.adm_1 if args.adm_1 is not None else args.uf
    return {
        "repository": args.repository,
        "disease": _normalize_disease_code(args.disease),
        "description": args.description,
        "commit": args.commit,
        "adm_level": args.adm_level,
        "case_definition": args.case_definition,
        "published": published,
        "adm_0": args.adm_0,
        "adm_1": _normalize_adm_1(adm_1) if adm_1 is not None else None,
        "adm_2": args.adm_2,
        "adm_3": args.adm_3,
    }


def _cmd_infodengue(args: argparse.Namespace) -> int:
    _require_location(args)
    mosqlient = _load_mosqlient(args.api_url)
    data = mosqlient.get_infodengue(
        api_key=_get_api_key(args),
        disease=_normalize_disease_name(args.disease),
        start_date=args.start,
        end_date=args.end,
        uf=_resolve_uf(args),
        geocode=args.geocode,
    )
    _write_records(_records(data), args.output, args.out)
    return 0


def _cmd_climate(args: argparse.Namespace) -> int:
    _require_location(args)
    mosqlient = _load_mosqlient(args.api_url)
    if args.weekly:
        data = mosqlient.get_climate_weekly(
            api_key=_get_api_key(args),
            start=args.start,
            end=args.end,
            uf=_resolve_uf(args),
            geocode=args.geocode,
            macro_health_code=args.macro_health_code,
        )
    else:
        data = mosqlient.get_climate(
            api_key=_get_api_key(args),
            start_date=args.start,
            end_date=args.end,
            uf=_resolve_uf(args),
            geocode=args.geocode,
        )
    _write_records(_records(data), args.output, args.out)
    return 0


def _cmd_episcanner(args: argparse.Namespace) -> int:
    _require_state(args)
    mosqlient = _load_mosqlient(args.api_url)
    kwargs: dict[str, Any] = {
        "api_key": _get_api_key(args),
        "uf": _resolve_uf(args),
        "disease": _normalize_disease_name(args.disease),
    }
    if args.year is not None:
        kwargs["year"] = args.year
    data = mosqlient.get_episcanner(**kwargs)
    _write_records(_records(data), args.output, args.out)
    return 0


def _cmd_mosquito(args: argparse.Namespace) -> int:
    mosqlient = _load_mosqlient(args.api_url)
    state = args.state if args.state is not None else args.uf
    if state is None:
        state = args.adm_1
    data = mosqlient.get_mosquito(
        api_key=_get_api_key(args),
        date_start=args.date_start,
        date_end=args.date_end,
        state=_normalize_uf(state) if state is not None else None,
        municipality=args.municipality,
        page=args.page,
    )
    _write_records(_records(data), args.output, args.out)
    return 0


def _cmd_vegetation(args: argparse.Namespace) -> int:
    _require_location(args)
    mosqlient = _load_mosqlient(args.api_url)
    data = mosqlient.get_vegetation(
        api_key=_get_api_key(args),
        start_date=args.start,
        end_date=args.end,
        uf=_resolve_uf(args),
        geocode=args.geocode,
        collection=args.collection,
        attribute=args.attribute,
    )
    _write_records(_records(data), args.output, args.out)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    mosqlient = _load_mosqlient(args.api_url)
    models = mosqlient.get_models(
        api_key=_get_api_key(args, required=True),
        id=args.id,
        repository_owner=args.repository_owner,
        repository_organization=args.repository_organization,
        repository_name=args.repository_name,
        disease=(
            _normalize_disease_code(args.disease)
            if args.disease is not None
            else None
        ),
        category=args.category,
        adm_level=args.adm_level,
        time_resolution=args.time_resolution,
        imdc_year=args.imdc_year,
    )
    _write_records(
        (_model_record(model) for model in models),
        args.output,
        args.out,
    )
    return 0


def _cmd_predictions(args: argparse.Namespace) -> int:
    mosqlient = _load_mosqlient(args.api_url)
    predictions = mosqlient.get_predictions(
        api_key=_get_api_key(args, required=True),
        id=args.id,
        model_id=args.model_id,
        model_name=args.model_name,
        model_owner=args.model_owner,
        model_organization=args.model_organization,
        adm_level=args.adm_level,
        model_time_resolution=args.model_time_resolution,
        disease=(
            _normalize_disease_code(args.disease)
            if args.disease is not None
            else None
        ),
        model_category=args.model_category,
        imdc_year=args.imdc_year,
        start=args.start,
        end=args.end,
    )

    if args.adm_1 is not None:
        adm_1 = _normalize_adm_1(args.adm_1)
        predictions = [
            prediction
            for prediction in predictions
            if prediction.adm_1 == adm_1
        ]
    if args.adm_2 is not None:
        adm_2 = _normalize_adm_2(args.adm_2)
        predictions = [
            prediction
            for prediction in predictions
            if prediction.adm_2 == adm_2
        ]

    _write_records(
        (
            _prediction_record(prediction, args.include_data)
            for prediction in predictions
        ),
        args.output,
        args.out,
    )
    return 0


def _cmd_validate_prediction(args: argparse.Namespace) -> int:
    mosqlient = _load_mosqlient(args.api_url)
    data = _load_prediction(args.prediction)
    mosqlient.validate_prediction(
        api_key=_get_api_key(args, required=True),
        prediction=data,
        **_prediction_kwargs(args),
    )
    print("Prediction is valid.")
    return 0


def _cmd_upload_prediction(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.yes:
        raise SystemExit(
            "Refusing to upload without --yes. Re-run with --dry-run to "
            "validate only, or --yes to upload."
        )
    mosqlient = _load_mosqlient(args.api_url)
    key = _get_api_key(args, required=True)
    data = _load_prediction(args.prediction)
    kwargs = _prediction_kwargs(args, published=args.published)
    mosqlient.validate_prediction(
        api_key=key,
        prediction=data,
        **kwargs,
    )
    if args.dry_run:
        print("Validation passed; nothing was uploaded (--dry-run).")
        return 0
    result = mosqlient.upload_prediction(
        api_key=key,
        prediction=data,
        **kwargs,
    )
    summary = {
        "id": result.id,
        "repository": result.model.repository,
        "published": result.published,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
