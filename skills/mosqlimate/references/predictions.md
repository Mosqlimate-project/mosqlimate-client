# Prediction uploads

Uploads write to the Mosqlimate registry. Always validate first, always ask the
user, and keep predictions unpublished unless the user explicitly asks to
publish.

## Required row schema

Every prediction row must contain exactly these columns:

| Column | Type | Notes |
| --- | --- | --- |
| `date` | date | `YYYY-MM-DD`. |
| `lower_95` | number | 95% lower bound. |
| `lower_90` | number | 90% lower bound. |
| `lower_80` | number | 80% lower bound. |
| `lower_50` | number | 50% lower bound. |
| `pred` | number | Point forecast. |
| `upper_50` | number | 50% upper bound. |
| `upper_80` | number | 80% upper bound. |
| `upper_90` | number | 90% upper bound. |
| `upper_95` | number | 95% upper bound. |

Bounds must be non-negative and ordered:

```text
lower_95 <= lower_90 <= lower_80 <= lower_50 <= pred
         <= upper_50 <= upper_80 <= upper_90 <= upper_95
```

## File formats

JSON list:

```json
[
  {
    "date": "2024-01-07",
    "lower_95": 100,
    "lower_90": 120,
    "lower_80": 140,
    "lower_50": 160,
    "pred": 180,
    "upper_50": 200,
    "upper_80": 220,
    "upper_90": 240,
    "upper_95": 260
  }
]
```

JSON object wrappers are also accepted:

```json
{"prediction": [ ... ]}
```

or:

```json
{"data": [ ... ]}
```

CSV uses the same column names with a header row.

## Validate

```bash
uv run scripts/mosqlimate.py validate-prediction \
  --prediction forecast.json \
  --repository owner/repo \
  --disease A90 \
  --description "Out-of-sample forecast for Rio de Janeiro" \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --adm-level 2 --adm-1 33 --adm-2 3304557
```

Validation checks the model lookup, schema, bounds, dates, weekly gaps, and
administrative hierarchy. It does not write anything.

## Upload

First run with `--dry-run`:

```bash
uv run scripts/mosqlimate.py upload-prediction \
  --prediction forecast.json \
  --repository owner/repo \
  --disease A90 \
  --description "Out-of-sample forecast for Rio de Janeiro" \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --adm-level 2 --adm-1 33 --adm-2 3304557 \
  --dry-run
```

Then, only after the user confirms, run the same command without `--dry-run`
and with `--yes`:

```bash
uv run scripts/mosqlimate.py upload-prediction \
  --prediction forecast.json \
  --repository owner/repo \
  --disease A90 \
  --description "Out-of-sample forecast for Rio de Janeiro" \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --adm-level 2 --adm-1 33 --adm-2 3304557 \
  --yes
```

Add `--published` only when the user explicitly asks for a public prediction.
Without it the prediction is uploaded as unpublished.

The command prints a small JSON summary with the new prediction id, repository,
and published flag.

## Case definition and geography

| Flag | Default | Notes |
| --- | --- | --- |
| `--case-definition` | `probable` | `probable` or `reported`. |
| `--adm-0` | `BRA` | Country code. |
| `--adm-1` | none | State abbreviation or IBGE state code; required at level 1 or higher. |
| `--uf` | none | Alias for `--adm-1`. |
| `--adm-2` | none | Required at level 2 or higher. |
| `--adm-3` | none | Required at level 3. |

The `--repository` value is `owner/name`, matching the model repository already
registered on the platform. Model registration is done through the Mosqlimate
web platform, not this skill.

## Weekly and IMDC rules

- For weekly models, each date must be the Sunday that starts the CDC week.
- Dates must not contain gaps for weekly or daily models.
- IMDC predictions must cover every week in the competition range, from week 41
  of the previous year through week 40 of the current year.
- Duplicate dates are rejected.

## Safety checklist

1. Confirm the model repository exists.
2. Confirm disease, administrative level, and geocode with the user. Disease
   can be a name or an ICD-10 code; `--adm-1` accepts a UF or state code.
3. Run validation or `--dry-run`.
4. Summarize the row count, date range, and whether the upload is published.
5. Ask for explicit confirmation.
6. Run with `--yes`, leaving `--published` off unless requested.
