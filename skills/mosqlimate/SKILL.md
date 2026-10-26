---
name: mosqlimate
description: Query Mosqlimate dengue, zika, and chikungunya data, models, and forecasts, and validate or upload predictions with mosqlient.
version: 1.0.0
homepage: https://github.com/Mosqlimate-project/mosqlimate-client
metadata:
  openclaw:
    requires:
      bins:
        - uv
    primaryEnv: MOSQLIMATE_API_KEY
    envVars:
      - name: MOSQLIMATE_API_KEY
        required: false
        description: Mosqlimate API key in username:uuid format. Without it mosqlient may create a limited demo user.
      - name: MOSQLIENT_API_URL
        required: false
        description: Override the API base URL, useful for staging.
    install:
      - kind: uv
        package: mosqlient>=2.6,<3
---

# Mosqlimate

Use this skill for Brazilian arbovirus data and forecasting work through the
[Mosqlimate platform](https://api.mosqlimate.org/). It wraps the public
`mosqlient` Python library and keeps write operations behind an explicit
confirmation.

## When to use it

Use this skill when the user asks for:

- dengue, zika, or chikungunya cases in a Brazilian state or municipality;
- climate, mosquito (ContaOvos), vegetation, or Episcanner data;
- Mosqlimate model or prediction searches;
- validation or upload of a forecast to the Mosqlimate registry.

Do not use it to invent epidemiological values, to guess IBGE geocodes, or for
countries other than Brazil.

## Setup

Run the helper with `uv`; it installs `mosqlient` in an isolated environment:

```bash
uv run scripts/mosqlimate.py --help
```

From the repository root the path is:

```bash
uv run skills/mosqlimate/scripts/mosqlimate.py --help
```

The base install covers data access and the registry. Forecasting and scoring
helpers need the `mosqlient[analyze]` extra and are not part of this skill. The
first `uv run` can take several minutes because `mosqlient` has a large
scientific dependency tree; later runs reuse the cache.

## Authentication

Set the API key in `MOSQLIMATE_API_KEY` or pass `--api-key`. The format is
`username:uuid`.

```bash
export MOSQLIMATE_API_KEY="your-user:00000000-0000-4000-8000-000000000000"
```

Never print, log, or commit the key. If no key is provided, `mosqlient` may
create a temporary demo user with limited access and rate limits.

## Commands

| Command | Purpose |
| --- | --- |
| `infodengue` | Fetch InfoDengue cases. |
| `climate` | Fetch daily climate data; `--weekly` for weekly data. |
| `episcanner` | Fetch Episcanner data. |
| `mosquito` | Fetch ContaOvos mosquito data. |
| `vegetation` | Fetch vegetation index data. |
| `models` | List or search registered models. |
| `predictions` | List or search predictions. |
| `validate-prediction` | Validate a forecast without uploading. |
| `upload-prediction` | Validate and upload a forecast. |

All read commands accept `--output csv|json` and `--out FILE`.

## Examples

Dengue cases for Rio de Janeiro city:

```bash
uv run scripts/mosqlimate.py infodengue \
  --disease dengue \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --geocode 3304557 \
  --output csv --out dengue_rj.csv
```

All municipalities in Parana state:

```bash
uv run scripts/mosqlimate.py infodengue \
  --disease dengue --start 2023-01-01 --end 2023-06-30 --uf PR
```

Daily climate for Curitiba:

```bash
uv run scripts/mosqlimate.py climate \
  --start 2023-01-01 --end 2023-01-31 --geocode 4106902
```

Weekly climate for a macro health region:

```bash
uv run scripts/mosqlimate.py climate --weekly \
  --start 202501 --end 202510 --macro-health-code 1101
```

2024 IMDC dengue models:

```bash
uv run scripts/mosqlimate.py models --disease A90 --imdc-year 2024
```

One prediction and its data:

```bash
uv run scripts/mosqlimate.py predictions --id 300 --include-data
```

Validate a forecast:

```bash
uv run scripts/mosqlimate.py validate-prediction \
  --prediction forecast.json \
  --repository owner/repo \
  --disease A90 \
  --description "Out-of-sample forecast for Rio de Janeiro" \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --adm-level 2 --adm-2 3304557
```

Upload an unpublished forecast:

```bash
uv run scripts/mosqlimate.py upload-prediction \
  --prediction forecast.json \
  --repository owner/repo \
  --disease A90 \
  --description "Out-of-sample forecast for Rio de Janeiro" \
  --commit 0123456789abcdef0123456789abcdef01234567 \
  --adm-level 2 --adm-2 3304557 \
  --dry-run          # remove this line to upload
  --yes
```

`--published` is required to make an uploaded prediction public. Without it the
prediction is uploaded as unpublished.

## Prediction file format

JSON files must contain a list of rows, or an object with a `prediction` or
`data` list. CSV files must use the same column names. Every row must contain:

```text
date, lower_95, lower_90, lower_80, lower_50, pred,
upper_50, upper_80, upper_90, upper_95
```

Dates use `YYYY-MM-DD`. For weekly models, each date must be the Sunday that
starts the CDC epidemiological week. IMDC predictions must cover every week in
the required range without gaps. Bounds must be non-negative and ordered:
`lower_95 <= lower_90 <= lower_80 <= lower_50 <= pred <= upper_50 <=
upper_80 <= upper_90 <= upper_95`.

## Safety rules

- Never invent case counts, dates, geocodes, or model identifiers. Fetch them
  and report the parameters used.
- Call `validate-prediction` or use `upload-prediction --dry-run` before any
  upload.
- Do not run `upload-prediction` without explicit user confirmation. The script
  refuses to write without `--yes`.
- Do not add `--published` unless the user explicitly asks to publish.
- Do not modify or delete existing predictions from this skill.
- Treat HTTP 422 as a validation problem and show the API message. Back off on
  HTTP 429 and surface the rate limit.
- Do not pass secrets on the command line in shared shells; prefer the
  environment variable.

## Conventions

- Datastore disease names: `dengue`, `zika`, `chikungunya`.
- Registry disease codes: `A90` (dengue), `A92.0` (chikungunya), `A92.5`
  (zika).
- Administrative levels: 0 country, 1 state, 2 municipality, 3 sub-municipality.
- Municipality geocodes are 7-digit IBGE codes. State filters use the two-letter
  UF code.

## References

- [Datastore commands](references/datastore.md)
- [Registry commands](references/registry.md)
- [Prediction uploads](references/predictions.md)
- [Troubleshooting](references/troubleshooting.md)
- [Mosqlimate API documentation](https://api.mosqlimate.org/docs/)
