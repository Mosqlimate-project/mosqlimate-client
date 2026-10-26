# Registry commands

The registry lists models and their predictions. Registry reads require a real
API key; demo users may have restricted access.

## models

List all models:

```bash
uv run scripts/mosqlimate.py models
```

Search with filters:

```bash
uv run scripts/mosqlimate.py models \
  --disease A90 --imdc-year 2024 --adm-level 1
```

| Flag | Notes |
| --- | --- |
| `--id` | Model id. |
| `--repository-owner` | GitHub user that owns the model repository. |
| `--repository-organization` | GitHub organization that owns it. |
| `--repository-name` | Repository name without the owner. |
| `--disease` | `A90`, `A92.0`, or `A92.5`. |
| `--category` | `quantitative`, `categorical`, `spatial_quantitative`, `spatial_categorical`, `spatio_temporal_quantitative`, or `spatio_temporal_categorical`. |
| `--adm-level` | `0`, `1`, `2`, or `3`. |
| `--time-resolution` | `day`, `week`, `month`, or `year`. |
| `--imdc-year` | IMDC competition year. |

Each result includes the model id, repository, description, category, time
resolution, IMDC year, prediction count, active flag, and timestamps.

## predictions

List predictions:

```bash
uv run scripts/mosqlimate.py predictions --disease A90 --start 2024-01-01
```

Filter by model:

```bash
uv run scripts/mosqlimate.py predictions \
  --model-owner owner --model-name repo --include-data
```

| Flag | Notes |
| --- | --- |
| `--id` | Prediction id. |
| `--model-id` | Model id. |
| `--model-name` | Repository name. |
| `--model-owner` | Repository owner. |
| `--model-organization` | Repository organization. |
| `--adm-level` | `0`, `1`, `2`, or `3`. |
| `--model-time-resolution` | `day`, `week`, `month`, or `year`. |
| `--disease` | `A90`, `A92.0`, or `A92.5`. |
| `--model-category` | Same values as the models command. |
| `--imdc-year` | IMDC competition year. |
| `--start` | Predictions starting on or after this date. |
| `--end` | Predictions ending on or before this date. |
| `--include-data` | Fetch and include all prediction rows. |

Without `--include-data`, each result contains metadata only: id, model id and
repository, disease, description, commit, case definition, published flag,
start and end dates, administrative fields, and scores.

`--include-data` performs one additional request per prediction. Use it for a
single prediction or a small filtered list, not for a broad search.

## Disease codes

| Code | Disease |
| --- | --- |
| `A90` | Dengue |
| `A92.0` | Chikungunya |
| `A92.5` | Zika |

Datastore commands use names (`dengue`, `zika`, `chikungunya`); registry
commands use these ICD-10 codes.

## Administrative levels

| Level | Meaning |
| --- | --- |
| `0` | Country (`adm_0`, usually `BRA`) |
| `1` | State (`adm_1`) |
| `2` | Municipality (`adm_2`, 7-digit IBGE geocode) |
| `3` | Sub-municipality (`adm_3`) |
