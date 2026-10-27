# Datastore commands

All commands print JSON by default. Use `--output csv` and `--out FILE` to save
a CSV. A one-line row count goes to stderr so stdout stays parseable.

Base URL: `https://api.mosqlimate.org/api/`

| Command | Method | Partial endpoint | Output |
| --- | --- | --- | --- |
| `infodengue` | `GET` | `datastore/infodengue/` | Paged `HistoricoAlertaSchema`. |
| `climate` | `GET` | `datastore/climate/` | Paged `CopernicusBrasilSchema`. |
| `climate --weekly` | `GET` | `datastore/climate/weekly/` | Paged `CopernicusBrasilWeeklySchema`. |
| `episcanner` | `GET` | `datastore/episcanner/` | `{"items": [EpiScannerSchema]}`. |
| `mosquito` | `GET` | `datastore/mosquito/` | `{"items": [ContaOvosSchema]}`. |
| `vegetation` | `GET` | `datastore/vegetation/` | Paged `VegetationIndexMetricSchema`. |

Full field schemas are in [api.md](api.md#datastore).

## infodengue

Fetch notified and estimated arbovirus cases.

```bash
uv run scripts/mosqlimate.py infodengue \
  --disease dengue \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --uf RJ
```

| Flag | Required | Notes |
| --- | --- | --- |
| `--disease` | no | `dengue` (default), `zika`, `chikungunya`, or `A90`, `A92.5`, `A92.0`. |
| `--start` | yes | `YYYY-MM-DD`. |
| `--end` | yes | `YYYY-MM-DD`. |
| `--uf` | one of | Two-letter state code or IBGE state code. |
| `--adm-1` | one of | Alias for `--uf`; accepts a UF or state code. |
| `--geocode` | one of | 7-digit IBGE municipality code. |

`--geocode` takes precedence when both are set. The API returns one row per
epidemiological week and location, including `data_iniSE`, `SE`, `casos`,
`casos_est`, `p_inc100k` and climate columns.

## climate

Daily climate data:

```bash
uv run scripts/mosqlimate.py climate \
  --start 2023-01-01 --end 2023-01-31 --geocode 4106902
```

Weekly climate data uses `YYYYWW` weeks:

```bash
uv run scripts/mosqlimate.py climate --weekly \
  --start 202501 --end 202510 --macro-health-code 1101
```

| Flag | Required | Notes |
| --- | --- | --- |
| `--start` | yes | `YYYY-MM-DD`, or `YYYYWW` with `--weekly`. |
| `--end` | yes | `YYYY-MM-DD`, or `YYYYWW` with `--weekly`. |
| `--uf` | one of | Two-letter state code or IBGE state code. |
| `--adm-1` | one of | Alias for `--uf`; accepts a UF or state code. |
| `--geocode` | one of | 7-digit IBGE municipality code. |
| `--weekly` | no | Use weekly aggregation. |
| `--macro-health-code` | no | 4-digit macro health geocode (weekly only). |

## episcanner

```bash
uv run scripts/mosqlimate.py episcanner --uf SP --disease dengue --year 2024
```

| Flag | Required | Notes |
| --- | --- | --- |
| `--uf` | yes* | Two-letter state code or IBGE state code. |
| `--adm-1` | yes* | Alias for `--uf`; accepts a UF or state code. |
| `--disease` | no | `dengue` (default), `zika`, `chikungunya`, or `A90`, `A92.5`, `A92.0`. |
| `--year` | no | Defaults to the current year in the API. |

*Provide one of `--uf` or `--adm-1`.

## mosquito

ContaOvos mosquito surveillance data.

```bash
uv run scripts/mosqlimate.py mosquito \
  --date-start 2024-01-01 --date-end 2024-12-31 --state MG
```

| Flag | Required | Notes |
| --- | --- | --- |
| `--date-start` | no | `YYYY-MM-DD`. |
| `--date-end` | no | `YYYY-MM-DD`. |
| `--state` | no | Two-letter state code or IBGE state code. |
| `--uf` / `--adm-1` | no | Aliases for `--state`. |
| `--municipality` | no | Municipality name, for example `Ponta Pora`. |
| `--page` | no | Pagination page. |

## vegetation

```bash
uv run scripts/mosqlimate.py vegetation \
  --start 2023-01-01 --end 2023-12-31 \
  --geocode 3304557 --attribute ndvi
```

| Flag | Required | Notes |
| --- | --- | --- |
| `--start` | yes | `YYYY-MM-DD`. |
| `--end` | yes | `YYYY-MM-DD`. |
| `--uf` | one of | Two-letter state code or IBGE state code. |
| `--adm-1` | one of | Alias for `--uf`; accepts a UF or state code. |
| `--geocode` | one of | 7-digit IBGE municipality code. |
| `--collection` | no | Satellite collection, for example `modis`. |
| `--attribute` | no | Index or band, for example `ndvi` or `evi`. |

## Output tips

- Prefer CSV for large time series and JSON for small exploratory queries.
- `--out` writes the file and still prints the row count to stderr.
- Use `jq` or pandas after the command for aggregation; this script only fetches
  and serializes data.
