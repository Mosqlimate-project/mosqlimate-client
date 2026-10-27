# API endpoints and output schemas

Base URL:

```text
https://api.mosqlimate.org/api/
```

Build an endpoint URL by appending a partial path below, for example:

```text
https://api.mosqlimate.org/api/datastore/infodengue/
```

The helper removes paged envelopes and returns rows as a DataFrame or records.
`--out` serializes those rows as CSV or JSON. Field names match the API exactly.

Interactive reference: <https://api.mosqlimate.org/api/docs/>
OpenAPI document: <https://api.mosqlimate.org/api/openapi.json>

## Method summary

| Method | Partial endpoint | Purpose | Output |
| --- | --- | --- | --- |
| `GET` | `datastore/infodengue/` | InfoDengue cases. | Paged `HistoricoAlertaSchema`. |
| `GET` | `datastore/climate/` | Daily climate. | Paged `CopernicusBrasilSchema`. |
| `GET` | `datastore/climate/weekly/` | Weekly climate. | Paged `CopernicusBrasilWeeklySchema`. |
| `GET` | `datastore/episcanner/` | Episcanner estimates. | `{"items": [EpiScannerSchema]}`. |
| `GET` | `datastore/mosquito/` | ContaOvos data. | `{"items": [ContaOvosSchema]}`. |
| `GET` | `datastore/vegetation/` | Vegetation metrics. | Paged `VegetationIndexMetricSchema`. |
| `GET` | `registry/models/` | List/search models. | Paged `Model`. |
| `GET` | `registry/models/{owner}/{repository}/` | One model by repository. | `Model`. |
| `GET` | `registry/predictions/` | List/search predictions. | Paged `Prediction`. |
| `POST` | `registry/predictions/` | Upload a prediction. | `{"id": integer}`. |
| `GET` | `registry/predictions/{id}/` | One prediction with data. | `PredictionDetail`. |
| `GET` | `registry/predictions/{id}/data/` | Prediction rows. | `[PredictionData]`. |
| `DELETE` | `registry/predictions/{id}/` | Delete a prediction. | Object; `403`/`404` use `{"message": string}`. |
| `PATCH` | `registry/prediction/{prediction_id}/publish/` | Change publication status. | `201` string; `403`/`404` use `{"message": string}`. |

There are no `PUT` endpoints for these resources.

## Response envelopes

Paged datastore and registry list endpoints return:

```json
{
  "items": [],
  "pagination": {
    "items": 0,
    "total_items": 0,
    "page": 1,
    "total_pages": 0,
    "per_page": 300
  }
}
```

`episcanner`, `mosquito`, and `predictions/{id}/data/` return an array or an
`{"items": [...]}` envelope. Single-record and write responses are plain objects.

Fields marked `| null` may be omitted or null. Dates use `YYYY-MM-DD`.

## Datastore

### GET `datastore/infodengue/`

Full URL: `https://api.mosqlimate.org/api/datastore/infodengue/`

Paged item (`HistoricoAlertaSchema`):

| Field | Type | Notes |
| --- | --- | --- |
| `data_iniSE` | date | Epidemiological week start. |
| `SE` | integer \| null | Epidemiological week number. |
| `casos_est` | number \| null | Estimated cases. |
| `casos_est_min` | integer \| null | Lower estimate. |
| `casos_est_max` | integer \| null | Upper estimate. |
| `casos` | integer \| null | Reported cases. |
| `municipio_geocodigo` | integer \| null | IBGE municipality code. |
| `p_rt1` | number \| null | Probability of Rt > 1. |
| `p_inc100k` | number \| null | Incidence per 100k. |
| `Localidade_id` | integer \| null | Localidade identifier. |
| `nivel` | integer \| null | Administrative level. |
| `id` | integer \| null | Record identifier. |
| `versao_modelo` | string \| null | Model version. |
| `Rt` | number \| null | Effective reproduction number. |
| `municipio_nome` | string \| null | Municipality name. |
| `pop` | number \| null | Population. |
| `tempmin` | number \| null | Minimum temperature. |
| `umidmax` | number \| null | Maximum humidity. |
| `receptivo` | integer \| null | Receptivity indicator. |
| `transmissao` | integer \| null | Transmission indicator. |
| `nivel_inc` | integer \| null | Incidence level. |
| `umidmed` | number \| null | Median humidity. |
| `umidmin` | number \| null | Minimum humidity. |
| `tempmed` | number \| null | Median temperature. |
| `tempmax` | number \| null | Maximum temperature. |
| `casprov` | integer \| null | Probable cases. |
| `casprov_est` | number \| null | Estimated probable cases. |
| `casprov_est_min` | integer \| null | Lower probable estimate. |
| `casprov_est_max` | integer \| null | Upper probable estimate. |
| `casconf` | integer \| null | Confirmed cases. |

### GET `datastore/climate/`

Full URL: `https://api.mosqlimate.org/api/datastore/climate/`

Paged item (`CopernicusBrasilSchema`):

| Field | Type |
| --- | --- |
| `date` | date |
| `geocodigo` | integer |
| `epiweek` | integer |
| `temp_min` | number |
| `temp_med` | number |
| `temp_max` | number |
| `precip_min` | number |
| `precip_med` | number |
| `precip_max` | number |
| `precip_tot` | number |
| `pressao_min` | number |
| `pressao_med` | number |
| `pressao_max` | number |
| `umid_min` | number |
| `umid_med` | number |
| `umid_max` | number |

### GET `datastore/climate/weekly/`

Full URL: `https://api.mosqlimate.org/api/datastore/climate/weekly/`

Paged item (`CopernicusBrasilWeeklySchema`):

| Field | Type |
| --- | --- |
| `epiweek` | integer |
| `geocodigo` | integer |
| `temp_min_avg` | number |
| `temp_med_avg` | number |
| `temp_max_avg` | number |
| `temp_amplit_avg` | number |
| `precip_tot_sum` | number |
| `umid_min_avg` | number |
| `umid_med_avg` | number |
| `umid_max_avg` | number |

### GET `datastore/episcanner/`

Full URL: `https://api.mosqlimate.org/api/datastore/episcanner/`

Item (`EpiScannerSchema`), wrapped in `{"items": [...]}`:

| Field | Type |
| --- | --- |
| `disease` | string |
| `CID10` | string |
| `year` | integer |
| `geocode` | integer |
| `muni_name` | string |
| `peak_week` | number |
| `beta` | number |
| `gamma` | number |
| `R0` | number |
| `total_cases` | number |
| `alpha` | number |
| `sum_res` | number |
| `ep_ini` | string |
| `ep_end` | string |
| `ep_dur` | integer |

### GET `datastore/mosquito/`

Full URL: `https://api.mosqlimate.org/api/datastore/mosquito/`

Item (`ContaOvosSchema`), wrapped in `{"items": [...]}`:

| Field | Type | Notes |
| --- | --- | --- |
| `counting_id` | integer | |
| `date` | string | |
| `date_collect` | string \| null | |
| `eggs` | integer | |
| `latitude` | number | |
| `longitude` | number | |
| `municipality` | string | |
| `municipality_code` | string | |
| `ovitrap_id` | string | |
| `ovitrap_website_id` | integer | |
| `state_code` | string | |
| `state_name` | string | |
| `time` | string | |
| `week` | integer | |
| `year` | integer | |

### GET `datastore/vegetation/`

Full URL: `https://api.mosqlimate.org/api/datastore/vegetation/`

Paged item (`VegetationIndexMetricSchema`):

| Field | Type |
| --- | --- |
| `date` | date |
| `geocode` | integer |
| `collection` | string |
| `attribute` | string |
| `mean` | number \| null |
| `std` | number \| null |
| `median` | number \| null |
| `q25` | number \| null |
| `q75` | number \| null |
| `min` | number \| null |
| `max` | number \| null |

## Registry

### GET `registry/models/`

Full URL: `https://api.mosqlimate.org/api/registry/models/`

Paged item (`Model`):

| Field | Type | Notes |
| --- | --- | --- |
| `id` | integer | |
| `repository` | string | `owner/name`. |
| `description` | string \| null | Defaults to `""`. |
| `category` | string | Model category. |
| `time_resolution` | string | `day`, `week`, `month`, or `year`. |
| `imdc_year` | integer \| null | |
| `predictions_count` | integer | |
| `active` | boolean | |
| `created_at` | date | |
| `last_update` | date | |

Single model: `https://api.mosqlimate.org/api/registry/models/{owner}/{repository}/`
returns one `Model` object.

### GET `registry/predictions/`

Full URL: `https://api.mosqlimate.org/api/registry/predictions/`

Paged item (`Prediction`):

| Field | Type | Notes |
| --- | --- | --- |
| `id` | integer | |
| `model` | object | Nested `Model` object (see above). |
| `disease` | string | ICD-10 code. |
| `commit` | string | 40-character commit hash. |
| `description` | string \| null | Defaults to `""`. |
| `start` | date \| null | |
| `end` | date \| null | |
| `scores` | object | Metric names to numbers. |
| `case_definition` | string | `probable` or `reported`. |
| `published` | boolean | |
| `created_at` | date-time | |
| `adm_level` | integer | `0`-`3`. |
| `adm_0` | string \| null | Country code. |
| `adm_1` | integer \| null | State code. |
| `adm_2` | integer \| null | Municipality code. |
| `adm_3` | integer \| null | Sub-municipality code. |

The list endpoint does not include prediction rows. Use `--include-data` or the
data endpoint below.

### GET `registry/predictions/{id}/`

Full URL: `https://api.mosqlimate.org/api/registry/predictions/{id}/`

Returns a `PredictionDetail`: the same fields as `Prediction` plus:

| Field | Type | Notes |
| --- | --- | --- |
| `data` | array | Array of `PredictionData` rows. |

### GET `registry/predictions/{id}/data/`

Full URL: `https://api.mosqlimate.org/api/registry/predictions/{id}/data/`

Returns an array of `PredictionData` rows:

| Field | Type |
| --- | --- |
| `date` | date |
| `lower_95` | number \| null |
| `lower_90` | number |
| `lower_80` | number \| null |
| `lower_50` | number \| null |
| `pred` | number |
| `upper_50` | number \| null |
| `upper_80` | number \| null |
| `upper_90` | number |
| `upper_95` | number \| null |

### POST `registry/predictions/`

Full URL: `https://api.mosqlimate.org/api/registry/predictions/`

Request body (`PredictionIn`):

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `repository` | string | yes | `owner/name`. |
| `disease` | string | yes | ICD-10 code. |
| `description` | string | no | Defaults to `""`. |
| `commit` | string | yes | 40-character hash. |
| `case_definition` | string | no | `probable` (default) or `reported`. |
| `published` | boolean | no | API default is `true`; the skill sends `false` unless requested. |
| `adm_level` | integer | yes | `0`-`3`. |
| `adm_0` | string | no | Defaults to `BRA`. |
| `adm_1` | integer \| null | no | State code. |
| `adm_2` | integer \| null | no | Municipality code. |
| `adm_3` | integer \| null | no | Sub-municipality code. |
| `prediction` | array | yes | `PredictionDataRowSchema` rows. |

Each prediction row requires `date`, `pred`, `lower_95`, `lower_90`,
`lower_80`, `lower_50`, `upper_50`, `upper_80`, `upper_90`, and `upper_95`.

Response (`CreatePredictionOut`):

```json
{"id": 123}
```

The client then fetches the created prediction by id.

### DELETE `registry/predictions/{id}/`

Full URL: `https://api.mosqlimate.org/api/registry/predictions/{id}/`

Deletes the prediction. The API returns an object on success; `403` and `404`
responses use `{"message": string}`. The helper does not expose this operation.

### PATCH `registry/prediction/{prediction_id}/publish/`

Full URL:
`https://api.mosqlimate.org/api/registry/prediction/{prediction_id}/publish/`

Request body:

```json
{"published": true}
```

Response: `201` with a string body; `403` and `404` use
`{"message": string}`. The helper does not expose this operation.

### Validation

There is no separate validation endpoint. `validate-prediction` validates the
rows locally and checks the model with `GET registry/models/`. It does not write.
