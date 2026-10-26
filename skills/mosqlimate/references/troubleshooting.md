# Troubleshooting

## `uv: command not found`

Install `uv` from <https://docs.astral.sh/uv/> and retry. This skill declares
`uv` as its only required binary because the helper script uses PEP 723 inline
dependencies.

## Missing or invalid API key

- The key format is `username:uuid`.
- Set `MOSQLIMATE_API_KEY` or pass `--api-key`.
- Registry reads and every write require a real key.
- Without a key, `mosqlient` may create a temporary demo user. Demo access is
  rate-limited and not suitable for reliable automation.
- Never paste the key into a command that is logged or shared.

## HTTP 401 or 403

The key is missing, malformed, or not allowed to access the resource. Check the
username and UUID, then confirm the account on <https://api.mosqlimate.org/>.

## HTTP 422

The API rejected the parameters. Common causes:

- date format is not `YYYY-MM-DD` (or `YYYYWW` for weekly climate);
- date is before 2010 or in the future;
- geocode is not 7 digits;
- UF is not a valid two-letter code;
- disease name or ICD-10 code is wrong for the command;
- prediction rows have missing, extra, or out-of-order values.

Show the API message to the user and correct the parameters; do not retry the
same request.

## HTTP 429

The rate limit was reached. Wait and retry with backoff. Do not loop quickly.
Prefer narrower queries and smaller date ranges.

## Model not found during validation

- Check `--repository owner/name` against the registry.
- Model registration happens on the Mosqlimate website, not through this skill.
- Confirm the repository owner/organization and name with `models`.

## Prediction date or gap errors

- Weekly dates must be Sundays that start the CDC week.
- Every week must be present for IMDC submissions.
- Duplicate dates are rejected.
- Daily models must not skip days.

## Prediction bound errors

Bounds must be non-negative and ordered:

```text
lower_95 <= lower_90 <= lower_80 <= lower_50 <= pred
         <= upper_50 <= upper_80 <= upper_90 <= upper_95
```

## Wrong region data

- `--uf` is a state filter; `--geocode` is a municipality filter and takes
  precedence when both are provided.
- Municipality geocodes are 7-digit IBGE codes.
- Do not guess a geocode. Ask the user or look it up.

## Using a staging API

Set `MOSQLIENT_API_URL` or pass `--api-url`:

```bash
MOSQLIENT_API_URL=https://staging.example/api/ \
  uv run scripts/mosqlimate.py models --disease A90
```

`--api-url` must be set before `mosqlient` is imported; the helper handles that
automatically.

## Forecasting and scoring

The helper installs only the base `mosqlient` package. ARIMA, XGBoost, LSTM,
ensemble, and scoring workflows require `mosqlient[analyze]` and are outside
this skill's scope. Do not attempt them unless the dependency is installed.
