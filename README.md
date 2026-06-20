# Mosqlimate client

[![ci](https://github.com/Mosqlimate-project/mosqlimate-client/workflows/ci/badge.svg)](https://github.com/Mosqlimate-project/mosqlimate-client/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://api.mosqlimate.org/docs/)
[![pypi version](https://img.shields.io/pypi/v/mosqlient.svg)](https://pypi.org/project/mosqlient)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/mosqlimate-client/community)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Client library for the [Mosqlimate](https://api.mosqlimate.org/) data platform — an open platform for epidemiological forecasting of arboviruses (dengue, zika, chikungunya) in Brazil. It provides access to epidemiological and climate data, a model registry, forecasting tools, and scoring utilities.

## Requirements

Python 3.10 or above.

## Installation

```bash
pip install mosqlient
```

For forecasting and scoring features (ARIMA baseline model, ensemble model, scoring metrics, visualization):

```bash
pip install "mosqlient[analyze]"
```

## Authentication

All API calls require an API key in the format `username:uuid`. Create an account on the [Mosqlimate platform](https://api.mosqlimate.org/) to obtain your key.

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")  # e.g. "myuser:550e8400-e29b-41d4-a716-446655440000"
```

## Tutorial

### 1. Fetching epidemiological data

Query InfoDengue data for a specific city (geocode) or an entire state (UF):

```python
import mosqlient

# Dengue cases for Rio de Janeiro city (geocode=3304557)
df = mosqlient.get_infodengue(
    api_key=api_key,
    disease="A90",  # ICD-10 code for dengue
    start_date="2010-01-01",
    end_date="2023-12-31",
    geocode=3304557,
)

# All cities in Paraná state
df_pr = mosqlient.get_infodengue(
    api_key=api_key,
    disease="A90",
    start_date="2022-01-01",
    end_date="2023-01-01",
    uf="PR",
)
```

The returned DataFrame includes columns such as `data_iniSE` (epidemiological week start date), `SE` (week number), `casos` (reported cases), `casos_est` (estimated cases), `p_inc100k` (incidence per 100k), temperature, humidity, and more.

### 2. Fetching climate data

```python
df_climate = mosqlient.get_climate(
    api_key=api_key,
    start_date="2022-01-01",
    end_date="2023-01-01",
    geocode=4108304,  # Curitiba
)
```

### 3. Model registry

Browse and search forecast models registered on the platform. Model registration is done through the [Mosqlimate platform](https://api.mosqlimate.org/) (not via API).

```python
from mosqlient import get_all_models, get_models

# List all registered models
all_models = get_all_models(api_key)

# Search with filters
imdc_2024_models = get_models(api_key, imdc_year=2024)
dengue_models = get_models(api_key, disease="A90", adm_level=1)
```

Disease codes use ICD-10 format: `"A90"` (Dengue), `"A92.0"` (Chikungunya), `"A92.5"` (Zika).

**Upload predictions:**

```python
from mosqlient import upload_prediction

prediction_data = [
    {
        "date": "2024-10-06",
        "lower_95": 500, "lower_50": 1200, "pred": 1491,
        "upper_50": 1800, "upper_95": 4000,
    },
    # ... must contain all weeks in the date range without gaps
]

upload_prediction(
    api_key=api_key,
    repository="username/repository",  # format: owner/repo_name
    disease="A90",
    description="Out-of-sample forecast for Rio de Janeiro",
    commit="abc123",
    adm_level=2,
    case_definition="probable",
    adm_0="BRA",
    adm_2=3304557,  # geocode for Rio de Janeiro
    prediction=prediction_data,
)
```

For IMDC submissions, prediction data must contain all weeks in the date range without gaps:

```python
import pandas as pd
from epiweeks import Week

# Generate all required weeks for an IMDC submission
prediction_data = pd.date_range(
    start=Week(2023, 41).startdate(),  # Year-1 week 41
    end=Week(2024, 40).startdate(),    # Current year week 40
    freq="W-SUN",
).to_frame(name="date")
```

### 4. Building a baseline ARIMA forecast

*Requires `mosqlient[analyze]`.*

```python
import pandas as pd
from datetime import date
from mosqlient.datastore import Infodengue
from mosqlient.forecast import Arima

# Load and prepare data
df = Infodengue.get(
    api_key=api_key,
    disease="A90",
    start="2010-01-01",
    end=date.today().strftime("%Y-%m-%d"),
    geocode=3304557,
)
df = pd.DataFrame(df)
df["data_iniSE"] = pd.to_datetime(df["data_iniSE"])
df.set_index("data_iniSE", inplace=True)
df = df[["casos"]].rename(columns={"casos": "y"})
df = df.resample("W-SUN").sum()

# Train ARIMA model
model = Arima(df=df)
model.train(train_ini_date="2010-01-01", train_end_date="2021-12-31")

# In-sample predictions
df_in = model.predict_in_sample(plot=True)

# Out-of-sample forecast
df_out = model.forecast(horizon=8, plot=True, last_obs=10)
```

### 5. Scoring predictions

*Requires `mosqlient[analyze]`.*

Compare forecasts against observed data with multiple metrics (MAE, MSE, CRPS, Log Score, Interval Score, WIS):

```python
import pandas as pd
from mosqlient.scoring import Scorer

# df_true must have 'date' and 'casos' columns
scorer = Scorer(
    api_key=api_key,
    df_true=observed_data,
    ids=[77, 78],          # prediction IDs from the platform
    dist="log_normal",
    fn_loss="median",
    conf_level=0.90,
)

# Score summary table
print(scorer.summary)

# Filter by date range
scorer.set_date_range("2022-01-01", "2023-06-25")

# Visualize
scorer.plot_predictions()     # observed vs predicted
scorer.plot_mae()             # mean absolute error
scorer.plot_crps()            # CRPS over time
scorer.plot_wis()             # weighted interval score
```

### 6. Ensemble model

Combine multiple predictions via logarithmic pooling or linear mixture:

```python
from mosqlient.forecast import Ensemble
from mosqlient.prediction_optimize import get_df_pars
from mosqlient import get_prediction_by_id

# Fetch and parameterize a prediction
pred = get_prediction_by_id(api_key, id=300).to_dataframe()
pred_pars = get_df_pars(pred, dist="log_normal", fn_loss="lower")

# Create ensemble
ensemble = Ensemble(
    df=pred_pars,
    order_models=[15, 42, 78],
    mixture="log",
    dist="log_normal",
)

# Optimize weights against observations
weights = ensemble.compute_weights(df_obs=observed_data, metric="crps")

# Generate combined forecast
ensemble_df = ensemble.apply_ensemble()
```

## Using from R

Despite `mosqlient` being a Python library, it can be used from R via the `reticulate` package. See the example [R Jupyter notebook](docs/tutorials/Using%20Mosqlient%20from%20R.ipynb).

## Documentation

Full documentation, including detailed API reference and additional tutorials, is available at [api.mosqlimate.org/docs](https://api.mosqlimate.org/docs/).

## License

This project is licensed under the GPLv3 License — see the [LICENSE](LICENSE) file for details.
In the examples folder, you can find an [R jupyter notebook](docs/tutorials/Using%20Mosqlient%20from%20R.ipynb) of how to use `mosqlient` from R.
