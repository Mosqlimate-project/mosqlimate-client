# Constants
DJANGO_APPS = ["registry", "datastore", "vis"]
DISEASES = ["chikungunya", "dengue", "zika"]
ADM_LEVELS = [0, 1, 2, 3]
TIME_RESOLUTIONS = ["day", "week", "month", "year"]
PREDICTION_DATA_COLUMNS = [
    "dates",
    "preds",
    "lower",
    "upper",
    "adm_2",
    "adm_1",
    "adm_0",
]

API_DEV_URL = "http://0.0.0.0:8042/api/"
API_PROD_URL = "https://api.mosqlimate.org/api/"
