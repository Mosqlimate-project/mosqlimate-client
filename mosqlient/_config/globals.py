import os

# Constants
DJANGO_APPS = ["registry", "datastore", "vis"]
DISEASES = ["chikungunya", "chik", "dengue", "zika"]
ADM_LEVELS = [0, 1, 2, 3]
TIME_RESOLUTIONS = ["day", "week", "month", "year"]
PREDICTION_DATA_COLUMNS = [
    "date",
    "pred",
    "lower",
    "upper",
    "adm_0",
    "adm_1",
    "adm_2",
    "adm_3",
]

API_URL = os.getenv("MOSQLIENT_API_URL", "https://api.mosqlimate.org/api/")
